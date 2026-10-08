"""A config save, restore or import reaches every worker process before it
answers.

Each runtime worker process and each ingestion worker holds tenants' configs
for up to the config manager's staleness bound. The config routes run here as
the runtime serves them, over the real config store and a real config events
channel on Redis. The other worker is a separate process with its own
ConfigManager on the same store, subscribed to the channel as ``main.py``
subscribes a runtime worker, holding the configs it read before the write.
The ingestion worker is the real ``cogniverse_runtime.ingestion_worker``
process.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import httpx
import pytest
import redis
from fastapi import FastAPI

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.cluster_events import (
    CONFIG_EVENT_CHANNEL,
    CONFIGS_CHANGED,
    ClusterEvents,
    release_held_configs,
)
from cogniverse_runtime.routers import admin, config_entries
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration

ROUTING = ("routing", "gateway_agent", "routing_config")


def _channel() -> str:
    return f"cogniverse:test-config-events:{uuid.uuid4().hex[:8]}"


def _tenant() -> str:
    return f"configevents{uuid.uuid4().hex[:8]}:main"


@pytest.fixture
def config_manager(vespa_instance):
    manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        )
    )
    admin.set_config_manager(manager)
    yield manager
    admin.reset_dependencies()


async def _route_worker(redis_url: str, channel: str, worker_id: str = "route-worker"):
    """The worker process serving the routes: subscribed as main.py wires it."""
    events = ClusterEvents(
        redis_url,
        worker_id,
        {CONFIGS_CHANGED: release_held_configs},
        channel=channel,
    )
    await events.start()
    return events


@pytest.fixture
async def channel(shared_state_redis_url, config_manager, monkeypatch):
    name = _channel()
    events = await _route_worker(shared_state_redis_url, name)
    monkeypatch.setattr(config_entries, "_config_events", events)
    yield name
    await events.close()


def _client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(config_entries.router, prefix="/admin")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://config-events"
    )


async def _save_routing(client, tenant_id: str, value: dict, version: int):
    return await client.put(
        "/admin/config/sections/routing",
        json={"tenant_id": tenant_id, "value": value, "version": version},
    )


def _stored_version(config_manager: ConfigManager, tenant_id: str) -> int:
    entry = config_manager.store.get_config(
        tenant_id, ConfigScope.ROUTING, "gateway_agent", "routing_config"
    )
    return 0 if entry is None else entry.version


_OTHER_WORKER = """
import asyncio, json, sys
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.cluster_events import (
    CONFIGS_CHANGED, ClusterEvents, release_held_configs)
from cogniverse_vespa.config.config_store import VespaConfigStore

async def main(redis_url, channel, port):
    manager = ConfigManager(store=VespaConfigStore(
        backend_url="http://localhost", backend_port=int(port)))
    events = ClusterEvents(redis_url, "other-worker", {
        CONFIGS_CHANGED: release_held_configs}, channel=channel)
    await events.start()
    print("ready", flush=True)
    loop = asyncio.get_running_loop()

    def held(tenant):
        if tenant == "_system":
            return {"application_name": manager.get_system_config().application_name}
        routing = manager.get_routing_config(tenant)
        return {
            "routing_mode": routing.routing_mode,
            "min_unique_queries": routing.min_unique_queries,
        }

    while line := await loop.run_in_executor(None, sys.stdin.readline):
        answer = await asyncio.to_thread(held, line.strip())
        print(json.dumps(answer, sort_keys=True), flush=True)
    await events.close()

asyncio.run(main(*sys.argv[1:]))
"""


class _OtherWorker:
    """A worker process answering, per tenant, the routing config it serves
    (for ``_system``, the system config's application name)."""

    def __init__(self, redis_url: str, channel: str, vespa_port: int):
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _OTHER_WORKER, redis_url, channel, str(vespa_port)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = self._proc.stdout.readline().strip()
        if ready != "ready":
            self.close()
            raise AssertionError(f"other worker did not subscribe: {ready!r}")

    async def serves(self, tenant_id: str) -> dict:
        def ask() -> dict:
            self._proc.stdin.write(tenant_id + "\n")
            self._proc.stdin.flush()
            return json.loads(self._proc.stdout.readline())

        return await asyncio.to_thread(ask)

    def close(self) -> None:
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


@pytest.fixture
def other_worker(shared_state_redis_url, channel, vespa_instance):
    worker = _OtherWorker(shared_state_redis_url, channel, vespa_instance["http_port"])
    yield worker
    worker.close()


@pytest.mark.asyncio
class TestEveryWorkerReadsTheWrite:
    async def test_a_section_save_is_served_by_the_other_worker_next(
        self, other_worker
    ):
        tenant_id = _tenant()
        assert await other_worker.serves(tenant_id) == {
            "routing_mode": "tiered",
            "min_unique_queries": 3,
        }
        async with _client() as c:
            saved = await _save_routing(
                c, tenant_id, {"routing_mode": "direct", "min_unique_queries": 7}, 0
            )
        assert saved.status_code == 200, saved.text
        assert await other_worker.serves(tenant_id) == {
            "routing_mode": "direct",
            "min_unique_queries": 7,
        }

    async def test_a_system_save_is_served_by_the_other_worker_next(
        self, other_worker, config_manager
    ):
        before = config_manager.store.get_config(
            "_system", ConfigScope.SYSTEM, "system", "system_config"
        )
        held = await other_worker.serves("_system")
        renamed = f"config-events-{uuid.uuid4().hex[:6]}"
        try:
            async with _client() as c:
                saved = await c.put(
                    "/admin/config/sections/system",
                    json={
                        "value": {"application_name": renamed},
                        "version": before.version if before else 0,
                    },
                )
            assert saved.status_code == 200, saved.text
            assert held != {"application_name": renamed}
            assert await other_worker.serves("_system") == {"application_name": renamed}
        finally:
            config_manager.set_system_config(
                SystemConfig.from_dict(before.config_value)
                if before
                else SystemConfig()
            )

    async def test_a_restored_version_is_served_by_the_other_worker_next(
        self, other_worker
    ):
        tenant_id = _tenant()
        async with _client() as c:
            for version, mode in enumerate(["direct", "adaptive"]):
                saved = await _save_routing(
                    c, tenant_id, {"routing_mode": mode}, version
                )
                assert saved.status_code == 200, saved.text
            assert (await other_worker.serves(tenant_id))["routing_mode"] == "adaptive"
            restored = await c.post(
                "/admin/config/rollback",
                json={
                    "tenant_id": tenant_id,
                    "scope": ROUTING[0],
                    "service": ROUTING[1],
                    "config_key": ROUTING[2],
                    "version": 1,
                    "expected_version": 2,
                },
            )
        assert restored.status_code == 200, restored.text
        assert restored.json()["version"] == 3
        assert (await other_worker.serves(tenant_id))["routing_mode"] == "direct"

    async def test_an_import_is_served_by_the_other_worker_next(self, other_worker):
        source, target = _tenant(), _tenant()
        assert await other_worker.serves(target) == {
            "routing_mode": "tiered",
            "min_unique_queries": 3,
        }
        async with _client() as c:
            saved = await _save_routing(
                c, source, {"routing_mode": "adaptive", "min_unique_queries": 12}, 0
            )
            assert saved.status_code == 200, saved.text
            exported = await c.get("/admin/config/export", params={"tenant_id": source})
            imported = await c.post(
                "/admin/config/import",
                json={"tenant_id": target, "configs": exported.json()},
            )
        assert imported.json() == {"tenant_id": target, "imported": 1}
        assert await other_worker.serves(target) == {
            "routing_mode": "adaptive",
            "min_unique_queries": 12,
        }

    async def test_eight_concurrent_saves_each_reach_the_other_worker(
        self, other_worker
    ):
        tenants = [_tenant() for _ in range(8)]
        for tenant_id in tenants:
            assert (await other_worker.serves(tenant_id))["min_unique_queries"] == 3
        async with _client() as c:
            saved = await asyncio.gather(
                *(
                    _save_routing(c, tenant_id, {"min_unique_queries": 20 + i}, 0)
                    for i, tenant_id in enumerate(tenants)
                )
            )
        assert [response.status_code for response in saved] == [200] * len(tenants)
        served = [await other_worker.serves(tenant_id) for tenant_id in tenants]
        assert [s["min_unique_queries"] for s in served] == [
            20 + i for i in range(len(tenants))
        ]


@pytest.mark.asyncio
class TestConfigWriteFaultContract:
    async def test_a_worker_that_does_not_confirm_fails_the_write_with_503(
        self, shared_state_redis_url, channel, config_manager, monkeypatch
    ):
        """The save is stored and the answering worker dropped what it held;
        the caller is told a worker did not."""
        tenant_id = _tenant()
        monkeypatch.setattr(config_entries, "CONFIG_CHANGE_ACK_TIMEOUT_S", 2.0)
        release = threading.Event()

        def stuck(payload):
            release.wait(30)
            return {}

        silent = ClusterEvents(
            shared_state_redis_url,
            "silent-worker",
            {CONFIGS_CHANGED: stuck},
            channel=channel,
        )
        await silent.start()
        try:
            async with _client() as c:
                saved = await _save_routing(c, tenant_id, {"routing_mode": "direct"}, 0)
        finally:
            release.set()
            await silent.close()

        assert saved.status_code == 503
        assert saved.json()["detail"] == {
            "error": "config_change_not_propagated",
            "message": (
                f"The routing config version 1 is stored for {tenant_id}, but "
                "not every worker dropped the configs it held; those workers "
                "read it within a minute."
            ),
            "failure": "ClusterEventIncomplete",
            "tenant_id": tenant_id,
        }
        assert _stored_version(config_manager, tenant_id) == 1
        assert config_manager.get_routing_config(tenant_id).routing_mode == "direct"

    async def test_an_unreachable_channel_fails_the_restore_with_503(
        self, shared_state_redis_url, config_manager, own_redis, monkeypatch
    ):
        tenant_id = _tenant()
        working = await _route_worker(shared_state_redis_url, _channel())
        monkeypatch.setattr(config_entries, "_config_events", working)
        try:
            async with _client() as c:
                for version, mode in enumerate(["direct", "adaptive"]):
                    saved = await _save_routing(
                        c, tenant_id, {"routing_mode": mode}, version
                    )
                    assert saved.status_code == 200, saved.text
        finally:
            await working.close()

        redis_url, pause, resume = own_redis
        events = await _route_worker(redis_url, _channel())
        monkeypatch.setattr(config_entries, "_config_events", events)
        try:
            async with _client() as c:
                pause()
                try:
                    restored = await c.post(
                        "/admin/config/rollback",
                        json={
                            "tenant_id": tenant_id,
                            "scope": ROUTING[0],
                            "service": ROUTING[1],
                            "config_key": ROUTING[2],
                            "version": 1,
                            "expected_version": 2,
                        },
                    )
                finally:
                    resume()
        finally:
            await events.close()

        assert restored.status_code == 503
        assert restored.json()["detail"]["message"] == (
            "Version 1 of routing/gateway_agent/routing_config, restored as "
            f"version 3, is stored for {tenant_id}, but not every worker "
            "dropped the configs it held; those workers read it within a minute."
        )
        assert _stored_version(config_manager, tenant_id) == 3

    async def test_a_write_without_the_channel_wired_stores_nothing(
        self, config_manager, monkeypatch
    ):
        monkeypatch.setattr(config_entries, "_config_events", None)
        tenant_id = _tenant()
        async with _client() as c:
            with pytest.raises(RuntimeError) as raised:
                await _save_routing(c, tenant_id, {"routing_mode": "direct"}, 0)
            with pytest.raises(RuntimeError):
                await c.post(
                    "/admin/config/import",
                    json={"tenant_id": tenant_id, "configs": {"configs": []}},
                )

        assert str(raised.value) == (
            "Config writes need the config events channel wired"
        )
        assert _stored_version(config_manager, tenant_id) == 0


def _subscribers(redis_url: str) -> int:
    client = redis.Redis.from_url(redis_url)
    try:
        return dict(client.pubsub_numsub(CONFIG_EVENT_CHANNEL))[
            CONFIG_EVENT_CHANNEL.encode()
        ]
    finally:
        client.close()


@pytest.fixture
def ingestion_worker(own_redis, vespa_instance, tmp_path):
    """The ingestion worker process, as its pod runs it, subscribed to the
    config events channel of a Redis this test alone uses."""
    redis_url, _pause, _resume = own_redis
    consumer_id = f"config-events-{uuid.uuid4().hex[:6]}"
    log = tmp_path / "ingestion_worker.log"
    env = dict(
        os.environ,
        REDIS_URL=redis_url,
        BACKEND_URL="http://localhost",
        BACKEND_PORT=str(vespa_instance["http_port"]),
        INGEST_CONSUMER_ID=consumer_id,
        INGEST_REAPER_ENABLED="false",
        INGEST_CLAIM_BLOCK_MS="200",
        LOG_LEVEL="INFO",
        PYTHONUNBUFFERED="1",
    )
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "cogniverse_runtime.ingestion_worker.worker"],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 180
        while _subscribers(redis_url) != 1:
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(
                    f"ingestion worker did not subscribe:\n{log.read_text()}"
                )
            time.sleep(0.5)
        yield redis_url, process, consumer_id
    finally:
        os.kill(process.pid, signal.SIGCONT)
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.mark.asyncio
class TestIngestionWorkerDropsWhatItHeld:
    async def test_the_ingestion_worker_handles_every_config_change(
        self, ingestion_worker, config_manager, monkeypatch
    ):
        redis_url, process, consumer_id = ingestion_worker
        tenant_id = _tenant()
        events = await _route_worker(redis_url, CONFIG_EVENT_CHANNEL)
        monkeypatch.setattr(config_entries, "_config_events", events)
        try:
            async with _client() as c:
                saved = await _save_routing(c, tenant_id, {"routing_mode": "direct"}, 0)
            answers = await events.publish(
                CONFIGS_CHANGED, {"tenant_id": tenant_id}, timeout_s=15
            )
        finally:
            await events.close()

        assert saved.status_code == 200, saved.text
        worker = f"ingestion:{consumer_id}:{process.pid}:"
        assert sorted(
            ("ingestion" if name.startswith(worker) else name, answer["tenant_id"])
            for name, answer in answers.items()
        ) == [("ingestion", tenant_id), ("route-worker", tenant_id)]

    async def test_a_save_the_ingestion_worker_does_not_confirm_answers_503(
        self, ingestion_worker, config_manager, monkeypatch
    ):
        redis_url, process, _consumer_id = ingestion_worker
        tenant_id = _tenant()
        monkeypatch.setattr(config_entries, "CONFIG_CHANGE_ACK_TIMEOUT_S", 2.0)
        events = await _route_worker(redis_url, CONFIG_EVENT_CHANNEL)
        monkeypatch.setattr(config_entries, "_config_events", events)
        try:
            os.kill(process.pid, signal.SIGSTOP)
            try:
                async with _client() as c:
                    stalled = await _save_routing(
                        c, tenant_id, {"routing_mode": "direct"}, 0
                    )
            finally:
                os.kill(process.pid, signal.SIGCONT)
            async with _client() as c:
                resumed = await _save_routing(
                    c, tenant_id, {"routing_mode": "adaptive"}, 1
                )
        finally:
            await events.close()

        assert stalled.status_code == 503
        assert stalled.json()["detail"]["error"] == "config_change_not_propagated"
        assert resumed.status_code == 200, resumed.text
        assert _stored_version(config_manager, tenant_id) == 2
