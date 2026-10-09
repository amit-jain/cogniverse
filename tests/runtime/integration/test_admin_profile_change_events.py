"""A profile write reaches every runtime and ingestion worker before it
answers.

Each worker process holds tenants' backend configs for up to the config
manager's staleness bound. The profile routes run here as the runtime serves
them, over the real config store and a real config events channel on Redis.
The other runtime worker is a separate process with its own ConfigManager on
the same store, subscribed to the channel as ``main.py`` subscribes a worker;
the ingestion worker is a separate process subscribed through the ingestion
worker's own subscription with the ConfigManager a job builds, or the real
``cogniverse_runtime.ingestion_worker`` process. Each holds the tenant's
profiles from before the write.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.cluster_events import (
    BACKEND_PROFILES_CHANGED,
    CONFIG_EVENT_CHANNEL,
    CONFIG_EVENT_HANDLERS,
    ClusterEvents,
)
from cogniverse_runtime.routers import admin
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


def _channel() -> str:
    return f"cogniverse:test-profile-events:{uuid.uuid4().hex[:8]}"


def _tenant() -> str:
    return f"profileevents{uuid.uuid4().hex[:8]}:main"


def _body(tenant_id: str, name: str, description: str = "") -> dict:
    return {
        "profile_name": name,
        "tenant_id": tenant_id,
        "type": "video",
        "description": description,
        "schema_name": "video_colpali_smol500_mv_frame",
        "embedding_model": "vidore/colsmol-500m",
        "embedding_type": "multi_vector",
        "model_loader": "colpali",
        "deploy_schema": False,
    }


def _stored(config_manager: ConfigManager, tenant_id: str) -> dict:
    entry = config_manager.store.get_config(
        tenant_id, ConfigScope.BACKEND, "backend", "backend_config"
    )
    return {} if entry is None else entry.config_value["profiles"]


@pytest.fixture
def config_manager(vespa_instance):
    manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        )
    )
    manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        )
    )
    return manager


async def _route_worker(redis_url: str, channel: str, worker_id: str = "route-worker"):
    """The worker process serving the routes: subscribed as main.py wires it."""
    events = ClusterEvents(redis_url, worker_id, CONFIG_EVENT_HANDLERS, channel=channel)
    await events.start()
    return events


@pytest.fixture
def wired_admin(config_manager):
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(FilesystemSchemaLoader(SCHEMAS_DIR))
    admin.set_profile_validator_schema_dir(SCHEMAS_DIR)
    yield
    admin.reset_dependencies()
    BackendRegistry.get_instance().clear_instances()


@pytest.fixture
async def channel(shared_state_redis_url, wired_admin, monkeypatch):
    name = _channel()
    events = await _route_worker(shared_state_redis_url, name)
    monkeypatch.setattr(admin, "_config_events", events)
    yield name
    await events.close()


def _client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://profile-events"
    )


_OTHER_WORKER = """
import asyncio, json, sys
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.cluster_events import CONFIG_EVENT_HANDLERS, ClusterEvents
from cogniverse_vespa.config.config_store import VespaConfigStore

async def main(redis_url, channel, port):
    manager = ConfigManager(store=VespaConfigStore(
        backend_url="http://localhost", backend_port=int(port)))
    events = ClusterEvents(
        redis_url, "other-worker", CONFIG_EVENT_HANDLERS, channel=channel)
    await events.start()
    print("ready", flush=True)
    loop = asyncio.get_running_loop()
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        held = await asyncio.to_thread(
            lambda: {
                name: profile.description
                for name, profile in manager.list_backend_profiles(
                    line.strip()).items()
            })
        print(json.dumps(held, sort_keys=True), flush=True)
    await events.close()

asyncio.run(main(*sys.argv[1:]))
"""


class _OtherWorker:
    """A worker process answering, per tenant, the profiles it serves."""

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

    async def profiles(self, tenant_id: str) -> dict:
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
class TestEveryWorkerServesTheProfileWrite:
    async def test_a_created_profile_is_served_by_the_other_worker_next(
        self, other_worker
    ):
        tenant_id = _tenant()
        assert await other_worker.profiles(tenant_id) == {}

        async with _client() as c:
            created = await c.post("/admin/profiles", json=_body(tenant_id, "made"))

        assert created.status_code == 201, created.text
        assert await other_worker.profiles(tenant_id) == {"made": ""}

    async def test_an_updated_profile_is_served_by_the_other_worker_next(
        self, other_worker
    ):
        tenant_id = _tenant()
        async with _client() as c:
            created = await c.post(
                "/admin/profiles", json=_body(tenant_id, "edited", "before")
            )
            assert created.status_code == 201, created.text
            assert await other_worker.profiles(tenant_id) == {"edited": "before"}

            updated = await c.put(
                "/admin/profiles/edited",
                json={"tenant_id": tenant_id, "description": "after"},
            )

        assert updated.status_code == 200, updated.text
        assert await other_worker.profiles(tenant_id) == {"edited": "after"}

    async def test_a_deleted_profile_is_no_longer_served_by_the_other_worker(
        self, other_worker
    ):
        tenant_id = _tenant()
        async with _client() as c:
            created = await c.post("/admin/profiles", json=_body(tenant_id, "gone"))
            assert created.status_code == 201, created.text
            assert await other_worker.profiles(tenant_id) == {"gone": ""}

            deleted = await c.delete(
                "/admin/profiles/gone", params={"tenant_id": tenant_id}
            )

        assert deleted.status_code == 200, deleted.text
        assert await other_worker.profiles(tenant_id) == {}


@pytest.mark.asyncio
class TestProfileWriteConcurrency:
    async def test_eight_concurrent_creates_each_reach_the_other_worker(
        self, other_worker
    ):
        """The other worker holds eight tenants with no profile; eight creates
        in flight together, each publishing its own event, leave it serving
        every one."""
        tenants = [_tenant() for _ in range(8)]
        for tenant_id in tenants:
            assert await other_worker.profiles(tenant_id) == {}

        async with _client() as c:
            created = await asyncio.gather(
                *(
                    c.post("/admin/profiles", json=_body(t, f"p{i}"))
                    for i, t in enumerate(tenants)
                )
            )

        assert [r.status_code for r in created] == [201] * 8
        assert [await other_worker.profiles(t) for t in tenants] == [
            {f"p{i}": ""} for i in range(8)
        ]


@pytest.mark.asyncio
class TestProfileWriteFaultContract:
    async def test_a_worker_that_does_not_confirm_fails_the_write_with_503(
        self, shared_state_redis_url, channel, config_manager, monkeypatch
    ):
        """The profile is stored and the answering worker dropped what it
        held; the caller is told a worker did not."""
        tenant_id = _tenant()
        monkeypatch.setattr(admin, "PROFILE_CHANGE_ACK_TIMEOUT_S", 2.0)
        release = threading.Event()

        def stuck(payload):
            release.wait(30)
            return {}

        silent = ClusterEvents(
            shared_state_redis_url,
            "silent-worker",
            {BACKEND_PROFILES_CHANGED: stuck},
            channel=channel,
        )
        await silent.start()
        try:
            assert config_manager.list_backend_profiles(tenant_id) == {}
            async with _client() as c:
                created = await c.post("/admin/profiles", json=_body(tenant_id, "late"))
        finally:
            release.set()
            await silent.close()

        assert created.status_code == 503
        assert created.json()["detail"]["message"] == (
            f"Profile 'late' is stored for tenant '{tenant_id}', but not every "
            "runtime or ingestion worker dropped the profiles it held; those "
            "workers read the change within a minute."
        )
        assert list(_stored(config_manager, tenant_id)) == ["late"]
        assert list(config_manager.list_backend_profiles(tenant_id)) == ["late"]

    async def test_an_unreachable_channel_fails_the_write_with_503(
        self, wired_admin, own_redis, config_manager, monkeypatch
    ):
        redis_url, pause, resume = own_redis
        events = await _route_worker(redis_url, _channel())
        monkeypatch.setattr(admin, "_config_events", events)
        tenant_id = _tenant()
        try:
            async with _client() as c:
                pause()
                try:
                    created = await c.post(
                        "/admin/profiles", json=_body(tenant_id, "unheard")
                    )
                finally:
                    resume()
        finally:
            await events.close()

        assert created.status_code == 503
        assert created.json()["detail"]["message"] == (
            f"Profile 'unheard' is stored for tenant '{tenant_id}', but not "
            "every runtime or ingestion worker dropped the profiles it held; "
            "those workers read the change within a minute."
        )
        assert list(_stored(config_manager, tenant_id)) == ["unheard"]

    async def test_a_delete_redis_cannot_carry_answers_503_and_stays_deleted(
        self, wired_admin, own_redis, config_manager, monkeypatch
    ):
        """Redis fails between the store's delete and the event: the delete
        is not undone and not half-done, and the caller is told which workers
        may still hold the profile."""
        redis_url, pause, resume = own_redis
        events = await _route_worker(redis_url, _channel())
        monkeypatch.setattr(admin, "_config_events", events)
        tenant_id = _tenant()
        try:
            async with _client() as c:
                for name in ("kept", "dropped"):
                    created = await c.post(
                        "/admin/profiles", json=_body(tenant_id, name)
                    )
                    assert created.status_code == 201, created.text
                pause()
                try:
                    deleted = await c.delete(
                        "/admin/profiles/dropped", params={"tenant_id": tenant_id}
                    )
                finally:
                    resume()
        finally:
            await events.close()

        assert deleted.status_code == 503
        assert deleted.json()["detail"]["message"] == (
            f"Profile 'dropped' is deleted for tenant '{tenant_id}', but not "
            "every runtime or ingestion worker dropped the profiles it held; "
            "those workers read the change within a minute."
        )
        assert list(_stored(config_manager, tenant_id)) == ["kept"]
        assert list(config_manager.list_backend_profiles(tenant_id)) == ["kept"]

    async def test_a_write_without_the_channel_wired_stores_nothing(
        self, wired_admin, config_manager, monkeypatch
    ):
        monkeypatch.setattr(admin, "_config_events", None)
        tenant_id = _tenant()
        async with _client() as c:
            with pytest.raises(RuntimeError) as raised:
                await c.post("/admin/profiles", json=_body(tenant_id, "nowhere"))

        assert str(raised.value) == (
            "Profile writes need the config events channel wired"
        )
        assert _stored(config_manager, tenant_id) == {}


_INGESTION_WORKER = """
import asyncio, json, sys, threading
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.ingestion_worker.worker import config_event_subscriber
from cogniverse_sdk.interfaces.config_store import ConfigScope

async def main(redis_url):
    manager = create_default_config_manager()
    backend_reads = []
    hold, held, release = threading.Event(), threading.Event(), threading.Event()
    stored = manager._stored_config_value

    def counted(scope, tenant_id, service, config_key):
        value = stored(scope, tenant_id, service, config_key)
        if scope == ConfigScope.BACKEND:
            backend_reads.append(tenant_id)
            if hold.is_set():
                hold.clear()
                held.set()
                release.wait(60)
        return value

    manager._stored_config_value = counted
    events = config_event_subscriber(redis_url, "profile-events")
    await events.start()
    print("ready", flush=True)
    loop = asyncio.get_running_loop()

    def profiles(tenant_id):
        return {
            name: profile.description
            for name, profile in manager.list_backend_profiles(tenant_id).items()
        }

    in_flight = None
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        command, tenant_id = line.split()
        if command == "profiles":
            answer = await asyncio.to_thread(profiles, tenant_id)
        elif command == "hold":
            hold.set()
            in_flight = loop.run_in_executor(None, profiles, tenant_id)
            answer = await asyncio.to_thread(held.wait, 60)
        elif command == "release":
            release.set()
            answer = await in_flight
        else:
            answer = backend_reads.count(tenant_id)
        print(json.dumps(answer, sort_keys=True), flush=True)
    await events.close()

asyncio.run(main(*sys.argv[1:]))
"""


class _IngestionWorker:
    """An ingestion worker process: the worker's own config events
    subscription and the ConfigManager a job builds. It answers, per tenant,
    the profiles it serves, and can hold one backend config read after the
    store answered it, until told to release it."""

    def __init__(self, redis_url: str, vespa_port: int):
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _INGESTION_WORKER, redis_url],
            env=dict(
                os.environ,
                BACKEND_URL="http://localhost",
                BACKEND_PORT=str(vespa_port),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = self._proc.stdout.readline().strip()
        if ready != "ready":
            self.close()
            raise AssertionError(f"ingestion worker did not subscribe: {ready!r}")

    async def _ask(self, command: str, tenant_id: str):
        def ask():
            self._proc.stdin.write(f"{command} {tenant_id}\n")
            self._proc.stdin.flush()
            return json.loads(self._proc.stdout.readline())

        return await asyncio.to_thread(ask)

    async def profiles(self, tenant_id: str) -> dict:
        return await self._ask("profiles", tenant_id)

    async def hold_read(self, tenant_id: str) -> bool:
        """Start a read of the tenant's profiles; True once the store has
        answered it and it waits to be cached."""
        return await self._ask("hold", tenant_id)

    async def release_read(self) -> dict:
        """The profiles the held read returns to its caller."""
        return await self._ask("release", "-")

    async def backend_reads(self, tenant_id: str) -> int:
        """How many times this process read the tenant's backend config from
        the store."""
        return await self._ask("reads", tenant_id)

    def close(self) -> None:
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


@pytest.fixture
async def ingestion_channel(own_redis, wired_admin, monkeypatch):
    """The routes' config events channel on a Redis this test alone uses, so
    the ingestion worker's fixed channel name reaches no other test."""
    redis_url, _pause, _resume = own_redis
    events = await _route_worker(redis_url, CONFIG_EVENT_CHANNEL)
    monkeypatch.setattr(admin, "_config_events", events)
    yield redis_url
    await events.close()


@pytest.fixture
def ingestion_holder(ingestion_channel, vespa_instance):
    worker = _IngestionWorker(ingestion_channel, vespa_instance["http_port"])
    yield worker
    worker.close()


@pytest.mark.asyncio
class TestIngestionWorkerServesTheProfileWrite:
    async def test_a_created_profile_is_served_by_the_ingestion_worker_next(
        self, ingestion_holder
    ):
        tenant_id = _tenant()
        assert await ingestion_holder.profiles(tenant_id) == {}

        async with _client() as c:
            created = await c.post("/admin/profiles", json=_body(tenant_id, "made"))

        assert created.status_code == 201, created.text
        assert await ingestion_holder.profiles(tenant_id) == {"made": ""}

    async def test_an_updated_profile_is_served_by_the_ingestion_worker_next(
        self, ingestion_holder
    ):
        tenant_id = _tenant()
        async with _client() as c:
            created = await c.post(
                "/admin/profiles", json=_body(tenant_id, "edited", "before")
            )
            assert created.status_code == 201, created.text
            assert await ingestion_holder.profiles(tenant_id) == {"edited": "before"}

            updated = await c.put(
                "/admin/profiles/edited",
                json={"tenant_id": tenant_id, "description": "after"},
            )

        assert updated.status_code == 200, updated.text
        assert await ingestion_holder.profiles(tenant_id) == {"edited": "after"}

    async def test_a_deleted_profile_is_no_longer_served_by_the_ingestion_worker(
        self, ingestion_holder
    ):
        tenant_id = _tenant()
        async with _client() as c:
            created = await c.post("/admin/profiles", json=_body(tenant_id, "gone"))
            assert created.status_code == 201, created.text
            assert await ingestion_holder.profiles(tenant_id) == {"gone": ""}

            deleted = await c.delete(
                "/admin/profiles/gone", params={"tenant_id": tenant_id}
            )

        assert deleted.status_code == 200, deleted.text
        assert await ingestion_holder.profiles(tenant_id) == {}


@pytest.mark.asyncio
class TestIngestionWorkerReadInFlight:
    async def test_a_read_in_flight_across_an_update_is_not_kept(
        self, ingestion_holder
    ):
        """The ingestion worker's read of the profiles has its answer from
        the store when the update is stored and published: the update
        answers once the worker dropped what it holds, the held read returns
        the profiles it read, and the worker's next read goes to the store
        and serves the update."""
        tenant_id = _tenant()
        async with _client() as c:
            created = await c.post(
                "/admin/profiles", json=_body(tenant_id, "raced", "before")
            )
            assert created.status_code == 201, created.text

            assert await ingestion_holder.hold_read(tenant_id) is True
            updated = await c.put(
                "/admin/profiles/raced",
                json={"tenant_id": tenant_id, "description": "after"},
            )
            assert updated.status_code == 200, updated.text
            assert await ingestion_holder.release_read() == {"raced": "before"}

        assert await ingestion_holder.profiles(tenant_id) == {"raced": "after"}
        assert await ingestion_holder.backend_reads(tenant_id) == 2


@pytest.mark.asyncio
class TestTheIngestionWorkerProcess:
    async def test_the_ingestion_worker_confirms_every_profile_write(
        self, ingestion_worker, wired_admin, config_manager, monkeypatch
    ):
        """Every profile write answers only once the ingestion worker process
        confirmed dropping the profiles it held."""
        redis_url, process, consumer_id = ingestion_worker
        events = await _route_worker(redis_url, CONFIG_EVENT_CHANNEL)
        monkeypatch.setattr(admin, "_config_events", events)
        tenant_id = _tenant()
        try:
            async with _client() as c:
                created = await c.post(
                    "/admin/profiles", json=_body(tenant_id, "worked", "before")
                )
                updated = await c.put(
                    "/admin/profiles/worked",
                    json={"tenant_id": tenant_id, "description": "after"},
                )
                stored = _stored(config_manager, tenant_id)
                deleted = await c.delete(
                    "/admin/profiles/worked", params={"tenant_id": tenant_id}
                )
            answers = await events.publish(
                BACKEND_PROFILES_CHANGED, {"tenant_id": tenant_id}, timeout_s=15
            )
        finally:
            await events.close()

        assert [created.status_code, updated.status_code, deleted.status_code] == [
            201,
            200,
            200,
        ], (created.text, updated.text, deleted.text)
        assert stored["worked"]["description"] == "after"
        assert _stored(config_manager, tenant_id) == {}
        worker = f"ingestion:{consumer_id}:{process.pid}:"
        assert sorted(
            ("ingestion" if name.startswith(worker) else name, answer["tenant_id"])
            for name, answer in answers.items()
        ) == [("ingestion", tenant_id), ("route-worker", tenant_id)]

    async def test_a_write_the_ingestion_worker_does_not_confirm_answers_503(
        self, ingestion_worker, wired_admin, config_manager, monkeypatch
    ):
        """The profile is stored and the serving worker reads it; the caller
        is told the ingestion worker did not confirm, and the next write
        once it answers again goes through."""
        redis_url, process, _consumer_id = ingestion_worker
        monkeypatch.setattr(admin, "PROFILE_CHANGE_ACK_TIMEOUT_S", 2.0)
        events = await _route_worker(redis_url, CONFIG_EVENT_CHANNEL)
        monkeypatch.setattr(admin, "_config_events", events)
        tenant_id = _tenant()
        try:
            os.kill(process.pid, signal.SIGSTOP)
            try:
                async with _client() as c:
                    stalled = await c.post(
                        "/admin/profiles", json=_body(tenant_id, "stalled")
                    )
            finally:
                os.kill(process.pid, signal.SIGCONT)
            async with _client() as c:
                resumed = await c.put(
                    "/admin/profiles/stalled",
                    json={"tenant_id": tenant_id, "description": "resumed"},
                )
        finally:
            await events.close()

        assert stalled.status_code == 503
        assert stalled.json()["detail"] == {
            "error": "profile_change_not_propagated",
            "message": (
                f"Profile 'stalled' is stored for tenant '{tenant_id}', but not "
                "every runtime or ingestion worker dropped the profiles it held; "
                "those workers read the change within a minute."
            ),
            "failure": "ClusterEventIncomplete",
            "profile_name": "stalled",
            "tenant_id": tenant_id,
        }
        assert resumed.status_code == 200, resumed.text
        assert {
            name: profile["description"]
            for name, profile in _stored(config_manager, tenant_id).items()
        } == {"stalled": "resumed"}
