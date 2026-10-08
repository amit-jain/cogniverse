"""A profile write reaches every runtime worker process before it answers.

Each worker process holds tenants' backend configs for up to the config
manager's staleness bound. The profile routes run here as the runtime serves
them, over the real config store and a real cluster-events channel on Redis;
the other worker is a separate process with its own ConfigManager on the same
store, subscribed to the channel as ``main.py`` subscribes a worker, holding
the tenant's profiles from before the write.
"""

from __future__ import annotations

import asyncio
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
from cogniverse_runtime.cluster_events import ClusterEvents
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
    events = ClusterEvents(
        redis_url,
        worker_id,
        {"backend_profiles_changed": admin.release_backend_profiles},
        channel=channel,
    )
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
    monkeypatch.setattr(admin, "_cluster_events", events)
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
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_runtime.routers import admin
from cogniverse_vespa.config.config_store import VespaConfigStore

async def main(redis_url, channel, port):
    manager = ConfigManager(store=VespaConfigStore(
        backend_url="http://localhost", backend_port=int(port)))
    events = ClusterEvents(redis_url, "other-worker", {
        "backend_profiles_changed": admin.release_backend_profiles},
        channel=channel)
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
        import json

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
            {"backend_profiles_changed": stuck},
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
            "runtime worker dropped the profiles it held; those workers read "
            "the change within a minute."
        )
        assert list(_stored(config_manager, tenant_id)) == ["late"]
        assert list(config_manager.list_backend_profiles(tenant_id)) == ["late"]

    async def test_an_unreachable_channel_fails_the_write_with_503(
        self, wired_admin, own_redis, config_manager, monkeypatch
    ):
        redis_url, pause, resume = own_redis
        events = await _route_worker(redis_url, _channel())
        monkeypatch.setattr(admin, "_cluster_events", events)
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
            f"Profile 'unheard' is stored for tenant '{tenant_id}', but not every "
            "runtime worker dropped the profiles it held; those workers read "
            "the change within a minute."
        )
        assert list(_stored(config_manager, tenant_id)) == ["unheard"]

    async def test_a_write_without_the_channel_wired_stores_nothing(
        self, wired_admin, config_manager, monkeypatch
    ):
        monkeypatch.setattr(admin, "_cluster_events", None)
        tenant_id = _tenant()
        async with _client() as c:
            with pytest.raises(RuntimeError) as raised:
                await c.post("/admin/profiles", json=_body(tenant_id, "nowhere"))

        assert str(raised.value) == (
            "Profile writes need the cluster events channel wired"
        )
        assert _stored(config_manager, tenant_id) == {}
