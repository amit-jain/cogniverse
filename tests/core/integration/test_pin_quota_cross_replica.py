"""A quota blob persisted by one replica is enforced by a cold second one."""

from __future__ import annotations

import asyncio
import json
from typing import Mapping

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_core.memory.pinning import PinQuotas
from cogniverse_runtime.routers import admin as admin_router

pytestmark = pytest.mark.integration

TENANT = "pinxrep:pinxrep"


@pytest.fixture
def real_artifact_backed_admin(telemetry_manager_with_phoenix, monkeypatch):
    provider = telemetry_manager_with_phoenix.get_provider(tenant_id=TENANT)
    monkeypatch.setattr(
        admin_router,
        "_build_artifact_manager",
        lambda key: ArtifactManager(provider, tenant_id=key),
    )
    admin_router._reset_admin_overrides_for_tests()
    yield
    admin_router._reset_admin_overrides_for_tests()


@pytest.mark.asyncio
async def test_cold_reader_enforces_the_persisted_blob(real_artifact_backed_admin):
    key = admin_router.canonical_tenant_id(TENANT)
    await admin_router._build_artifact_manager(key).save_blob(
        admin_router._PIN_QUOTA_BLOB_KIND,
        admin_router._PIN_QUOTA_BLOB_KEY,
        json.dumps({"user": 3, "tenant_admin": 7, "org_admin": -1}),
    )
    admin_router._reset_admin_overrides_for_tests()

    overrides = await admin_router._load_pin_quotas(TENANT)
    quotas = PinQuotas.for_tenant(TENANT, admin_overrides=overrides)

    assert (quotas.user, quotas.tenant_admin, quotas.org_admin) == (3, 7, None)


@pytest.mark.asyncio
async def test_lifecycle_pin_lookup_uses_the_same_loader(real_artifact_backed_admin):
    from cogniverse_runtime.main import build_pin_lookup

    key = admin_router.canonical_tenant_id(TENANT)
    await admin_router._build_artifact_manager(key).save_blob(
        admin_router._PIN_QUOTA_BLOB_KIND,
        admin_router._PIN_QUOTA_BLOB_KEY,
        json.dumps({"user": 2, "tenant_admin": 4, "org_admin": -1}),
    )
    admin_router._reset_admin_overrides_for_tests()

    seen: list[Mapping[str, int]] = []

    def loader(tenant_id: str):
        loaded = asyncio.run_coroutine_threadsafe(
            admin_router._load_pin_quotas(tenant_id), asyncio.get_event_loop()
        ).result(timeout=10)
        seen.append(loaded)
        return loaded

    pin_lookup = build_pin_lookup(_RecordingRegistry(), loader)
    await asyncio.to_thread(pin_lookup, _FakeManager(TENANT))

    assert seen == [{"user": 2, "tenant_admin": 4, "org_admin": -1}]


@pytest.mark.asyncio
async def test_quota_store_outage_raises_instead_of_enforcing_defaults(
    telemetry_manager_with_phoenix, monkeypatch
):
    """A dead store must not silently degrade every tenant to defaults."""

    class _DeadManager:
        async def load_blob(self, kind, key):
            raise ConnectionError("phoenix unreachable")

    monkeypatch.setattr(
        admin_router, "_build_artifact_manager", lambda key: _DeadManager()
    )
    admin_router._reset_admin_overrides_for_tests()

    with pytest.raises(ConnectionError, match="phoenix unreachable"):
        await admin_router._load_pin_quotas(TENANT)


@pytest.mark.asyncio
async def test_two_concurrent_cold_readers_agree_on_the_persisted_value(
    real_artifact_backed_admin,
):
    key = admin_router.canonical_tenant_id(TENANT)
    await admin_router._build_artifact_manager(key).save_blob(
        admin_router._PIN_QUOTA_BLOB_KIND,
        admin_router._PIN_QUOTA_BLOB_KEY,
        json.dumps({"user": 5, "tenant_admin": 9, "org_admin": -1}),
    )
    admin_router._reset_admin_overrides_for_tests()

    barrier = asyncio.Barrier(2)

    async def read():
        await barrier.wait()
        return await admin_router._load_pin_quotas(TENANT)

    first, second = await asyncio.gather(read(), read())

    assert first == second == {"user": 5, "tenant_admin": 9, "org_admin": -1}


class _RecordingRegistry:
    def get_schema(self, *args, **kwargs):
        return None


class _FakeManager:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id
        self.memory = object()

    def list_memories(self, *args, **kwargs):
        return []
