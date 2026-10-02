"""A quota record stored by one process is enforced by a cold second one."""

from __future__ import annotations

import asyncio
import uuid
from typing import Mapping

import pytest

from cogniverse_core.memory.pinning import (
    PIN_AGENT_NAME,
    PIN_RECORD_KIND,
    PinQuotas,
)
from cogniverse_core.memory.schema import Pinnable
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin as admin_router
from cogniverse_sdk.interfaces.config_store import (
    ConfigScope,
    ConfigStoreUnavailableError,
)
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration

# Nothing listens here: the repo-wide dead-port convention.
DEAD_PORT = 29071


@pytest.fixture
def tenant() -> str:
    name = f"pinxrep{uuid.uuid4().hex[:8]}"
    return f"{name}:{name}"


@pytest.fixture
def peer_store(shared_vespa):
    """Another process's session on the same config store."""
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=shared_vespa["http_port"]
    )
    yield store
    store.close()


@pytest.fixture
def admin_on_vespa(shared_vespa):
    """This process's admin router wired to the real Vespa config store."""
    previous = admin_router._config_manager
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=shared_vespa["http_port"]
    )
    admin_router.set_config_manager(ConfigManager(store=store))
    yield store
    admin_router.set_config_manager(previous)
    store.close()


def _store_quotas(store: VespaConfigStore, tenant: str, quotas: dict) -> None:
    store.set_config(
        tenant, ConfigScope.SYSTEM, "admin_overrides", "pin_quotas", quotas
    )


@pytest.mark.asyncio
async def test_cold_reader_enforces_the_stored_record(
    admin_on_vespa, peer_store, tenant
):
    _store_quotas(peer_store, tenant, {"user": 3, "tenant_admin": 7, "org_admin": -1})

    overrides = await admin_router._load_pin_quotas(tenant)
    quotas = PinQuotas.for_tenant(tenant, admin_overrides=overrides)

    assert overrides == {"user": 3, "tenant_admin": 7, "org_admin": -1}
    assert (quotas.user, quotas.tenant_admin, quotas.org_admin) == (3, 7, None)


@pytest.mark.asyncio
async def test_lifecycle_pin_lookup_uses_the_same_loader(
    admin_on_vespa, peer_store, tenant
):
    from cogniverse_runtime.main import _PIN_QUOTA_LOAD_TIMEOUT_S, build_pin_lookup

    _store_quotas(peer_store, tenant, {"user": 2, "tenant_admin": 4, "org_admin": -1})

    seen: list[Mapping[str, int]] = []
    loop = asyncio.get_running_loop()

    def loader(tenant_id: str):
        loaded = asyncio.run_coroutine_threadsafe(
            admin_router._load_pin_quotas(tenant_id), loop
        ).result(timeout=_PIN_QUOTA_LOAD_TIMEOUT_S)
        seen.append(loaded)
        return loaded

    pin_lookup = build_pin_lookup(_RecordingRegistry(), loader)
    manager = _FakeManager(tenant)
    pinned = await asyncio.to_thread(pin_lookup, manager)

    assert seen == [{"user": 2, "tenant_admin": 4, "org_admin": -1}]
    assert manager.get_all_calls == [(tenant, PIN_AGENT_NAME, None)]
    assert pinned == {"mem_target_1"}


@pytest.mark.asyncio
async def test_quota_store_outage_raises_instead_of_enforcing_defaults(tenant):
    """A dead store must not silently degrade every tenant to defaults."""
    previous = admin_router._config_manager
    dead = VespaConfigStore(backend_url="http://localhost", backend_port=DEAD_PORT)
    admin_router.set_config_manager(ConfigManager(store=dead))
    try:
        with pytest.raises(ConfigStoreUnavailableError) as caught:
            await admin_router._load_pin_quotas(tenant)
    finally:
        admin_router.set_config_manager(previous)
        dead.close()

    assert str(caught.value).startswith(
        "Failed to read Vespa config visit after 5 attempts over "
    )


@pytest.mark.asyncio
async def test_two_concurrent_cold_readers_agree_on_the_stored_value(
    admin_on_vespa, peer_store, tenant
):
    _store_quotas(peer_store, tenant, {"user": 5, "tenant_admin": 9, "org_admin": -1})

    barrier = asyncio.Barrier(2)

    async def read():
        await barrier.wait()
        return await admin_router._load_pin_quotas(tenant)

    first, second = await asyncio.gather(read(), read())

    assert first == second == {"user": 5, "tenant_admin": 9, "org_admin": -1}


class _RecordingRegistry:
    def get_schema(self, *args, **kwargs):
        return None


class _FakeManager:
    """Mirrors the manager surface PinService reads.

    ``limit`` is required: retention treats an unlisted pin as absent and
    deletes its target, so the pin enumeration must walk the whole partition.
    """

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id
        self.memory = object()
        self.get_all_calls: list[tuple[str, str, int | None]] = []

    def get_all_memories(self, *, tenant_id: str, agent_name: str, limit):
        self.get_all_calls.append((tenant_id, agent_name, limit))
        return [
            {
                "id": "pin_rec_1",
                "metadata": {
                    "kind": PIN_RECORD_KIND,
                    "target_memory_id": "mem_target_1",
                    "pinned_by": Pinnable.USER.value,
                    "target_kind": "fact",
                    "pin_actor_id": "actor_1",
                },
            }
        ]
