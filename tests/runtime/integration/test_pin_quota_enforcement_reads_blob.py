"""Pin-quota enforcement must read the stored config record.

Enforcement resolves quotas through PinQuotas.for_tenant from the record
``_load_pin_quotas`` reads. A process that answers from anything it holds in
memory enforces a stale or default quota after another process's PUT. These
drive the REAL admin routes against a REAL Vespa config store — a PUT stores
the record, a reader wired to another store session reads it back, and the
resolved quotas equal the exact stored values.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.memory.pinning import PinQuotas
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin as admin_router
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


@pytest.fixture
def tenant() -> str:
    name = f"pinqenforce{uuid.uuid4().hex[:8]}"
    return f"{name}:{name}"


def _session(vespa_instance) -> VespaConfigStore:
    return VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )


@pytest.fixture
def admin_on_vespa(vespa_instance):
    """This process's admin router on its own real config-store session."""
    previous = admin_router._config_manager
    store = _session(vespa_instance)
    admin_router.set_config_manager(ConfigManager(store=store))
    yield store
    admin_router.set_config_manager(previous)
    store.close()


@pytest.fixture
def peer(vespa_instance):
    """The admin routes as another process serves them: its own session."""
    store = _session(vespa_instance)
    yield store
    store.close()


def _record(store: VespaConfigStore, tenant: str):
    return store.get_config(tenant, ConfigScope.SYSTEM, "admin_overrides", "pin_quotas")


@pytest.mark.asyncio
async def test_cold_replica_enforces_stored_quotas(admin_on_vespa, peer, tenant):
    peer.set_config(
        tenant,
        ConfigScope.SYSTEM,
        "admin_overrides",
        "pin_quotas",
        {"user": 3, "tenant_admin": 7, "org_admin": -1},
    )

    loaded = await admin_router._load_pin_quotas(tenant)

    quotas = PinQuotas.for_tenant(tenant, admin_overrides=loaded)
    assert loaded == {"user": 3, "tenant_admin": 7, "org_admin": -1}
    assert quotas.user == 3
    assert quotas.tenant_admin == 7
    assert quotas.org_admin is None  # -1 sentinel == unlimited


@pytest.mark.asyncio
async def test_defaults_when_no_record_stored(admin_on_vespa, tenant):
    loaded = await admin_router._load_pin_quotas(tenant)

    assert loaded == admin_router._default_pin_quotas()
    assert _record(admin_on_vespa, tenant) is None
    quotas = PinQuotas.for_tenant(tenant, admin_overrides=loaded)
    assert (quotas.user, quotas.tenant_admin, quotas.org_admin) == (
        50,
        500,
        None,
    )


@pytest.mark.asyncio
async def test_another_processs_put_is_enforced_on_the_next_read(
    admin_on_vespa, peer, tenant
):
    """Nothing is held in memory: a PUT stored by another process is what
    the very next enforcement here reads."""
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://admin"
    ) as client:
        first = await client.put(
            f"/admin/tenants/{tenant}/pin_quotas", json={"user": 1, "tenant_admin": 1}
        )
        assert first.json()["quotas"] == {"user": 1, "tenant_admin": 1, "org_admin": -1}
        assert (await admin_router._load_pin_quotas(tenant))["user"] == 1

        peer.update_config(
            tenant,
            ConfigScope.SYSTEM,
            "admin_overrides",
            "pin_quotas",
            lambda entry: {**entry.config_value, "user": 9},
        )

        assert (await admin_router._load_pin_quotas(tenant))["user"] == 9
        served = await client.get(f"/admin/tenants/{tenant}/pin_quotas")

    assert served.status_code == 200
    assert served.json() == {
        "tenant_id": tenant,
        "quotas": {"user": 9, "tenant_admin": 1, "org_admin": -1},
    }
    stored = _record(peer, tenant)
    assert (stored.version, stored.config_value) == (
        2,
        {"user": 9, "tenant_admin": 1, "org_admin": -1},
    )
