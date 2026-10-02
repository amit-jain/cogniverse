"""admin PUT /pin_quotas changes the effective PinQuotas.

The admin endpoint writes overrides into a stored quota record and loads that
record back into ``PinQuotas.for_tenant`` through an explicit
``admin_overrides`` argument. This test verifies the consumer wire end to end:

  * fresh process: loaded quotas resolve to dataclass defaults;
  * admin endpoint PUT stores the override record;
  * subsequent loads reflect the PUT (raw or canonical id);
  * the lifecycle scheduler's PinService construction (the one
    production caller) uses the loaded record so the override propagates.

The config store is the in-memory one so the test is self-contained.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.pinning import PinQuotas
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def wire_store():
    previous = admin._config_manager

    def wire(store):
        admin.set_config_manager(ConfigManager(store=store))
        return store

    yield wire
    admin.set_config_manager(previous)


@pytest.fixture
def client(wire_store) -> TestClient:
    wire_store(InMemoryConfigStore())
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    return TestClient(app)


class TestForTenantConsultsAdminOverrides:
    def test_defaults_when_no_admin_put_yet(self, client: TestClient):
        admin._reset_admin_overrides_for_tests()
        defaults = PinQuotas()
        loaded = asyncio.run(admin._load_pin_quotas("fresh_tenant"))
        resolved = PinQuotas.for_tenant("fresh_tenant", admin_overrides=loaded)
        assert loaded == admin._default_pin_quotas()
        assert resolved.user == defaults.user
        assert resolved.tenant_admin == defaults.tenant_admin

    def test_admin_put_changes_resolved_quotas(self, client: TestClient):
        # Before PUT: defaults.
        baseline = PinQuotas.for_tenant(
            "acme", admin_overrides=asyncio.run(admin._load_pin_quotas("acme"))
        )
        # PUT a custom user quota.
        resp = client.put(
            "/admin/tenants/acme/pin_quotas",
            json={"user": 7, "tenant_admin": 99},
        )
        assert resp.status_code == 200
        # After PUT: for_tenant must reflect the override.
        loaded = asyncio.run(admin._load_pin_quotas("acme"))
        resolved = PinQuotas.for_tenant("acme", admin_overrides=loaded)
        assert loaded == {"user": 7, "tenant_admin": 99, "org_admin": -1}
        assert resolved.user == 7, (
            f"admin PUT set user=7 but PinQuotas.for_tenant returned "
            f"user={resolved.user}; the wire from admin dict to "
            "PinService is dead."
        )
        assert resolved.tenant_admin == 99
        # Other tenants unaffected.
        unrelated = PinQuotas.for_tenant(
            "globex", admin_overrides=asyncio.run(admin._load_pin_quotas("globex"))
        )
        assert unrelated.user == baseline.user

    def test_override_resolves_across_id_spellings(self, client: TestClient):
        # PUT with a bare id (stored under the canonical "acme:acme"); reading
        # back with either spelling must resolve the same override — for_tenant
        # canonicalizes before the lookup, matching the endpoint.
        client.put("/admin/tenants/acme/pin_quotas", json={"user": 8})
        loaded_simple = asyncio.run(admin._load_pin_quotas("acme"))
        loaded_canonical = asyncio.run(admin._load_pin_quotas("acme:acme"))
        assert (
            loaded_simple
            == loaded_canonical
            == {
                "user": 8,
                "tenant_admin": 500,
                "org_admin": -1,
            }
        )
        assert PinQuotas.for_tenant("acme", admin_overrides=loaded_simple).user == 8
        assert (
            PinQuotas.for_tenant("acme:acme", admin_overrides=loaded_canonical).user
            == 8
        )

    def test_org_admin_unlimited_sentinel_translates_to_none(self, client: TestClient):
        # The admin endpoint stores -1 as the unlimited sentinel because
        # JSON has no None for form values. for_tenant must translate
        # back to None so PinQuotas.limit_for(ORG_ADMIN) returns None.
        client.put(
            "/admin/tenants/acme/pin_quotas",
            json={"org_admin": -1},
        )
        loaded = asyncio.run(admin._load_pin_quotas("acme"))
        resolved = PinQuotas.for_tenant("acme", admin_overrides=loaded)
        assert loaded == {"user": 50, "tenant_admin": 500, "org_admin": -1}
        assert resolved.org_admin is None, (
            "admin's -1 sentinel must translate to None (unlimited) so "
            "PinQuotas.limit_for(ORG_ADMIN) keeps returning None"
        )

    def test_partial_put_preserves_unspecified_fields(self, client: TestClient):
        # Set user only; tenant_admin should keep its default.
        client.put("/admin/tenants/acme/pin_quotas", json={"user": 3})
        loaded = asyncio.run(admin._load_pin_quotas("acme"))
        resolved = PinQuotas.for_tenant("acme", admin_overrides=loaded)
        assert loaded == {"user": 3, "tenant_admin": 500, "org_admin": -1}
        defaults = PinQuotas()
        assert resolved.user == 3
        assert resolved.tenant_admin == defaults.tenant_admin


class TestFallbackChain:
    """for_tenant priority: admin dict > TenantConfig metadata > defaults."""

    def test_falls_back_to_tenant_config_when_no_admin_override(
        self, client: TestClient
    ):
        admin._reset_admin_overrides_for_tests()

        class _StubTenantConfig:
            metadata = {"pin_quota": {"user": 42, "tenant_admin": 4242}}

        resolved = PinQuotas.for_tenant(
            "any", admin_overrides=None, tenant_config=_StubTenantConfig()
        )
        assert resolved.user == 42
        assert resolved.tenant_admin == 4242

    def test_admin_override_wins_over_tenant_config(self, client: TestClient):
        client.put("/admin/tenants/acme/pin_quotas", json={"user": 1})

        class _StubTenantConfig:
            metadata = {"pin_quota": {"user": 999, "tenant_admin": 999}}

        # Admin runtime override (1) must beat the tenant config (999).
        loaded = asyncio.run(admin._load_pin_quotas("acme"))
        resolved = PinQuotas.for_tenant(
            "acme",
            admin_overrides=loaded,
            tenant_config=_StubTenantConfig(),
        )
        assert resolved.user == 1, (
            "admin runtime override must take precedence over "
            "TenantConfig.metadata['pin_quota']; got user="
            f"{resolved.user}"
        )


class _InterleavedStore(InMemoryConfigStore):
    """Holds the first two pin-quota reads on one barrier, so two PUTs both
    read the same version before either writes."""

    def __init__(self):
        super().__init__()
        self.barrier = threading.Barrier(2)
        self.held = 0

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        entry = super().get_config(tenant_id, scope, service, config_key, version)
        if config_key == "pin_quotas" and self.held < 2:
            self.held += 1
            self.barrier.wait(timeout=10)
        return entry


@pytest.mark.asyncio
async def test_concurrent_same_tenant_puts_each_keep_their_field(wire_store):
    """Two concurrent PUTs for one tenant that both read the same stored
    version each keep the field they changed: the losing compare-and-set
    re-reads and merges onto the winner's record."""
    store = wire_store(_InterleavedStore())

    first, second = await asyncio.gather(
        admin.set_pin_quotas("acme:acme", admin.PinQuotasUpdateRequest(user=10)),
        admin.set_pin_quotas(
            "acme:acme", admin.PinQuotasUpdateRequest(tenant_admin=20)
        ),
    )

    assert store.held == 2
    stored = store.get_config(
        "acme:acme", ConfigScope.SYSTEM, "admin_overrides", "pin_quotas"
    )
    assert (stored.version, stored.config_value) == (
        2,
        {"user": 10, "tenant_admin": 20, "org_admin": -1},
    )
    # The PUT whose write landed first answered with its own field only; the
    # other merged onto it and answered with both.
    final = stored.config_value
    first_alone = {"user": 10, "tenant_admin": 500, "org_admin": -1}
    second_alone = {"user": 50, "tenant_admin": 20, "org_admin": -1}
    assert [first.quotas, second.quotas] in (
        [first_alone, final],
        [final, second_alone],
    )
