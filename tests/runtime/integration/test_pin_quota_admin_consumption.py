"""admin PUT /pin_quotas actually changes the effective PinQuotas.

The previous fix-up shipped the admin endpoint that wrote into
``_pin_quota_overrides`` (a process-local module dict). The audit
caught that **PinService never reads that dict** — the writes were
black-holed. This test verifies the consumer wire:

  * fresh process: ``PinQuotas.for_tenant(t)`` returns dataclass defaults;
  * admin endpoint PUT mutates the override dict;
  * subsequent ``PinQuotas.for_tenant(t)`` reflects the PUT;
  * the lifecycle scheduler's PinService construction (the one
    production caller) uses ``for_tenant`` so the override propagates.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.pinning import PinQuotas
from cogniverse_runtime.routers import admin

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    admin._reset_admin_overrides_for_tests()
    return TestClient(app)


class TestForTenantConsultsAdminOverrides:
    def test_defaults_when_no_admin_put_yet(self, client: TestClient):
        admin._reset_admin_overrides_for_tests()
        defaults = PinQuotas()
        resolved = PinQuotas.for_tenant("fresh_tenant")
        assert resolved.user == defaults.user
        assert resolved.tenant_admin == defaults.tenant_admin

    def test_admin_put_changes_resolved_quotas(self, client: TestClient):
        # Before PUT: defaults.
        baseline = PinQuotas.for_tenant("acme")
        # PUT a custom user quota.
        resp = client.put(
            "/admin/tenants/acme/pin_quotas",
            json={"user": 7, "tenant_admin": 99},
        )
        assert resp.status_code == 200
        # After PUT: for_tenant must reflect the override.
        resolved = PinQuotas.for_tenant("acme")
        assert resolved.user == 7, (
            f"admin PUT set user=7 but PinQuotas.for_tenant returned "
            f"user={resolved.user}; the wire from admin dict to "
            "PinService is dead — this is the original audit gap."
        )
        assert resolved.tenant_admin == 99
        # Other tenants unaffected.
        unrelated = PinQuotas.for_tenant("globex")
        assert unrelated.user == baseline.user

    def test_org_admin_unlimited_sentinel_translates_to_none(self, client: TestClient):
        # The admin endpoint stores -1 as the unlimited sentinel because
        # JSON has no None for form values. for_tenant must translate
        # back to None so PinQuotas.limit_for(ORG_ADMIN) returns None.
        client.put(
            "/admin/tenants/acme/pin_quotas",
            json={"org_admin": -1},
        )
        resolved = PinQuotas.for_tenant("acme")
        assert resolved.org_admin is None, (
            "admin's -1 sentinel must translate to None (unlimited) so "
            "PinQuotas.limit_for(ORG_ADMIN) keeps returning None"
        )

    def test_partial_put_preserves_unspecified_fields(self, client: TestClient):
        # Set user only; tenant_admin should keep its default.
        client.put("/admin/tenants/acme/pin_quotas", json={"user": 3})
        resolved = PinQuotas.for_tenant("acme")
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

        resolved = PinQuotas.for_tenant("any", tenant_config=_StubTenantConfig())
        assert resolved.user == 42
        assert resolved.tenant_admin == 4242

    def test_admin_override_wins_over_tenant_config(self, client: TestClient):
        client.put("/admin/tenants/acme/pin_quotas", json={"user": 1})

        class _StubTenantConfig:
            metadata = {"pin_quota": {"user": 999, "tenant_admin": 999}}

        # Admin runtime override (1) must beat the tenant config (999).
        resolved = PinQuotas.for_tenant("acme", tenant_config=_StubTenantConfig())
        assert resolved.user == 1, (
            "admin runtime override must take precedence over "
            "TenantConfig.metadata['pin_quota']; got user="
            f"{resolved.user}"
        )
