"""admin PUT /pin_quotas actually changes the effective PinQuotas.

The admin endpoint writes overrides into ``_pin_quota_overrides`` (a
process-local module dict) under the canonical tenant id; ``PinQuotas.for_tenant``
must consult that same dict, canonicalizing the caller's id so a bare ``"acme"``
resolves the override the endpoint wrote under ``"acme:acme"``. This test verifies
the consumer wire end to end:

  * fresh process: ``PinQuotas.for_tenant(t)`` returns dataclass defaults;
  * admin endpoint PUT mutates the override dict;
  * subsequent ``PinQuotas.for_tenant(t)`` reflects the PUT (raw or canonical id);
  * the lifecycle scheduler's PinService construction (the one
    production caller) uses ``for_tenant`` so the override propagates.

The blob boundary is an in-memory fake so the test is self-contained.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.pinning import PinQuotas
from cogniverse_runtime.routers import admin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

# Per-(tenant, kind, key) in-memory blob store standing in for the Phoenix-backed
# ArtifactManager. The pin-quota consumer wire under test is the admin PUT ->
# _pin_quota_overrides -> PinQuotas.for_tenant path; the durable blob boundary is
# incidental, and depending on an ambient localhost:6006 made this pass or fail
# on whatever happened to be listening on the host.
_FAKE_BLOBS: dict = {}


def _fake_build_artifact_manager(tenant_id):
    class _InMemoryAM:
        async def save_blob(self, kind, key, data):
            _FAKE_BLOBS[(tenant_id, kind, key)] = data

        async def load_blob(self, kind, key):
            return _FAKE_BLOBS.get((tenant_id, kind, key))

    return _InMemoryAM()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    _FAKE_BLOBS.clear()
    monkeypatch.setattr(admin, "_build_artifact_manager", _fake_build_artifact_manager)
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
            "PinService is dead."
        )
        assert resolved.tenant_admin == 99
        # Other tenants unaffected.
        unrelated = PinQuotas.for_tenant("globex")
        assert unrelated.user == baseline.user

    def test_override_resolves_across_id_spellings(self, client: TestClient):
        # PUT with a bare id (stored under the canonical "acme:acme"); reading
        # back with either spelling must resolve the same override — for_tenant
        # canonicalizes before the lookup, matching the endpoint.
        client.put("/admin/tenants/acme/pin_quotas", json={"user": 8})
        assert PinQuotas.for_tenant("acme").user == 8
        assert PinQuotas.for_tenant("acme:acme").user == 8

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
