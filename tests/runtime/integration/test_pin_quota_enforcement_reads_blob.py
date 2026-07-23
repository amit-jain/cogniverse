"""Pin-quota enforcement must read the durable Phoenix-backed blob.

Enforcement resolves quotas through PinQuotas.for_tenant, which reads a
process-local override cache. That cache is only warmed by a pin-quota GET/PUT
on the same replica, so a replica that never served one enforced hardcoded
defaults and ignored an admin PUT persisted by another replica. These drive the
REAL admin routes against a REAL Phoenix container (the store the pin-quota blob
persists through) — a PUT persists the blob, a cold reader reads it back, and
the resolved quotas equal the exact persisted values.
"""

from __future__ import annotations

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_core.memory.pinning import PinQuotas
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANT = "pinq-enforce:pinq-enforce"


@pytest.fixture
def real_artifact_backed_admin(telemetry_manager_with_phoenix, monkeypatch):
    """Point the admin router's artifact factory at a REAL ArtifactManager
    wired to the REAL Phoenix container (not a stub) and clear the process cache
    so each test starts from a cold replica."""
    provider = telemetry_manager_with_phoenix.get_provider(tenant_id=TENANT)

    def _factory(key: str):
        return ArtifactManager(provider, tenant_id=key)

    monkeypatch.setattr(admin_router, "_build_artifact_manager", _factory)
    admin_router._reset_admin_overrides_for_tests()
    yield
    admin_router._reset_admin_overrides_for_tests()


@pytest.mark.asyncio
async def test_cold_replica_enforces_persisted_quotas(real_artifact_backed_admin):
    # An admin PUT persists the quota blob to real Phoenix.
    key = admin_router.canonical_tenant_id(TENANT)
    am = admin_router._build_artifact_manager(key)
    import json

    await am.save_blob(
        admin_router._PIN_QUOTA_BLOB_KIND,
        admin_router._PIN_QUOTA_BLOB_KEY,
        json.dumps({"user": 3, "tenant_admin": 7, "org_admin": -1}),
    )

    # A cold replica (cache cleared) enforces a pin — it must read the persisted
    # blob, not fall back to hardcoded defaults.
    admin_router._reset_admin_overrides_for_tests()
    await admin_router._load_pin_quotas(TENANT)

    quotas = PinQuotas.for_tenant(TENANT)
    assert quotas.user == 3
    assert quotas.tenant_admin == 7
    assert quotas.org_admin is None  # -1 sentinel == unlimited


@pytest.mark.asyncio
async def test_defaults_when_no_blob_persisted(real_artifact_backed_admin):
    # A tenant whose blob was never written (distinct from the TENANT other
    # tests persist to the shared module-scoped Phoenix): enforcement resolves
    # the dataclass defaults, and must not cache them under the tenant key (so a
    # later PUT is still picked up).
    unwritten = "pinq-empty:pinq-empty"
    loaded = await admin_router._load_pin_quotas(unwritten)
    assert loaded == admin_router._default_pin_quotas()
    assert (
        admin_router.canonical_tenant_id(unwritten)
        not in admin_router._pin_quota_overrides
    )


@pytest.mark.asyncio
async def test_cache_rereads_persisted_blob_after_ttl(
    real_artifact_backed_admin, monkeypatch
):
    """The write-through cache is TTL-bounded: a blob updated by another replica
    (persisted straight to the real store) converges here after the TTL rather
    than being masked forever."""
    import json

    key = admin_router.canonical_tenant_id(TENANT)
    am = admin_router._build_artifact_manager(key)

    clock = {"now": 1000.0}
    monkeypatch.setattr(admin_router.time, "monotonic", lambda: clock["now"])

    await am.save_blob(
        admin_router._PIN_QUOTA_BLOB_KIND,
        admin_router._PIN_QUOTA_BLOB_KEY,
        json.dumps({"user": 1, "tenant_admin": 1, "org_admin": -1}),
    )
    assert (await admin_router._load_pin_quotas(TENANT))["user"] == 1

    # Another replica overwrites the persisted blob.
    await am.save_blob(
        admin_router._PIN_QUOTA_BLOB_KIND,
        admin_router._PIN_QUOTA_BLOB_KEY,
        json.dumps({"user": 9, "tenant_admin": 9, "org_admin": -1}),
    )

    # Within the TTL: this replica still serves its cached value.
    clock["now"] = 1000.0 + admin_router._PIN_QUOTA_CACHE_TTL_S - 1
    assert (await admin_router._load_pin_quotas(TENANT))["user"] == 1

    # Past the TTL: it re-reads the real store and converges.
    clock["now"] = 1000.0 + admin_router._PIN_QUOTA_CACHE_TTL_S + 1
    assert (await admin_router._load_pin_quotas(TENANT))["user"] == 9
