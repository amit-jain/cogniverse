"""The admin tier routes against the real tenant registry and config store.

Tier is a per-tenant attribute an operator sets through the product. These
route the real FastAPI app over real Vespa: a set is read back exactly, an
unknown tenant is a 404, a value the router binds no group for is refused with
the vocabulary in the message, a store outage is a 503 rather than a tenant
silently reading as default, and eight concurrent sets on distinct tenants each
land on their own tenant.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.tenant_tiers import read_tenant_tier
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    ROUTER_TIERS,
)
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration

# Nothing listens here; the CI-parity default for a dead backend port.
DEAD_VESPA_PORT = 29071

CREATED_AT = 1757000000000
SEEDED_SCHEMAS = ["video_colpali_smol500_mv_frame"]


@pytest.fixture
def wired_tenant_manager(config_manager, schema_loader):
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    BackendRegistry.get_instance().clear_instances()


@pytest.fixture
def client(wired_tenant_manager):
    transport = httpx.ASGITransport(app=tm.app)
    return httpx.AsyncClient(transport=transport, base_url="http://tier-test")


def _seed_tenant() -> str:
    org_id = f"tier{uuid.uuid4().hex[:8]}"
    tenant_id = f"{org_id}:production"
    backend = tm.get_backend()
    stored = backend.create_metadata_document(
        schema="tenant_metadata",
        doc_id=tenant_id,
        fields={
            "tenant_full_id": tenant_id,
            "org_id": org_id,
            "tenant_name": "production",
            "created_at": CREATED_AT,
            "created_by": "tier-test",
            "status": "active",
            "schemas_deployed": SEEDED_SCHEMAS,
        },
    )
    assert stored is True
    return tenant_id


class TestTierRoundTrip:
    async def test_a_tenant_with_no_tier_reads_as_the_default(self, client):
        tenant_id = _seed_tenant()
        async with client as c:
            response = await c.get(f"/admin/tenants/{tenant_id}/tier")
        assert response.status_code == 200
        assert response.json() == {
            "tenant_id": tenant_id,
            "tier": DEFAULT_ROUTER_TIER,
        }

    async def test_every_tier_set_is_read_back_exactly(self, client, config_manager):
        tenant_id = _seed_tenant()
        async with client as c:
            for tier in sorted(ROUTER_TIERS):
                put = await c.put(
                    f"/admin/tenants/{tenant_id}/tier", json={"tier": tier}
                )
                assert put.status_code == 200
                assert put.json() == {"tenant_id": tenant_id, "tier": tier}

                got = await c.get(f"/admin/tenants/{tenant_id}/tier")
                assert got.status_code == 200
                assert got.json() == {"tenant_id": tenant_id, "tier": tier}
                assert read_tenant_tier(config_manager, tenant_id) == tier

    async def test_the_simple_form_addresses_the_canonical_tenant(self, client):
        tenant_id = _seed_tenant()
        org_id = tenant_id.split(":")[0]
        simple = f"{org_id}:{org_id}"
        # A tenant whose org and name match is the simple form's canonical id.
        tm.get_backend().create_metadata_document(
            schema="tenant_metadata",
            doc_id=simple,
            fields={
                "tenant_full_id": simple,
                "org_id": org_id,
                "tenant_name": org_id,
                "created_at": CREATED_AT,
                "created_by": "tier-test",
                "status": "active",
                "schemas_deployed": SEEDED_SCHEMAS,
            },
        )
        async with client as c:
            put = await c.put(f"/admin/tenants/{org_id}/tier", json={"tier": "pro"})
            assert put.status_code == 200
            assert put.json() == {"tenant_id": simple, "tier": "pro"}
            got = await c.get(f"/admin/tenants/{simple}/tier")
            assert got.json() == {"tenant_id": simple, "tier": "pro"}


class TestRefusals:
    async def test_a_tier_outside_the_vocabulary_is_422_naming_the_set(self, client):
        tenant_id = _seed_tenant()
        async with client as c:
            response = await c.put(
                f"/admin/tenants/{tenant_id}/tier", json={"tier": "gold"}
            )
        assert response.status_code == 422
        assert response.json() == {
            "detail": (
                f"Unknown router tier 'gold'. Valid tiers: {sorted(ROUTER_TIERS)}"
            )
        }

    async def test_a_refused_tier_leaves_the_stored_tier_untouched(
        self, client, config_manager
    ):
        tenant_id = _seed_tenant()
        async with client as c:
            await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "pro"})
            await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "gold"})
            got = await c.get(f"/admin/tenants/{tenant_id}/tier")
        assert got.json() == {"tenant_id": tenant_id, "tier": "pro"}
        assert read_tenant_tier(config_manager, tenant_id) == "pro"

    async def test_an_unknown_tenant_is_404_on_both_verbs(self, client):
        missing = f"nosuch{uuid.uuid4().hex[:8]}:production"
        async with client as c:
            got = await c.get(f"/admin/tenants/{missing}/tier")
            put = await c.put(f"/admin/tenants/{missing}/tier", json={"tier": "pro"})
        assert got.status_code == 404
        assert got.json() == {"detail": f"Tenant {missing} not found"}
        assert put.status_code == 404
        assert put.json() == {"detail": f"Tenant {missing} not found"}


class TestFaultContract:
    async def test_a_dead_config_store_is_503_not_a_default_tier(
        self, client, wired_tenant_manager
    ):
        tenant_id = _seed_tenant()
        dead = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://localhost", backend_port=DEAD_VESPA_PORT
            )
        )
        previous = tm._config_manager
        try:
            async with client as c:
                tm._config_manager = dead
                got = await c.get(f"/admin/tenants/{tenant_id}/tier")
        finally:
            tm._config_manager = previous
        assert got.status_code == 503
        assert got.json() == {"detail": "Tenant registry temporarily unavailable"}


class TestConcurrency:
    async def test_eight_concurrent_sets_on_distinct_tenants_land_on_their_own(
        self, client, config_manager
    ):
        """Eight PUTs in flight on one loop, one per tenant: each tenant ends on
        the tier its own request named, and every response says so."""
        import asyncio

        tiers = sorted(ROUTER_TIERS)
        tenants = [_seed_tenant() for _ in range(8)]
        expected = {t: tiers[i % len(tiers)] for i, t in enumerate(tenants)}

        async with client as c:
            puts = await asyncio.gather(
                *(
                    c.put(f"/admin/tenants/{t}/tier", json={"tier": expected[t]})
                    for t in tenants
                )
            )
            gets = await asyncio.gather(
                *(c.get(f"/admin/tenants/{t}/tier") for t in tenants)
            )

        assert [r.status_code for r in puts] == [200] * 8
        assert [r.json() for r in puts] == [
            {"tenant_id": t, "tier": expected[t]} for t in tenants
        ]
        assert [r.json() for r in gets] == [
            {"tenant_id": t, "tier": expected[t]} for t in tenants
        ]
        assert {t: read_tenant_tier(config_manager, t) for t in tenants} == expected
