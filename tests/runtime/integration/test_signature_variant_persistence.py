"""Signature-variant selections must persist and reach every replica.

A PUT of a tenant's per-agent variant wrote only the process dict of the
replica that served it — lost on restart and invisible to the dispatcher on
every other replica, while the route returned 200 as if applied. These drive
the REAL admin route against a REAL Phoenix container: a PUT persists the blob,
a cold replica reads it back, and the dispatcher's resolver returns the exact
persisted variant.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANT = "sigvar-persist:sigvar-persist"


@pytest.fixture
def real_admin(telemetry_manager_with_phoenix, monkeypatch):
    provider = telemetry_manager_with_phoenix.get_provider(tenant_id=TENANT)
    monkeypatch.setattr(
        admin_router,
        "_build_artifact_manager",
        lambda key: ArtifactManager(provider, tenant_id=key),
    )
    admin_router._reset_admin_overrides_for_tests()
    yield
    admin_router._reset_admin_overrides_for_tests()


async def _put(app, tenant, agent, variant):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.put(
            f"/admin/tenants/{tenant}/signature_variants/{agent}",
            json={"variant_id": variant},
        )


@pytest.mark.asyncio
async def test_variant_persists_and_resolves_on_cold_replica(real_admin):
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")

    resp = await _put(app, TENANT, "search_agent", "search_v2")
    assert resp.status_code == 200
    assert resp.json()["selections"]["search_agent"] == "search_v2"

    # Cold replica: cache cleared. It must resolve the persisted variant from
    # the durable blob, not fall back to the default.
    admin_router._reset_admin_overrides_for_tests()
    loaded = await admin_router.load_signature_variants(TENANT)
    assert loaded == {"search_agent": "search_v2"}

    # The dispatcher's resolver (the real consumer) reads the warmed cache.
    assert (
        AgentDispatcher._resolve_signature_variant(TENANT, "search_agent")
        == "search_v2"
    )
    # An agent the tenant never selected still resolves to the default.
    assert (
        AgentDispatcher._resolve_signature_variant(TENANT, "summarizer_agent")
        == "default"
    )


@pytest.mark.asyncio
async def test_second_agent_selection_merges_not_replaces(real_admin):
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")

    await _put(app, TENANT, "search_agent", "search_v2")
    resp = await _put(app, TENANT, "summarizer_agent", "sum_v3")
    assert resp.status_code == 200

    admin_router._reset_admin_overrides_for_tests()
    loaded = await admin_router.load_signature_variants(TENANT)
    assert loaded == {"search_agent": "search_v2", "summarizer_agent": "sum_v3"}
