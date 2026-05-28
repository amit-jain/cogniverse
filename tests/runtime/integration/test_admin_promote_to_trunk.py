"""Admin promote-to-org-trunk endpoint.

Exercises POST .../memories/{id}/promote_to_org_trunk end-to-end against real
Vespa: a tenant admin promotes an org-shared ``knowledge_summary`` memory and
the org-trunk store gains a record carrying the promotion stamp; a
tenant_private kind is rejected with 403. The endpoint resolves promotable
kinds through ``build_promotable_registry`` (which marks ``knowledge_summary``
org-shared) — the default registry marks everything tenant_private, so a
promote against it always 403s.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.federation import org_trunk_tenant_id
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import build_default_registry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.routers import admin
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.async_polling import wait_for_vespa_indexing
from tests.utils.llm_config import get_llm_base_url, get_llm_model

pytestmark = pytest.mark.integration

TENANT = "test:unit"  # matches the runtime conftest's memory_manager fixture
TRUNK_TENANT = org_trunk_tenant_id(TENANT)


@pytest.fixture(scope="module")
def trunk_setup(vespa_instance, shared_denseon, memory_manager):
    """Initialise the org-trunk Mem0 manager so promotion has a target.

    The tenant Mem0 manager comes from the runtime conftest's
    ``memory_manager`` fixture (tenant_id="test:unit"); we initialise a
    second manager for the org-trunk tenant against the same Vespa
    instance.
    """
    Mem0MemoryManager._instances.pop(TRUNK_TENANT, None)
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    trunk_mm = Mem0MemoryManager(tenant_id=TRUNK_TENANT)
    trunk_mm.initialize(
        backend_host="http://localhost",
        backend_port=vespa_instance["http_port"],
        backend_config_port=vespa_instance["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=True,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    yield memory_manager, trunk_mm
    try:
        memory_manager.clear_agent_memory(memory_manager.tenant_id, "h7_promote")
        trunk_mm.clear_agent_memory(TRUNK_TENANT, "h7_promote")
    except Exception:
        pass
    Mem0MemoryManager._instances.pop(TRUNK_TENANT, None)


@pytest.fixture
def promote_client(trunk_setup, vespa_instance):
    tenant_mm, trunk_mm = trunk_setup
    # No registry monkeypatch: the endpoint resolves promotable kinds through
    # build_promotable_registry, where knowledge_summary is genuinely org-shared.
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    yield (
        TestClient(app, raise_server_exceptions=False),
        tenant_mm,
        trunk_mm,
        vespa_instance,
    )


class TestPromoteToOrgTrunk:
    def test_round_trip_promotion(self, promote_client):
        client, tenant_mm, trunk_mm, vespa_instance = promote_client
        # Seed a tenant memory of the org-shared, promotable kind. knowledge_summary
        # requires provenance, so attach a block (a real summary cites its sources).
        meta = attach_to_metadata(
            {"kind": "knowledge_summary"},
            make_provenance(
                written_by="agent:test",
                derivation_kind=DerivationKind.SYNTHESIS,
                confidence=0.9,
                derived_from=[CitationRef.external("https://wiki/p21")],
            ),
        )
        mid = tenant_mm.add_memory(
            content="HOUSE_RULE_promote_me_h7",
            tenant_id=TENANT,
            agent_name="h7_promote",
            metadata=meta,
            infer=False,
        )
        assert mid

        resp = client.post(
            f"/admin/tenants/{TENANT}/memories/{mid}/promote_to_org_trunk",
            json={"actor_role": "tenant_admin", "actor_id": "admin_alpha"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source_tenant_id"] == TENANT
        assert body["org_trunk_tenant_id"] == TRUNK_TENANT
        promoted_id = body["promoted_memory_id"]
        assert promoted_id

        # Vespa search/YQL is eventually consistent. Mem0's get_all goes
        # through that path, so reading immediately after the promotion's
        # internal feed can race the indexer and return zero rows. Wait
        # for indexing to settle before asserting visibility.
        wait_for_vespa_indexing(
            backend_url=f"http://localhost:{vespa_instance['http_port']}",
            delay=3,
            description="org-trunk promotion indexing",
        )

        # The promoted record must be visible in the org-trunk store
        # and carry the promotion stamp on its metadata. FederationService
        # falls back to "_promoted" as the agent_name when the source
        # memory dict (from Mem0.get_all) doesn't carry an "agent_name"
        # key — Mem0's row uses "agent_id" instead, which the helper
        # doesn't read.
        trunk_rows = trunk_mm.get_all_memories(
            tenant_id=TRUNK_TENANT, agent_name="_promoted"
        )
        promoted = next(
            (r for r in trunk_rows if str(r.get("id")) == promoted_id), None
        )
        assert promoted is not None, (
            f"promoted memory {promoted_id} not visible in trunk; "
            f"trunk_rows ids={[str(r.get('id')) for r in trunk_rows]}"
        )
        meta = promoted.get("metadata") or {}
        assert meta.get("promoted_from_tenant") == TENANT, (
            f"promotion stamp missing/wrong; got metadata={meta!r}"
        )
        assert meta.get("promoted_by") == "admin_alpha"
        assert meta.get("promoted_by_role") == "tenant_admin"

    def test_tenant_private_kind_rejected_403(self, promote_client):
        client, tenant_mm, _trunk_mm, _vi = promote_client
        # tenant_instruction is tenant_private in build_promotable_registry, so
        # the sensitivity gate must reject it (proves promotion isn't blanket-on).
        mid = tenant_mm.add_memory(
            content="HOUSE_RULE_private_h7",
            tenant_id=TENANT,
            agent_name="h7_promote",
            metadata={"kind": "tenant_instruction"},
            infer=False,
        )
        assert mid

        resp = client.post(
            f"/admin/tenants/{TENANT}/memories/{mid}/promote_to_org_trunk",
            json={"actor_role": "tenant_admin", "actor_id": "admin_alpha"},
        )
        assert resp.status_code == 403, resp.text
        assert "tenant_private" in resp.json()["detail"]

    def test_unknown_memory_returns_404(self, promote_client):
        client, _t, _tr, _vi = promote_client
        resp = client.post(
            f"/admin/tenants/{TENANT}/memories/no-such-id/promote_to_org_trunk",
            json={"actor_role": "tenant_admin", "actor_id": "admin_alpha"},
        )
        assert resp.status_code == 404

    def test_invalid_actor_role_returns_400(self, promote_client):
        client, _t, _tr, _vi = promote_client
        resp = client.post(
            f"/admin/tenants/{TENANT}/memories/anything/promote_to_org_trunk",
            json={"actor_role": "superadmin", "actor_id": "x"},
        )
        assert resp.status_code == 400
        assert "invalid actor_role" in resp.json()["detail"]
