"""Admin promote-to-org-trunk endpoint.

Federation was missing an admin endpoint: only the disabled-and-
unreachable KnowledgeSummarizationAgent called
FederationService.promote_to_org_trunk. Operators need to promote a
tenant memory back to the trunk via a CLI or admin endpoint.

This test exercises the new POST .../memories/{id}/promote_to_org_trunk
endpoint end-to-end against real Vespa: tenant admin promotes a
tenant_instruction memory; the org-trunk store gains a record carrying
the promotion stamp; an org-shared kind succeeds while a tenant_private
kind is rejected with 403.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.federation import org_trunk_tenant_id
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import (
    KnowledgeSchema,
    Pinnable,
    Sensitivity,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.routers import admin
from cogniverse_vespa.config.config_store import VespaConfigStore
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
def promote_client(trunk_setup, monkeypatch):
    tenant_mm, trunk_mm = trunk_setup
    # Override the default registry's sensitivity for tenant_instruction
    # so the promote endpoint's schema gate considers it org-shared.
    # We do this by monkeypatching FederationService's lookup path —
    # the runtime's build_default_registry returns a fresh registry per
    # call; we patch the build to return a registry where the schema
    # is mutated.

    def _patched_registry():
        reg = build_default_registry()
        reg.register(
            KnowledgeSchema(
                kind="tenant_instruction",
                sensitivity=Sensitivity.ORG_SHARED,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
                default_trust=0.95,
            ),
            replace=True,
        )
        return reg

    monkeypatch.setattr(
        "cogniverse_core.memory.schema.build_default_registry",
        _patched_registry,
    )
    # Re-import the symbol the admin router uses lazily so the patch
    # is in scope when the endpoint runs.
    import cogniverse_runtime.routers.admin as _admin_mod  # noqa: F401

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    yield TestClient(app, raise_server_exceptions=False), tenant_mm, trunk_mm


class TestPromoteToOrgTrunk:
    def test_round_trip_promotion(self, promote_client):
        client, tenant_mm, trunk_mm = promote_client
        # Seed a tenant memory.
        mid = tenant_mm.add_memory(
            content="HOUSE_RULE_promote_me_h7",
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
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source_tenant_id"] == TENANT
        assert body["org_trunk_tenant_id"] == TRUNK_TENANT
        promoted_id = body["promoted_memory_id"]
        assert promoted_id

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

    def test_unknown_memory_returns_404(self, promote_client):
        client, _t, _tr = promote_client
        resp = client.post(
            f"/admin/tenants/{TENANT}/memories/no-such-id/promote_to_org_trunk",
            json={"actor_role": "tenant_admin", "actor_id": "admin_alpha"},
        )
        assert resp.status_code == 404

    def test_invalid_actor_role_returns_400(self, promote_client):
        client, _t, _tr = promote_client
        resp = client.post(
            f"/admin/tenants/{TENANT}/memories/anything/promote_to_org_trunk",
            json={"actor_role": "superadmin", "actor_id": "x"},
        )
        assert resp.status_code == 400
        assert "invalid actor_role" in resp.json()["detail"]
