"""Phase 9a — CrossTenantComparisonAgent end-to-end.

Pins:

  * Two same-org tenants writing the same subject_key with different
    content → POST /knowledge/cross_tenant/compare returns one
    TenantViewOut per tenant; matching_memory_ids hold the per-tenant
    write; distinct_signatures_count == 2 (two distinct contents).
  * A target tenant_id from a DIFFERENT org → HTTP 403 with the exact
    "cross-org comparison is forbidden" substring (via the route's
    ACLRejected handler).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

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
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.conftest import RUNTIME, skip_if_no_runtime, unique_id

VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"
# CrossTenantComparisonAgent reads via FederationService.federated_get_all
# whose default agent_name filter is "_promoted". Writing under that
# name lets the route find the data without needing a custom filter.
PROMOTED_AGENT = "_promoted"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    cm = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
        )
    )
    # In-memory only: cm.set_system_config would persist a denseon-only
    # localhost URL map into config_metadata and starve the in-cluster
    # ingestor (which reads inference_service_urls from the same store).
    cm._system_config_cache = SystemConfig(  # noqa: SLF001
        backend_url="http://localhost",
        backend_port=VESPA_HTTP_PORT,
        inference_service_urls={"denseon": DENSEON_URL},
    )
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=VESPA_HTTP_PORT,
        backend_config_port=VESPA_CONFIG_PORT,
        base_schema_name="agent_memories",
        llm_model="google/gemma-4-e4b-it",
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://cogniverse-vllm-llm-student.cogniverse:8000/v1",
        embedder_base_url=DENSEON_URL,
        auto_create_schema=True,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    return mm


def _write_fact(mm: Mem0MemoryManager, *, subject: str, content: str) -> str:
    """Write an entity_fact under agent_name='_promoted' so federation reads it."""
    prov = make_provenance(
        written_by="agent:phase9",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("phase9://src")],
    )
    metadata = attach_to_metadata({"kind": "entity_fact", "subject_key": subject}, prov)
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=PROMOTED_AGENT,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


# ---------------------------------------------------------------------------
# 1. Same-org compare returns per-tenant views
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestCompareReturnsPerTenantViews:
    """Two same-org tenants disagree on market_share → 2 views, 2 signatures."""

    def test_same_org_compare(self) -> None:
        Mem0MemoryManager._instances.clear()
        org = unique_id("kagent_xt")  # org id portion shared by both tenants
        t1 = f"{org}:t1"
        t2 = f"{org}:t2"
        mm1 = _build_manager(t1)
        mm2 = _build_manager(t2)
        try:
            id1 = _write_fact(mm1, subject="market_share", content="market_share: 30%")
            id2 = _write_fact(mm2, subject="market_share", content="market_share: 55%")

            with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
                resp = client.post(
                    f"/admin/tenants/{t1}/knowledge/cross_tenant/compare",
                    json={
                        "subject_key": "market_share",
                        "tenant_ids": [t1, t2],
                        "actor_role": "org_admin",
                        "actor_id": "oadmin",
                    },
                )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            assert body["subject_key"] == "market_share"
            assert len(body["tenant_views"]) == 2
            views_by_tid = {v["tenant_id"]: v for v in body["tenant_views"]}
            assert sorted(views_by_tid) == sorted([t1, t2])
            assert views_by_tid[t1]["matching_memory_ids"] == [id1]
            assert views_by_tid[t2]["matching_memory_ids"] == [id2]
            # Distinct signatures across the two views: 2 (different content).
            assert body["distinct_signatures_count"] == 2
        finally:
            try:
                mm1.clear_agent_memory(t1, PROMOTED_AGENT)
                mm2.clear_agent_memory(t2, PROMOTED_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. Cross-org compare → HTTP 403 with the exact substring
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestCrossOrgComparisonRejected:
    """Targeting a tenant in a different org returns 403 from the route."""

    def test_cross_org_returns_403(self) -> None:
        Mem0MemoryManager._instances.clear()
        org_a = unique_id("kagent_xta")
        org_b = unique_id("kagent_xtb")
        t_a = f"{org_a}:t1"
        t_b_other_org = f"{org_b}:t1"
        mm_a = _build_manager(t_a)
        mm_b = _build_manager(t_b_other_org)
        try:
            _write_fact(mm_a, subject="cross.org.subj", content="org A view")
            _write_fact(mm_b, subject="cross.org.subj", content="org B view")

            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                resp = client.post(
                    f"/admin/tenants/{t_a}/knowledge/cross_tenant/compare",
                    json={
                        "subject_key": "cross.org.subj",
                        "tenant_ids": [t_a, t_b_other_org],
                        "actor_role": "org_admin",
                        "actor_id": "oadmin",
                    },
                )
            assert resp.status_code == 403, resp.text[:500]
            detail = resp.json().get("detail", "")
            assert "cross-org comparison is forbidden" in detail, detail
            assert org_b in detail
        finally:
            try:
                mm_a.clear_agent_memory(t_a, PROMOTED_AGENT)
                mm_b.clear_agent_memory(t_b_other_org, PROMOTED_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
