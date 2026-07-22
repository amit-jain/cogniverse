"""FederatedQueryAgent end-to-end.

Pins:

  * Three same-org tenants each with a "lithium"-mentioning fact under
    agent_name="_promoted" → POST /knowledge/federated/query returns
    one hit per tenant whose excerpt contains the exact substring;
  * A target tenant in a DIFFERENT org → HTTP 403 (route's ACLRejected
    handler) with the exact "cross-org query is forbidden" substring.
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
VESPA_CONFIG_PORT = 33071
DENSEON_URL = "http://localhost:33906"
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


def _write_fact(mm: Mem0MemoryManager, *, content: str, subject: str) -> str:
    """Write under '_promoted' so federation reads find it."""
    prov = make_provenance(
        written_by="agent:fedq",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("fedq://fed")],
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
# 1. Three same-org tenants → 3 hits
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestFederatedFanOutCollectsHits:
    """Each of three same-org tenants holds one lithium fact → exactly 3 hits."""

    def test_three_tenants_yield_three_hits(self) -> None:
        Mem0MemoryManager._instances.clear()
        org = unique_id("kagent_fed")
        t1, t2, t3 = f"{org}:t1", f"{org}:t2", f"{org}:t3"
        mm1 = _build_manager(t1)
        mm2 = _build_manager(t2)
        mm3 = _build_manager(t3)
        try:
            _write_fact(mm1, content="lithium reserves at site A", subject="li.a")
            _write_fact(mm2, content="lithium prices in market B", subject="li.b")
            _write_fact(mm3, content="lithium mining in region C", subject="li.c")

            with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
                resp = client.post(
                    f"/admin/tenants/{t1}/knowledge/federated/query",
                    json={
                        "query": "lithium",
                        "tenant_ids": [t1, t2, t3],
                        "actor_role": "org_admin",
                        "actor_id": "oadmin",
                        "top_k": 5,
                    },
                )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            assert body["query"] == "lithium"
            hits = body["hits"]
            assert len(hits) == 3, hits
            assert sorted(h["tenant_id"] for h in hits) == sorted([t1, t2, t3])
            for h in hits:
                assert "lithium" in h["excerpt"].lower(), h
                assert h["origin"] == "tenant"
                assert h["memory_id"]
        finally:
            try:
                mm1.clear_agent_memory(t1, PROMOTED_AGENT)
                mm2.clear_agent_memory(t2, PROMOTED_AGENT)
                mm3.clear_agent_memory(t3, PROMOTED_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. Cross-org tenant in tenant_ids → 403
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestFederatedQueryACLDeniesNonOrgTenant:
    """Including a cross-org tenant in tenant_ids returns 403."""

    def test_cross_org_in_target_returns_403(self) -> None:
        Mem0MemoryManager._instances.clear()
        org_a = unique_id("kagent_fda")
        org_b = unique_id("kagent_fdb")
        t_a = f"{org_a}:t1"
        t_other = f"{org_b}:t1"
        mm_a = _build_manager(t_a)
        mm_other = _build_manager(t_other)
        try:
            _write_fact(mm_a, content="lithium fact in org A", subject="li.a")
            _write_fact(mm_other, content="lithium fact in org B", subject="li.b")

            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                resp = client.post(
                    f"/admin/tenants/{t_a}/knowledge/federated/query",
                    json={
                        "query": "lithium",
                        "tenant_ids": [t_a, t_other],
                        "actor_role": "org_admin",
                        "actor_id": "oadmin",
                        "top_k": 5,
                    },
                )
            assert resp.status_code == 403, resp.text[:500]
            detail = resp.json().get("detail", "")
            assert "cross-org query is forbidden" in detail, detail
            assert org_b in detail
        finally:
            try:
                mm_a.clear_agent_memory(t_a, PROMOTED_AGENT)
                mm_other.clear_agent_memory(t_other, PROMOTED_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
