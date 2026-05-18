"""Phase 8b — KGTraversalAgent end-to-end.

Pins the shipped KG walker against the deployed cluster:

  * given a 3-node graph alice→acme→london (kg_node + kg_edge memories),
    POST /knowledge/kg/traverse from ``start_subject_key=alice`` returns
    every node + every edge in the connected subgraph;
  * with ``relation_filter=["works_at"]`` the walker stops at acme and
    returns only the alice→acme edge + the alice and acme nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

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


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    Mem0MemoryManager._instances.clear()
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


def _write_kg_node(
    mm: Mem0MemoryManager,
    *,
    subject_key: str,
    content: str,
    agent_name: str = "kg_traversal_agent",
) -> str:
    """kg_node is provenance_required (default). Attach a minimal provenance."""
    prov = make_provenance(
        written_by="agent:phase8",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("phase8://node")],
    )
    metadata = attach_to_metadata(
        {"kind": "kg_node", "subject_key": subject_key, "label": subject_key.title()},
        prov,
    )
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=agent_name,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


def _write_kg_edge(
    mm: Mem0MemoryManager,
    *,
    src: str,
    dst: str,
    relation: str,
    agent_name: str = "kg_traversal_agent",
) -> str:
    """kg_edge is provenance_required (default)."""
    prov = make_provenance(
        written_by="agent:phase8",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("phase8://edge")],
    )
    metadata = attach_to_metadata(
        {
            "kind": "kg_edge",
            "subject_key": f"{src}->{dst}",
            "from_subject_key": src,
            "to_subject_key": dst,
            "relation": relation,
        },
        prov,
    )
    mid = mm.add_memory(
        content=f"{src} {relation} {dst}",
        tenant_id=mm.tenant_id,
        agent_name=agent_name,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


def _seed_alice_acme_london(mm: Mem0MemoryManager) -> Tuple[List[str], List[str]]:
    """Build alice—works_at→acme—based_in→london and return (node_ids, edge_ids)."""
    nodes = [
        _write_kg_node(mm, subject_key="alice", content="alice is a person"),
        _write_kg_node(mm, subject_key="acme", content="acme is a company"),
        _write_kg_node(mm, subject_key="london", content="london is a city"),
    ]
    edges = [
        _write_kg_edge(mm, src="alice", dst="acme", relation="works_at"),
        _write_kg_edge(mm, src="acme", dst="london", relation="based_in"),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# 1. Full traversal returns the connected subgraph
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestKGTraversalReturnsExpectedSubgraph:
    """Walk from alice yields {alice, acme, london} nodes + both edges."""

    def test_full_walk_returns_3_nodes_and_2_edges(self) -> None:
        tenant_id = unique_id("kagent_kg") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            _seed_alice_acme_london(mm)

            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/kg/traverse",
                    json={
                        "start_subject_key": "alice",
                        "max_depth": 3,
                        "max_nodes": 50,
                    },
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()
            assert body["start_subject_key"] == "alice"
            assert sorted(n["subject_key"] for n in body["nodes"]) == [
                "acme",
                "alice",
                "london",
            ]
            edges_triples = sorted(
                (e["from_subject_key"], e["to_subject_key"], e["relation"])
                for e in body["edges"]
            )
            assert edges_triples == [
                ("acme", "london", "based_in"),
                ("alice", "acme", "works_at"),
            ]
        finally:
            mm.clear_agent_memory(tenant_id, "kg_traversal_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. relation_filter prunes to the matching edges
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestKGTraversalRespectsRelationFilter:
    """relation_filter=[works_at] → only alice→acme edge + {alice, acme} nodes."""

    def test_works_at_filter_returns_only_alice_acme_edge(self) -> None:
        tenant_id = unique_id("kagent_kgf") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            _seed_alice_acme_london(mm)

            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/kg/traverse",
                    json={
                        "start_subject_key": "alice",
                        "relation_filter": ["works_at"],
                        "max_depth": 3,
                        "max_nodes": 50,
                    },
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()
            assert sorted(n["subject_key"] for n in body["nodes"]) == [
                "acme",
                "alice",
            ]
            edges_triples = sorted(
                (e["from_subject_key"], e["to_subject_key"], e["relation"])
                for e in body["edges"]
            )
            assert edges_triples == [("alice", "acme", "works_at")]
        finally:
            mm.clear_agent_memory(tenant_id, "kg_traversal_agent")
            Mem0MemoryManager._instances.clear()
