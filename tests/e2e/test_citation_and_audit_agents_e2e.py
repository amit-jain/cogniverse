"""Phase 8a — CitationTracingAgent (truncate path) + AuditExplanationAgent.

Pins two read-only knowledge agents against the deployed cluster:

  * CitationTracingAgent: a chain of three memories (LEAF←MID←ROOT) trimmed
    to ``max_depth=1`` returns ``truncated=True`` AND exactly the LEAF +
    one child node;
  * AuditExplanationAgent: a synthesis memory whose provenance ties to two
    conflicting same-subject facts surfaces those sources with trust
    annotations AND lists the conflict in ``contradictions_touched``.

The Phase 2 file already pins the happy-path citation chain walk via the
HTTP route; this module only adds the truncation and audit contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

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
    _warmup_provenance_schema(mm)
    return mm


def _warmup_provenance_schema(mm: Mem0MemoryManager, timeout_s: float = 120.0) -> None:
    """Block until the per-tenant provenance schema accepts writes (Phase 2 pattern)."""
    import time
    import uuid

    probe_id = f"warmup-{uuid.uuid4().hex[:8]}"
    probe_prov = make_provenance(
        written_by="warmup",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.5,
        derived_from=[CitationRef.external("warmup://probe")],
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            mm.provenance_store.attach(probe_id, probe_prov)
            time.sleep(1.0)
            if mm.provenance_store.get(probe_id) is not None:
                return
        except Exception:
            pass
        time.sleep(1.0)
    pytest.fail(
        f"provenance schema for tenant {mm.tenant_id!r} not online within {timeout_s}s"
    )


def _wait_for_provenance(
    mm: Mem0MemoryManager, memory_id: str, timeout_s: float = 30.0
) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if mm.provenance_store.get(memory_id) is not None:
            return
        time.sleep(1.0)
    pytest.fail(f"provenance for memory {memory_id!r} not indexed within {timeout_s}s")


def _write_with_provenance(
    mm: Mem0MemoryManager,
    *,
    kind: str,
    content: str,
    derivation_kind: DerivationKind,
    derived_from: List[CitationRef],
    subject_key: str,
    confidence: float = 0.85,
    agent_name: str = "phase8_agent",
) -> str:
    prov = make_provenance(
        written_by="agent:phase8",
        derivation_kind=derivation_kind,
        confidence=confidence,
        derived_from=derived_from,
    )
    metadata = attach_to_metadata({"kind": kind, "subject_key": subject_key}, prov)
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=agent_name,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


# ---------------------------------------------------------------------------
# 1. citation/trace walks the full chain back to the primary source
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestCitationTracingAgentWalksToPrimary:
    """3-deep chain ROOT←MID←LEAF; trace from LEAF returns the full ordered chain."""

    def test_walks_leaf_mid_root_with_exact_depths(self) -> None:
        tenant_id = unique_id("kagent_cw") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            root = _write_with_provenance(
                mm,
                kind="external_doc",
                content="ROOT primary source",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                derived_from=[CitationRef.external("phase8://primary")],
                subject_key="cw.root",
            )
            _wait_for_provenance(mm, root)
            mid = _write_with_provenance(
                mm,
                kind="entity_fact",
                content="MID extracted from ROOT",
                derivation_kind=DerivationKind.EXTRACTION,
                derived_from=[CitationRef.memory(root)],
                subject_key="cw.mid",
            )
            _wait_for_provenance(mm, mid)
            leaf = _write_with_provenance(
                mm,
                kind="entity_fact",
                content="LEAF synthesised from MID",
                derivation_kind=DerivationKind.SYNTHESIS,
                derived_from=[CitationRef.memory(mid)],
                subject_key="cw.leaf",
            )
            _wait_for_provenance(mm, leaf)

            # max_depth=10 ensures the walker reaches ROOT (depth=2) without
            # truncation. max_nodes=10 is well above the chain length.
            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/citations/trace",
                    json={"memory_id": leaf, "max_depth": 10, "max_nodes": 10},
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()
            assert body["root_memory_id"] == leaf
            assert body["truncated"] is False
            # Exact id list in BFS order: LEAF (depth 0) → MID (1) → ROOT (2).
            assert [n["memory_id"] for n in body["nodes"]] == [leaf, mid, root]
            assert [n["depth"] for n in body["nodes"]] == [0, 1, 2]
            assert [n["derivation_kind"] for n in body["nodes"]] == [
                "synthesis",
                "extraction",
                "direct_ingest",
            ]
        finally:
            mm.clear_agent_memory(tenant_id, "phase8_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. citation/trace truncates the chain at max_depth=1
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestCitationTracingAgentTruncatesAtMaxDepth:
    """3-deep chain (LEAF←MID←ROOT) walked at max_depth=1 → truncated + 2 nodes."""

    def test_truncates_at_max_depth_1(self) -> None:
        tenant_id = unique_id("kagent_ct") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            root = _write_with_provenance(
                mm,
                kind="external_doc",
                content="ROOT doc",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                derived_from=[CitationRef.external("phase8://root")],
                subject_key="ct.root",
            )
            _wait_for_provenance(mm, root)
            mid = _write_with_provenance(
                mm,
                kind="entity_fact",
                content="MID derived from ROOT",
                derivation_kind=DerivationKind.EXTRACTION,
                derived_from=[CitationRef.memory(root)],
                subject_key="ct.mid",
            )
            _wait_for_provenance(mm, mid)
            leaf = _write_with_provenance(
                mm,
                kind="entity_fact",
                content="LEAF synthesised from MID",
                derivation_kind=DerivationKind.SYNTHESIS,
                derived_from=[CitationRef.memory(mid)],
                subject_key="ct.leaf",
            )
            _wait_for_provenance(mm, leaf)

            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/citations/trace",
                    json={"memory_id": leaf, "max_depth": 1, "max_nodes": 10},
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()
            assert body["root_memory_id"] == leaf
            assert body["truncated"] is True
            # max_depth=1 → only LEAF (depth=0) and MID (depth=1); ROOT
            # at depth=2 is past the cap. Exact id list pinned.
            assert [n["memory_id"] for n in body["nodes"]] == [leaf, mid]
            assert [n["depth"] for n in body["nodes"]] == [0, 1]
        finally:
            mm.clear_agent_memory(tenant_id, "phase8_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. audit/explain surfaces sources with trust + the touched contradiction
# ---------------------------------------------------------------------------


def _write_entity_fact_no_prov(
    mm: Mem0MemoryManager,
    *,
    subject_key: str,
    content: str,
    derivation_kind: DerivationKind = DerivationKind.DIRECT_INGEST,
    confidence: float = 0.9,
    agent_name: str = "phase8_agent",
) -> str:
    """Write an entity_fact with provenance (entity_fact is provenance_required)."""
    prov = make_provenance(
        written_by="agent:phase8",
        derivation_kind=derivation_kind,
        confidence=confidence,
        derived_from=[CitationRef.external("phase8://src")],
    )
    metadata = attach_to_metadata(
        {"kind": "entity_fact", "subject_key": subject_key}, prov
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


@pytest.mark.e2e
@skip_if_no_runtime
class TestAuditExplanationAgentSurfacesTrustAndContradictions:
    """Synthesis derived from two conflicting facts → audit shows both + conflict."""

    def test_audit_lists_sources_and_contradictions(self) -> None:
        tenant_id = unique_id("kagent_au") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            # Two conflicting same-subject entity_facts.
            a = _write_entity_fact_no_prov(
                mm, subject_key="company.ceo", content="ceo: Alice"
            )
            _wait_for_provenance(mm, a)
            b = _write_entity_fact_no_prov(
                mm, subject_key="company.ceo", content="ceo: Bob"
            )
            _wait_for_provenance(mm, b)

            # Synthesis memory whose provenance.derived_from cites BOTH
            # facts — that's how AuditExplanationAgent walks back to them.
            answer = _write_with_provenance(
                mm,
                kind="entity_fact",
                content="answer: there is disagreement about company.ceo",
                derivation_kind=DerivationKind.SYNTHESIS,
                derived_from=[CitationRef.memory(a), CitationRef.memory(b)],
                subject_key="answer.company.ceo",
            )
            _wait_for_provenance(mm, answer)

            with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/audit/explain",
                    json={
                        "answer_memory_id": answer,
                        "include_trust": True,
                        "include_contradictions": True,
                        "max_chain_depth": 5,
                        "max_chain_nodes": 20,
                    },
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()
            assert body["answer_memory_id"] == answer

            source_ids = sorted(s["memory_id"] for s in body["sources"])
            # Both conflicting parents must appear as sources at depth 1
            # (answer itself is depth 0).
            assert {a, b}.issubset(set(source_ids)), (
                f"sources missing parents: got {source_ids}"
            )
            # trust scores attached on each source row.
            for s in body["sources"]:
                if s["memory_id"] in {a, b}:
                    assert s["depth"] == 1
                    assert s["derivation_kind"] == "direct_ingest"
                    assert isinstance(s["trust_score"], float)
                    assert 0.0 <= s["trust_score"] <= 1.0

            # Exactly one ConflictSet for the colliding subject_key.
            contradictions = body["contradictions_touched"]
            assert len(contradictions) == 1, contradictions
            conflict = contradictions[0]
            assert conflict["subject_key"] == "company.ceo"
            assert sorted(conflict["conflicting_memory_ids"]) == sorted([a, b])
        finally:
            mm.clear_agent_memory(tenant_id, "phase8_agent")
            Mem0MemoryManager._instances.clear()
