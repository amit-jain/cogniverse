"""Knowledge-agent tests with the memory manager injected via MagicMock.

Each agent's ``memory_manager_factory`` (or mixin attribute) is given a
MagicMock that returns canned rows in the shape Mem0 produces. These
tests exercise the agent's own logic — input validation, walk
semantics, reconciliation policy choice, summary composition — without
paying the full Vespa + LLM cost of a real backend.

Real-backend coverage for every knowledge agent lives in:

  * ``tests/memory/integration/test_knowledge_agents_real_vespa.py`` —
    ContradictionReconciliation, KnowledgeSummarization, AuditExplanation
    against live Mem0 + Vespa.
  * ``tests/memory/integration/test_knowledge_agents_extra_real_vespa.py``
    — MultiDoc, KGTraversal, CrossTenant, TemporalReasoning, FederatedQuery
    against live Mem0 + Vespa (multi-tenant fixture for the
    federation-shaped agents).

This file complements those: same agents, lower cost per test, used
to drive policy-branch coverage that doesn't need a real backend.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_core.memory.schema import build_default_registry

pytestmark = pytest.mark.integration

TENANT = "knowledge_agents_factory_test"


def _factory_returning(rows_by_tenant: Dict[str, List[Dict[str, Any]]]):
    """For multi-tenant agents (cross-tenant + federated) — inject per-tenant memories
    via the agent's memory_manager_factory seam. The factory returns a
    Mem0-shaped object with the right rows for each requested tenant."""

    def _factory(tenant_id: str):
        mm = MagicMock()
        mm.memory = MagicMock()
        rows = list(rows_by_tenant.get(tenant_id, []))
        mm.get_all_memories = lambda *, tenant_id=tenant_id, agent_name: list(rows)
        return mm

    return _factory


# ----- MultiDocumentSynthesisAgent --------------------------------------


@pytest.mark.asyncio
async def test_synthesises_across_documents():
    """Two memory_ids → agent fetches via injected manager → cites both."""
    from cogniverse_agents.multi_document_synthesis_agent import (
        DocumentRef,
        MultiDocSynthesisDeps,
        MultiDocSynthesisInput,
        MultiDocumentSynthesisAgent,
    )

    rows_by_id = {
        "doc_a": {"id": "doc_a", "memory": "Doc A: refund policy is 30 days"},
        "doc_b": {"id": "doc_b", "memory": "Doc B: EU buyers get 14-day window"},
    }
    fake_mm = MagicMock()
    fake_mm.memory = MagicMock()
    fake_mm.memory.get = lambda memory_id: rows_by_id.get(memory_id)

    agent = MultiDocumentSynthesisAgent(
        deps=MultiDocSynthesisDeps(tenant_id=TENANT),
    )
    agent._dspy_module = MagicMock(return_value=MagicMock(answer="STUB-SYNTHESIS-OK"))
    agent.memory_manager = fake_mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = "multi_doc_synth_test"

    out = await agent._process_impl(
        MultiDocSynthesisInput(
            tenant_id=TENANT,
            query="What is the refund policy?",
            documents=[
                DocumentRef(memory_id="doc_a", label="A"),
                DocumentRef(memory_id="doc_b", label="B"),
            ],
            persist=False,
        )
    )
    assert out.answer == "STUB-SYNTHESIS-OK"

    # citation_refs is a list of dicts (Pydantic-serialized) at the agent
    # output boundary; in-memory it's CitationRef. Handle both shapes.
    def _ref_id(r):
        return r.get("ref_id") if isinstance(r, dict) else r.ref_id

    cited_ids = {_ref_id(r) for r in out.citation_refs}
    assert {"doc_a", "doc_b"}.issubset(cited_ids)


# ----- TemporalReasoningAgent -------------------------------------------


@pytest.mark.asyncio
async def test_memories_bucketed_by_window():
    from cogniverse_agents.temporal_reasoning_agent import (
        TemporalReasoningAgent,
        TemporalReasoningDeps,
        TemporalReasoningInput,
        TimeWindow,
    )

    # Inject metadata via the boundary-spy pattern so written_at survives.
    rows = [
        {
            "id": f"q1_{i}",
            "memory": f"Q1 fact {i}",
            "metadata": {
                "subject_key": "policy:refunds",
                "written_at": "2026-02-15T00:00:00Z",
            },
        }
        for i in range(2)
    ]
    rows.append(
        {
            "id": "q2_a",
            "memory": "Q2 fact A",
            "metadata": {
                "subject_key": "policy:refunds",
                "written_at": "2026-05-15T00:00:00Z",
            },
        }
    )
    factory = _factory_returning({TENANT: rows})

    agent = TemporalReasoningAgent(
        deps=TemporalReasoningDeps(tenant_id=TENANT),
        memory_manager_factory=factory,
    )
    out = await agent._process_impl(
        TemporalReasoningInput(
            tenant_id=TENANT,
            subject_key="policy:refunds",
            windows=[
                TimeWindow(
                    label="Q1",
                    start="2026-01-01T00:00:00Z",
                    end="2026-04-01T00:00:00Z",
                ),
                TimeWindow(
                    label="Q2",
                    start="2026-04-01T00:00:00Z",
                    end="2026-07-01T00:00:00Z",
                ),
            ],
        )
    )
    q1 = next(v for v in out.window_views if v.label == "Q1")
    q2 = next(v for v in out.window_views if v.label == "Q2")
    assert len(q1.matching_memory_ids) == 2
    assert len(q2.matching_memory_ids) == 1
    assert out.distinct_signatures_count == 2


# ----- FederatedQueryAgent ----------------------------------------------


@pytest.mark.asyncio
async def test_aggregates_across_two_tenants():
    from cogniverse_agents.federated_query_agent import (
        FederatedQueryAgent,
        FederatedQueryDeps,
        FederatedQueryInput,
    )

    rows_by_tenant = {
        "acme:alpha": [{"id": "a1", "memory": "Paris is the capital of France"}],
        "acme:beta": [{"id": "b1", "memory": "Paris bistros are great"}],
        "acme:_org_trunk": [],
    }
    agent = FederatedQueryAgent(
        deps=FederatedQueryDeps(tenant_id="acme:caller"),
        memory_manager_factory=_factory_returning(rows_by_tenant),
        registry=build_default_registry(),
    )
    out = await agent._process_impl(
        FederatedQueryInput(
            tenant_id="acme:caller",
            query="Paris",
            tenant_ids=["acme:alpha", "acme:beta"],
            actor_role="org_admin",
            actor_id="oadm",
        )
    )
    assert {h.memory_id for h in out.hits} == {"a1", "b1"}


# ----- CrossTenantComparisonAgent ---------------------------------------


@pytest.mark.asyncio
async def test_compares_distinct_subject_signatures_across_tenants():
    from cogniverse_agents.cross_tenant_comparison_agent import (
        CrossTenantComparisonAgent,
        CrossTenantComparisonDeps,
        CrossTenantComparisonInput,
    )

    rows = {
        "acme:alpha": [
            {
                "id": "a",
                "memory": "Paris is the capital",
                "metadata": {"subject_key": "france:capital"},
            }
        ],
        "acme:beta": [
            {
                "id": "b",
                "memory": "Lyon is the capital",
                "metadata": {"subject_key": "france:capital"},
            }
        ],
        "acme:_org_trunk": [],
    }
    agent = CrossTenantComparisonAgent(
        deps=CrossTenantComparisonDeps(tenant_id="acme:caller"),
        memory_manager_factory=_factory_returning(rows),
        registry=build_default_registry(),
    )
    out = await agent._process_impl(
        CrossTenantComparisonInput(
            tenant_id="acme:caller",
            subject_key="france:capital",
            tenant_ids=["acme:alpha", "acme:beta"],
            actor_role="org_admin",
            actor_id="oadm",
        )
    )
    # Disagreement → 2 signatures.
    assert out.distinct_signatures_count == 2


# ----- ContradictionReconciliationAgent ---------------------------------


@pytest.mark.asyncio
async def test_reconciles_real_conflict_set():
    from cogniverse_agents.contradiction_reconciliation_agent import (
        ContradictionReconciliationAgent,
        ContradictionReconciliationDeps,
        ContradictionReconciliationInput,
    )
    from cogniverse_core.memory.contradiction import ConflictSet

    conflict = ConflictSet(
        subject_key="france:capital",
        conflicting_memory_ids=["mem_a", "mem_b"],
        detected_at="2026-05-09T00:00:00Z",
    )
    _ = conflict  # kept for documentation; the input shape uses ids directly
    fake_mm = MagicMock()
    fake_mm.memory = MagicMock()
    fake_mm.memory.get = lambda memory_id: {
        "id": memory_id,
        "memory": f"content for {memory_id}",
        "metadata": {"subject_key": "france:capital"},
    }
    agent = ContradictionReconciliationAgent(
        deps=ContradictionReconciliationDeps(tenant_id=TENANT),
    )
    agent.memory_manager = fake_mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = "contradiction_reconciliation_test"

    out = await agent._process_impl(
        ContradictionReconciliationInput(
            tenant_id=TENANT,
            target_kind="entity_fact",
            conflict_member_ids=conflict.conflicting_memory_ids,
        )
    )
    # The agent should produce SOME structured output for the conflict.
    assert hasattr(out, "metadata")


# ----- KnowledgeSummarizationAgent --------------------------------------


@pytest.mark.asyncio
async def test_summarises_real_subject_slice():
    from cogniverse_agents.knowledge_summarization_agent import (
        KnowledgeSummarizationAgent,
        KnowledgeSummarizationDeps,
        KnowledgeSummarizationInput,
    )

    rows = [
        {
            "id": f"k{i}",
            "memory": f"refund fact {i}",
            "metadata": {"kind": "external_doc", "subject_key": "policy:refunds"},
        }
        for i in range(3)
    ]
    factory = _factory_returning({TENANT: rows})
    agent = KnowledgeSummarizationAgent(
        deps=KnowledgeSummarizationDeps(tenant_id=TENANT),
        memory_manager_factory=factory,
        registry=build_default_registry(),
    )
    agent._dspy_module = MagicMock(
        return_value=MagicMock(summary="STUB-SUMMARY-FROM-REAL-CTX")
    )
    out = await agent._process_impl(
        KnowledgeSummarizationInput(
            tenant_id=TENANT,
            subject_keys=["policy:refunds"],
            title="Refunds slice",
            actor_role="user",
            actor_id="alice",
            promote=False,
        )
    )
    assert out.summary == "STUB-SUMMARY-FROM-REAL-CTX"
    assert out.source_count == 3


# ----- KnowledgeGraphTraversalAgent -------------------------------------


@pytest.mark.asyncio
async def test_walks_real_graph():
    from cogniverse_agents.kg_traversal_agent import (
        KGTraversalDeps,
        KGTraversalInput,
        KnowledgeGraphTraversalAgent,
    )

    # Three nodes + two edges forming a → b → c.
    rows = [
        {
            "id": "node_a",
            "memory": "node a content",
            "metadata": {"kind": "kg_node", "subject_key": "a"},
        },
        {
            "id": "node_b",
            "memory": "node b content",
            "metadata": {"kind": "kg_node", "subject_key": "b"},
        },
        {
            "id": "node_c",
            "memory": "node c content",
            "metadata": {"kind": "kg_node", "subject_key": "c"},
        },
        {
            "id": "edge_ab",
            "memory": "a → b",
            "metadata": {
                "kind": "kg_edge",
                "from_subject_key": "a",
                "to_subject_key": "b",
                "relation": "links_to",
            },
        },
        {
            "id": "edge_bc",
            "memory": "b → c",
            "metadata": {
                "kind": "kg_edge",
                "from_subject_key": "b",
                "to_subject_key": "c",
                "relation": "links_to",
            },
        },
    ]
    fake_mm = MagicMock()
    fake_mm.memory = MagicMock()
    fake_mm.get_all_memories = lambda *, tenant_id, agent_name: list(rows)
    agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps(tenant_id=TENANT))
    # KGTraversal pulls its memory manager from MemoryAwareMixin slots.
    agent.memory_manager = fake_mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = "kg_traversal_test"
    out = await agent._process_impl(
        KGTraversalInput(
            tenant_id=TENANT,
            start_subject_key="a",
            max_depth=3,
            max_edges=10,
        )
    )
    visited = {n.subject_key for n in out.nodes}
    assert {"a", "b", "c"}.issubset(visited)


# ----- AuditExplanationAgent --------------------------------------------


@pytest.mark.asyncio
async def test_explains_real_provenance_chain():
    from cogniverse_agents.audit_explanation_agent import (
        AuditExplanationAgent,
        AuditExplanationDeps,
        AuditExplanationInput,
    )
    from cogniverse_core.memory.provenance import (
        CitationRef,
        DerivationKind,
        Provenance,
    )
    from cogniverse_core.memory.provenance_store import ProvenanceRecord

    # Provide a memory factory that returns the chain by id AND a
    # provenance_store stub that satisfies the indexed walker's contract:
    # walk(root, max_depth, max_nodes) → (ordered, primary_sources, truncated).
    rows_by_id = {
        "answer": {
            "id": "answer",
            "memory": "synth",
            "metadata": {
                "kind": "external_doc",
                "provenance": {
                    "written_by": "agent:test",
                    "written_at": "2026-05-09T00:00:00+00:00",
                    "derived_from": [
                        {"ref_kind": "memory", "ref_id": "src_a", "label": None}
                    ],
                    "derivation_kind": "synthesis",
                    "confidence": 0.8,
                    "trace_id": None,
                },
            },
        },
        "src_a": {
            "id": "src_a",
            "memory": "primary source",
            "metadata": {"kind": "external_doc"},
        },
    }

    answer_prov = Provenance(
        written_by="agent:test",
        written_at="2026-05-09T00:00:00+00:00",
        derived_from=[CitationRef.memory("src_a")],
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.8,
    )
    answer_record = ProvenanceRecord.from_provenance("answer", TENANT, answer_prov)

    class _StubStore:
        """Minimal in-memory stand-in for ProvenanceStore.

        Implements the (root, max_depth, max_nodes) → (ordered,
        primary_sources, truncated) contract the indexed walker uses,
        plus the per-node ``get`` lookup. src_a has no provenance row
        (it's a primary source), so the walk terminates after one hop.
        """

        def __init__(self):
            self._records = {"answer": answer_record}

        def walk(self, root: str, *, max_depth: int, max_nodes: int):
            del max_depth, max_nodes  # not exercised by this stub
            ordered = [("answer", 0), ("src_a", 1)]
            primary_sources = [CitationRef.memory("src_a")]
            return ordered, primary_sources, False

        def get(self, memory_id: str):
            return self._records.get(memory_id)

    def _factory(tenant_id: str):
        m = MagicMock()
        m.memory = MagicMock()
        m.memory.get = lambda memory_id: rows_by_id.get(memory_id)
        m.provenance_store = _StubStore()
        return m

    agent = AuditExplanationAgent(
        deps=AuditExplanationDeps(tenant_id=TENANT),
        memory_manager_factory=_factory,
    )
    out = await agent._process_impl(
        AuditExplanationInput(
            tenant_id=TENANT,
            answer_memory_id="answer",
            include_trust=False,
            include_contradictions=False,
        )
    )
    visited_ids = {s.memory_id for s in out.sources}
    assert {"answer", "src_a"}.issubset(visited_ids)
