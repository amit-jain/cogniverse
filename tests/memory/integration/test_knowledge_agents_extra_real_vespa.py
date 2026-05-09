"""Real-Vespa integration tests for the knowledge agents that previously
had only factory-injected MagicMock coverage.

A companion file (``test_knowledge_agents_real_vespa.py``) covers
ContradictionReconciliation, KnowledgeSummarization, and AuditExplanation
against real Mem0 + Vespa. This file fills in:

  * MultiDocumentSynthesisAgent — input documents pulled from real Vespa
    via memory_id refs; agent persists synthesis with provenance back to
    the same store.
  * KnowledgeGraphTraversalAgent — kg_node + kg_edge memories seeded
    with subject_keys; agent walks the graph by querying real Vespa.
  * CrossTenantComparisonAgent — two tenants under the same org; agent
    reads from both via the federation path.
  * TemporalReasoningAgent — same subject_key with three different
    ``written_at`` stamps; agent slices by time windows.
  * FederatedQueryAgent — two tenants, query string matches in both;
    agent merges the hits.

The DSPy LLM modules are stubbed where each agent uses one — these
tests assert the persistence + traversal wires, not the LLM synthesis
quality (which is already covered in the agent unit tests and dashboard
A/B suites).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "test_tenant"
AGENT = "knowledge_agents_extra"


def _build_manager(
    tenant_id: str, shared_memory_vespa, shared_denseon, *, auto_create: bool
):
    Mem0MemoryManager._instances.pop(tenant_id, None)
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=auto_create,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    return mm


@pytest.fixture(scope="module")
def primary_mm(shared_memory_vespa, shared_denseon):
    Mem0MemoryManager._instances.clear()
    mm = _build_manager(TENANT, shared_memory_vespa, shared_denseon, auto_create=False)
    yield mm
    for cleanup_agent in (
        AGENT,
        "multi_document_synthesis_agent",
        "kg_traversal_agent",
    ):
        try:
            mm.clear_agent_memory(TENANT, cleanup_agent)
        except Exception:
            pass
    Mem0MemoryManager._instances.clear()


def _inject_memory(agent, mm, agent_name: str) -> None:
    """Wire the per-tenant manager onto an agent's mixin attributes."""
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = mm.tenant_id
    agent._memory_agent_name = agent_name


# ----- MultiDocumentSynthesisAgent --------------------------------------


@pytest.mark.asyncio
async def test_multi_doc_synthesis_real_vespa(primary_mm):
    """Seed 3 docs, ask the agent to synthesise across them, assert the
    persisted synthesis carries citations to all three input ids."""
    from cogniverse_agents.multi_document_synthesis_agent import (
        DocumentRef,
        MultiDocSynthesisDeps,
        MultiDocSynthesisInput,
        MultiDocumentSynthesisAgent,
    )

    mm = primary_mm
    doc_ids: list[str] = []
    for i in range(3):
        prov = make_provenance(
            written_by=f"agent:doc_{i}",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
            derived_from=[CitationRef.external(f"https://docs/multi_doc_{i}")],
        )
        mid = mm.add_memory(
            content=f"fact {i}: refunds policy detail",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata=attach_to_metadata({"kind": "external_doc"}, prov),
            infer=False,
        )
        assert mid
        doc_ids.append(mid)

    agent = MultiDocumentSynthesisAgent(deps=MultiDocSynthesisDeps(tenant_id=TENANT))
    _inject_memory(agent, mm, "multi_document_synthesis_agent")
    # Stub the DSPy module so the test doesn't require an LLM.
    agent._dspy_module = MagicMock(return_value=MagicMock(answer="STUB-SYNTH"))

    out = await agent._process_impl(
        MultiDocSynthesisInput(
            tenant_id=TENANT,
            query="What's the refund policy?",
            documents=[DocumentRef(memory_id=mid) for mid in doc_ids],
            persist=True,
        )
    )
    assert out.answer == "STUB-SYNTH"
    cited_ids = {
        ref["ref_id"] for ref in out.citation_refs if ref.get("ref_kind") == "memory"
    }
    for mid in doc_ids:
        assert mid in cited_ids, (
            f"synthesis citation_refs must reference every input document; "
            f"missing {mid} in {cited_ids!r}"
        )
    assert out.persisted_memory_id, (
        "persist=True must yield a persisted_memory_id from the new synthesis row"
    )

    # The persisted synthesis row must round-trip through real Vespa with
    # provenance pointing at the input doc ids. The agent persists under
    # its own agent_name namespace ("multi_document_synthesis_agent"),
    # NOT under whatever name we used to seed the inputs.
    rows = mm.get_all_memories(
        tenant_id=TENANT, agent_name="multi_document_synthesis_agent"
    )
    persisted = next(
        (r for r in rows if str(r.get("id")) == out.persisted_memory_id), None
    )
    assert persisted is not None, (
        f"persisted synthesis {out.persisted_memory_id} not visible in Vespa"
    )
    persisted_meta = persisted.get("metadata") or {}
    persisted_prov = persisted_meta.get("provenance") or {}
    derived = {d.get("ref_id") for d in persisted_prov.get("derived_from") or []}
    for mid in doc_ids:
        assert mid in derived, (
            f"persisted provenance must cite input doc {mid}; got {derived!r}"
        )


# ----- KGTraversalAgent -------------------------------------------------


@pytest.mark.asyncio
async def test_kg_traversal_real_vespa(primary_mm):
    """Seed a tiny knowledge graph (3 nodes + 2 edges) and assert the
    walker reaches every node from the start_subject_key."""
    from cogniverse_agents.kg_traversal_agent import (
        KGTraversalDeps,
        KGTraversalInput,
        KnowledgeGraphTraversalAgent,
    )

    mm = primary_mm
    # KGTraversalAgent reads via mm.get_all_memories with its own
    # agent_name namespace — seed under that name so the snapshot
    # picks them up.
    KG_AGENT = "kg_traversal_agent"

    def _seed_node(subject: str, label: str) -> str:
        prov = make_provenance(
            written_by="agent:kg",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
            derived_from=[CitationRef.external(f"https://kg/{subject}")],
        )
        mid = mm.add_memory(
            content=label,
            tenant_id=TENANT,
            agent_name=KG_AGENT,
            metadata=attach_to_metadata(
                {"kind": "kg_node", "subject_key": subject}, prov
            ),
            infer=False,
        )
        assert mid
        return mid

    def _seed_edge(from_subject: str, to_subject: str, relation: str) -> str:
        prov = make_provenance(
            written_by="agent:kg",
            derivation_kind=DerivationKind.EXTRACTION,
            confidence=0.8,
            derived_from=[CitationRef.external("https://kg/edges")],
        )
        mid = mm.add_memory(
            content=f"{from_subject} {relation} {to_subject}",
            tenant_id=TENANT,
            agent_name=KG_AGENT,
            metadata=attach_to_metadata(
                {
                    "kind": "kg_edge",
                    "subject_key": from_subject,
                    "from_subject_key": from_subject,
                    "to_subject_key": to_subject,
                    "relation": relation,
                },
                prov,
            ),
            infer=False,
        )
        assert mid
        return mid

    _seed_node("entity:a", "Entity A")
    _seed_node("entity:b", "Entity B")
    _seed_node("entity:c", "Entity C")
    _seed_edge("entity:a", "entity:b", "relates_to")
    _seed_edge("entity:b", "entity:c", "depends_on")

    agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps(tenant_id=TENANT))
    _inject_memory(agent, mm, KG_AGENT)

    out = await agent._process_impl(
        KGTraversalInput(
            tenant_id=TENANT,
            start_subject_key="entity:a",
            max_depth=3,
            max_edges=50,
        )
    )
    visited_subjects = {n.subject_key for n in out.nodes}
    assert {"entity:a", "entity:b", "entity:c"}.issubset(visited_subjects), (
        f"BFS from entity:a must reach b and c; got {visited_subjects!r}"
    )
    edge_pairs = {(e.from_subject_key, e.to_subject_key, e.relation) for e in out.edges}
    assert ("entity:a", "entity:b", "relates_to") in edge_pairs
    assert ("entity:b", "entity:c", "depends_on") in edge_pairs


# ----- TemporalReasoningAgent -------------------------------------------


@pytest.mark.asyncio
async def test_temporal_reasoning_real_vespa(primary_mm):
    """Seed three memories on the same subject_key with distinct
    ``written_at`` stamps; assert the agent slices each window correctly."""
    from cogniverse_agents.temporal_reasoning_agent import (
        TemporalReasoningAgent,
        TemporalReasoningDeps,
        TemporalReasoningInput,
        TimeWindow,
    )

    mm = primary_mm
    subject = "policy:evolving"

    now = datetime.now(timezone.utc)
    times = [
        now - timedelta(days=60),
        now - timedelta(days=30),
        now - timedelta(days=5),
    ]
    seeded_ids: list[str] = []
    for i, when in enumerate(times):
        # The temporal agent reads metadata.written_at directly to
        # bucket per window — set the top-level field to the back-dated
        # stamp so the test produces a deterministic distribution.
        mid = mm.add_memory(
            content=f"Policy revision {i}",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={
                "kind": "tenant_instruction",
                "subject_key": subject,
                "written_at": when.isoformat(),
            },
            infer=False,
        )
        assert mid
        seeded_ids.append(mid)

    agent = TemporalReasoningAgent(
        deps=TemporalReasoningDeps(tenant_id=TENANT),
        memory_manager_factory=lambda _tid: mm,
    )

    out = await agent._process_impl(
        TemporalReasoningInput(
            tenant_id=TENANT,
            subject_key=subject,
            agent_name_filter=AGENT,
            windows=[
                TimeWindow(
                    label="ancient",
                    start=(now - timedelta(days=90)).isoformat(),
                    end=(now - timedelta(days=45)).isoformat(),
                ),
                TimeWindow(
                    label="recent",
                    start=(now - timedelta(days=10)).isoformat(),
                    end=None,
                ),
            ],
        )
    )
    by_label = {w.label: w for w in out.window_views}
    assert seeded_ids[0] in by_label["ancient"].matching_memory_ids, (
        f"60-day-old revision must land in the 'ancient' window; "
        f"got matching={by_label['ancient'].matching_memory_ids!r}"
    )
    assert seeded_ids[2] in by_label["recent"].matching_memory_ids, (
        f"5-day-old revision must land in the 'recent' window; "
        f"got matching={by_label['recent'].matching_memory_ids!r}"
    )


# ----- multi-tenant fixtures (cross-tenant + federated agents) -----------


@pytest.fixture(scope="module")
def multitenant_setup(shared_memory_vespa, shared_denseon):
    """Two tenants under the same org so federation/cross-tenant agents
    have a real second tenant to read from."""
    Mem0MemoryManager._instances.clear()
    tenant_a = "acme:cell_a"
    tenant_b = "acme:cell_b"
    mm_a = _build_manager(
        tenant_a, shared_memory_vespa, shared_denseon, auto_create=True
    )
    mm_b = _build_manager(
        tenant_b, shared_memory_vespa, shared_denseon, auto_create=True
    )
    # The federated/cross-tenant agents look up org-trunk too — initialise it.
    trunk = org_trunk_tenant_id(tenant_a)  # acme:_org_trunk
    mm_trunk = _build_manager(
        trunk, shared_memory_vespa, shared_denseon, auto_create=True
    )
    yield {"a": mm_a, "b": mm_b, "trunk": mm_trunk}
    for mm in (mm_a, mm_b, mm_trunk):
        try:
            mm.clear_agent_memory(mm.tenant_id, AGENT)
        except Exception:
            pass
    Mem0MemoryManager._instances.clear()


# ----- CrossTenantComparisonAgent ---------------------------------------


@pytest.mark.asyncio
async def test_cross_tenant_comparison_real_vespa(multitenant_setup):
    """Two tenants assert different content for the same subject_key;
    agent surfaces both views."""
    from cogniverse_agents.cross_tenant_comparison_agent import (
        CrossTenantComparisonAgent,
        CrossTenantComparisonDeps,
        CrossTenantComparisonInput,
    )

    mm_a = multitenant_setup["a"]
    mm_b = multitenant_setup["b"]
    subject = "policy:jurisdiction"

    for mm, sentence in [(mm_a, "Cell A says rule X"), (mm_b, "Cell B says rule Y")]:
        prov = make_provenance(
            written_by="agent:policy",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
            derived_from=[CitationRef.external("https://docs/c33")],
        )
        mid = mm.add_memory(
            content=sentence,
            tenant_id=mm.tenant_id,
            agent_name=AGENT,
            metadata=attach_to_metadata(
                {"kind": "tenant_instruction", "subject_key": subject}, prov
            ),
            infer=False,
        )
        assert mid

    agent = CrossTenantComparisonAgent(
        deps=CrossTenantComparisonDeps(tenant_id=mm_a.tenant_id),
        memory_manager_factory=lambda tid: Mem0MemoryManager(tid),
        registry=build_default_registry(),
    )

    out = await agent._process_impl(
        CrossTenantComparisonInput(
            tenant_id=mm_a.tenant_id,
            subject_key=subject,
            tenant_ids=[mm_a.tenant_id, mm_b.tenant_id],
            actor_role="org_admin",
            actor_id="alice",
            agent_name_filter=AGENT,
        )
    )
    assert out.subject_key == subject
    by_tenant = {v.tenant_id: v for v in out.tenant_views}
    assert mm_a.tenant_id in by_tenant
    assert mm_b.tenant_id in by_tenant
    assert by_tenant[mm_a.tenant_id].matching_memory_ids, (
        f"tenant A must contribute at least one row; got view={by_tenant[mm_a.tenant_id]!r}"
    )
    assert by_tenant[mm_b.tenant_id].matching_memory_ids, (
        f"tenant B must contribute at least one row; got view={by_tenant[mm_b.tenant_id]!r}"
    )
    assert out.distinct_signatures_count >= 2, (
        f"distinct content per tenant should yield ≥2 signatures; got {out.distinct_signatures_count}"
    )


# ----- FederatedQueryAgent ----------------------------------------------


@pytest.mark.asyncio
async def test_federated_query_real_vespa(multitenant_setup):
    """Both tenants carry a memory matching the query string; agent
    merges hits from both."""
    from cogniverse_agents.federated_query_agent import (
        FederatedQueryAgent,
        FederatedQueryDeps,
        FederatedQueryInput,
    )

    mm_a = multitenant_setup["a"]
    mm_b = multitenant_setup["b"]
    needle = "FEDERATED_NEEDLE_C37"

    for mm in (mm_a, mm_b):
        prov = make_provenance(
            written_by="agent:planner",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
            derived_from=[CitationRef.external("https://docs/c37")],
        )
        mid = mm.add_memory(
            content=f"This row contains {needle} in its body.",
            tenant_id=mm.tenant_id,
            agent_name=AGENT,
            metadata=attach_to_metadata({"kind": "tenant_instruction"}, prov),
            infer=False,
        )
        assert mid

    agent = FederatedQueryAgent(
        deps=FederatedQueryDeps(tenant_id=mm_a.tenant_id),
        memory_manager_factory=lambda tid: Mem0MemoryManager(tid),
        registry=build_default_registry(),
    )

    out = await agent._process_impl(
        FederatedQueryInput(
            tenant_id=mm_a.tenant_id,
            query=needle,
            tenant_ids=[mm_a.tenant_id, mm_b.tenant_id],
            actor_role="org_admin",
            actor_id="alice",
            agent_name_filter=AGENT,
            top_k_per_tenant=20,
        )
    )
    hit_tenants = {h.tenant_id for h in out.hits}
    assert mm_a.tenant_id in hit_tenants, (
        f"federated query must surface a hit from tenant A; hits={out.hits!r}"
    )
    assert mm_b.tenant_id in hit_tenants, (
        f"federated query must surface a hit from tenant B; hits={out.hits!r}"
    )
