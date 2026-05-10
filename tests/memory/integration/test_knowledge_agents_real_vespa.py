"""Knowledge agents against real Vespa-backed Mem0.

The previous round shipped ``test_knowledge_agents_real_mem0.py (renamed)`` which the
audit caught was misleadingly named — every test mocked the memory
manager with ``MagicMock``, so the agents never touched a real
backend. This file replaces the missing real-service coverage by
exercising the knowledge agents that are USEFUL to test against real
Vespa+Mem0:

  * CitationTracingAgent — already covered by the existing
    ``test_citation_tracing_agent_integration.py`` (kept as-is).
  * AuditExplanationAgent — composes ProvenanceWalker + trust +
    contradiction; needs real persisted provenance + trust.
  * ContradictionReconciliationAgent — reconciles real ConflictSets
    written by the detector against real memories.
  * KnowledgeSummarizationAgent — reads a real subject slice and
    optionally promotes to org trunk via real FederationService.

The other 4 (multi_doc_synth, kg_traversal, cross_tenant, federated_query,
temporal_reasoning) are exercised via factory-injection in the original
fix-up file because their dispatching is per-tenant fan-out — real
multi-tenant Vespa setup adds setup cost without proving more than
the factory-injected version. The existing tests still verify those
agents against real data shapes; this file fills the gap for the
agents whose value comes from real persistence round-trips.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pytest

from cogniverse_agents.audit_explanation_agent import (
    AuditExplanationAgent,
    AuditExplanationDeps,
    AuditExplanationInput,
)
from cogniverse_agents.knowledge_summarization_agent import (
    KnowledgeSummarizationAgent,
    KnowledgeSummarizationDeps,
    KnowledgeSummarizationInput,
)
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
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "test_tenant"
AGENT = "knowledge_agents_real"


@pytest.fixture(scope="module")
def real_mm(shared_memory_vespa, shared_denseon):
    """Real Mem0+Vespa manager wired with the schema registry so add_memory
    enforces provenance and auto-attaches trust."""
    Mem0MemoryManager._instances.clear()
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
    mm = Mem0MemoryManager(tenant_id=TENANT)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://localhost:11434",
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    yield mm
    try:
        mm.clear_agent_memory(TENANT, AGENT)
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


# ----- AuditExplanationAgent against real Mem0+Vespa --------------------


@pytest.mark.asyncio
async def test_audit_walks_real_provenance_chain(real_mm: Mem0MemoryManager):
    """Write a real chain to Vespa: leaf → derived. AuditExplanationAgent
    must walk it back via the real ProvenanceWalker."""
    # Leaf: direct ingest from an external source.
    leaf_prov = make_provenance(
        written_by="agent:ingest",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("https://wiki/audit_leaf")],
    )
    leaf_id = real_mm.add_memory(
        content="leaf — primary source",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, leaf_prov),
        infer=False,
    )
    assert leaf_id

    # Derived: synthesises from the leaf.
    derived_prov = make_provenance(
        written_by="agent:synthesizer",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.85,
        derived_from=[CitationRef.memory(leaf_id, label="primary")],
    )
    derived_id = real_mm.add_memory(
        content="derived synthesis from leaf",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata({"kind": "entity_fact"}, derived_prov),
        infer=False,
    )
    assert derived_id

    # Run the audit agent against the real chain.
    agent = AuditExplanationAgent(
        deps=AuditExplanationDeps(tenant_id=TENANT),
        memory_manager_factory=lambda _t: real_mm,
    )
    out = await agent._process_impl(
        AuditExplanationInput(
            tenant_id=TENANT,
            answer_memory_id=derived_id,
            include_trust=False,
            include_contradictions=False,
        )
    )
    visited = {s.memory_id for s in out.sources}
    assert derived_id in visited, "audit must include the answer node"
    assert leaf_id in visited, (
        "the walker must follow derived_from from the synthesis to the "
        f"leaf; visited={visited!r}. If the leaf is missing, metadata "
        "round-trip dropped provenance.derived_from on the read path."
    )
    primary_refs = {p["ref_id"] for p in out.primary_sources}
    assert "https://wiki/audit_leaf" in primary_refs, (
        "the external citation on the leaf must surface as a primary "
        f"source after the walk reaches it; got primary_refs={primary_refs!r}"
    )


# ----- KnowledgeSummarizationAgent against real Mem0+Vespa --------------


@pytest.mark.asyncio
async def test_knowledge_summarises_real_subject_slice(
    real_mm: Mem0MemoryManager, dspy_lm
):
    """Real summary must (a) read all 3 seeded memories, (b) not be a
    verbatim copy of any source, (c) reference content from ≥2 sources."""
    subject = "policy:refunds_summary"
    seed_facts: List[Tuple[str, str]] = [
        (
            "Standard refund policy: customers may request a refund within "
            "30 days of purchase, no questions asked.",
            "30",
        ),
        (
            "Customers in the European Union receive an additional 14 days "
            "of return rights under EU consumer protection law.",
            "14",
        ),
        (
            "Digital downloads are non-refundable once the file has been "
            "accessed; refund requests on accessed downloads are denied.",
            "digital",
        ),
    ]
    written_ids = []
    for i, (content, _expected) in enumerate(seed_facts):
        prov = make_provenance(
            written_by=f"agent:doc_{i}",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.8,
            derived_from=[CitationRef.external(f"https://docs/refunds_{i}")],
        )
        mid = real_mm.add_memory(
            content=content,
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata=attach_to_metadata(
                {"kind": "external_doc", "subject_key": subject}, prov
            ),
            infer=False,
        )
        assert mid, f"write {i} failed"
        written_ids.append(mid)

    agent = KnowledgeSummarizationAgent(
        deps=KnowledgeSummarizationDeps(tenant_id=TENANT),
        memory_manager_factory=lambda _t: real_mm,
        registry=build_default_registry(),
    )
    out = await agent._process_impl(
        KnowledgeSummarizationInput(
            tenant_id=TENANT,
            subject_keys=[subject],
            kinds=["external_doc"],
            agent_name_filter=AGENT,
            title="refunds policy summary",
            actor_role="user",
            actor_id="alice",
            promote=False,
        )
    )

    assert out.source_count == 3, out
    summary_text = out.summary or ""
    assert summary_text.strip(), f"empty summary: {summary_text!r}"
    # KnowledgeSummarizationAgent._summarise_without_rlm catches DSPy
    # failures and returns "[FALLBACK: ...] <source>" — would mask a
    # broken LM behind source-text echoes.
    assert not summary_text.startswith("[FALLBACK:"), summary_text[:300]

    summary_norm = " ".join(summary_text.split())
    for content, _ in seed_facts:
        source_norm = " ".join(content.split())
        assert source_norm not in summary_norm, (
            f"summary echoes source verbatim: {source_norm!r}\n"
            f"full summary: {summary_text!r}"
        )

    summary_lower = summary_text.lower()
    # Each source's distinguishing fact appears in the summary, AND
    # surrounded by enough context to prove it isn't word-salad.
    # (token-only checks pass on garbage like "30 days 14 days digital".)
    assert "30 days" in summary_lower, summary_text
    assert "14 days" in summary_lower or "14 additional" in summary_lower, summary_text
    assert "digital" in summary_lower, summary_text
    # The summary is prose: long enough to contain the 3 facts in
    # context, bounded so we catch a runaway repetition. Three short
    # facts → ~150-1500 chars of natural prose.
    assert 150 <= len(summary_text) <= 1500, (
        f"summary length {len(summary_text)} chars outside the prose range "
        f"[150, 1500]; either truncated/garbage or runaway. "
        f"summary={summary_text!r}"
    )
    # Multi-sentence — a real synthesis of 3 facts is not one sentence.
    assert summary_text.count(".") >= 2, (
        f"summary has < 2 sentence terminators; likely a fragment, not a "
        f"synthesis. summary={summary_text!r}"
    )
    cited = {ref.ref_id for ref in out.citation_refs}
    for mid in written_ids:
        assert mid in cited, (
            f"memory {mid} written to real Vespa was not cited; the "
            f"agent's get_all_memories slice missed it. cited={cited!r}"
        )


# ----- ContradictionReconciliationAgent on real conflicting memories ---


@pytest.mark.asyncio
async def test_contradiction_reconciles_real_conflicting_memories(
    real_mm: Mem0MemoryManager,
):
    """Write two real memories disagreeing on the same subject_key,
    then ask the reconciliation agent to resolve them. The detector +
    reconciler operate over real Mem0 reads, not synthetic dicts."""
    subject = "france:capital"

    prov_a = make_provenance(
        written_by="agent:doc_a",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("https://wiki/paris")],
    )
    id_a = real_mm.add_memory(
        content="Paris is the capital of France",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata(
            {"kind": "entity_fact", "subject_key": subject}, prov_a
        ),
        infer=False,
    )

    prov_b = make_provenance(
        written_by="agent:doc_b",
        derivation_kind=DerivationKind.AGENT_INFERENCE,
        confidence=0.4,
        derived_from=[CitationRef.external("https://wiki/lyon-disputed")],
    )
    id_b = real_mm.add_memory(
        content="Lyon is the capital of France",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata=attach_to_metadata(
            {"kind": "entity_fact", "subject_key": subject}, prov_b
        ),
        infer=False,
    )
    assert id_a and id_b

    from cogniverse_agents.contradiction_reconciliation_agent import (
        ContradictionReconciliationAgent,
        ContradictionReconciliationDeps,
        ContradictionReconciliationInput,
    )

    # ContradictionReconciliationAgent reads via MemoryAwareMixin —
    # inject the real manager via the mixin attributes, not a factory
    # kwarg (the agent's constructor doesn't take a factory).
    agent = ContradictionReconciliationAgent(
        deps=ContradictionReconciliationDeps(tenant_id=TENANT),
        registry=build_default_registry(),
    )
    agent.memory_manager = real_mm
    agent._memory_initialized = True
    agent._memory_tenant_id = TENANT
    agent._memory_agent_name = AGENT
    out = await agent._process_impl(
        ContradictionReconciliationInput(
            tenant_id=TENANT,
            target_kind="entity_fact",
            conflict_member_ids=[id_a, id_b],
        )
    )
    # The agent must produce a resolved view over the real conflict set.
    assert out is not None
    resolved_ids = {m.memory_id for m in out.resolved}
    assert resolved_ids == {id_a, id_b}, (
        "reconciliation must process both members of the conflict set; "
        f"got resolved_ids={resolved_ids!r}, expected={{{id_a!r}, {id_b!r}}}"
    )
    # trust_ranked policy: doc_a (DIRECT_INGEST conf=0.9) must outrank
    # doc_b (AGENT_INFERENCE conf=0.4). The high-trust member survives;
    # the low-trust one is excluded.
    assert out.policy_used == "trust_ranked", (
        f"entity_fact schema policy is trust_ranked; got {out.policy_used!r}"
    )
    assert out.survivors == [id_a], (
        "trust_ranked must keep the higher-trust DIRECT_INGEST source "
        f"({id_a}) and drop the lower-trust AGENT_INFERENCE source "
        f"({id_b}); got survivors={out.survivors!r}"
    )
    survived_by_id = {m.memory_id: m.survived for m in out.resolved}
    assert survived_by_id[id_a] is True
    assert survived_by_id[id_b] is False
