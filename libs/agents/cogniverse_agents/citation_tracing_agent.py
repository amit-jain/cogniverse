"""CitationTracingAgent.

Read-only agent that walks a memory's provenance chain back to its primary
sources. Wraps :class:`ProvenanceWalker` inside the standard A2A +
MemoryAware harness so the orchestrator can dispatch citation queries the
same way it dispatches search.

Use cases:
  * "show me the sources for this answer" — operator audit, trust dashboard
  * compliance: walk a synthesised claim back to its primary citations
  * debugging: see why a particular memory was promoted by the system

The agent does NOT write to memory and does NOT call out to the LLM. It is
deterministic and cheap; orchestrator can chain it after any synthesis
agent that returned a memory id.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from cogniverse_agents.graph_bindable import GraphBindableMixin
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class CitationTracingInput(AgentInput):
    """What memory to trace, plus walk parameters."""

    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    query: str = Field(
        "",
        description=(
            "Optional context label for telemetry; the actual trace target is "
            "memory_id. Kept on the AgentInput contract for orchestrator "
            "compatibility."
        ),
    )
    memory_id: str = Field(..., description="ID of the memory whose chain to walk")
    claim_id: Optional[str] = Field(
        None,
        description=(
            "Optional KG Edge ``edge_id`` to ground. When set and a graph is "
            "bound at dispatch, the agent surfaces the claim's segment-level "
            "KG provenance in ``kg_primary_sources``, complementary to the mem0 "
            "provenance walk over ``memory_id``."
        ),
    )
    max_depth: int = Field(
        10,
        ge=1,
        le=25,
        description="Stop walking past this depth in the citation chain",
    )
    max_nodes: int = Field(
        100,
        ge=1,
        le=500,
        description="Stop after visiting this many nodes total (cycle protection)",
    )


class CitationNodeOut(AgentInput):
    """Serialisable view of a single citation chain node."""

    memory_id: str
    depth: int
    content_excerpt: str
    written_by: Optional[str] = None
    derivation_kind: Optional[str] = None
    confidence: Optional[float] = None


class CitationRefOut(AgentInput):
    """Serialisable view of a primary-source citation reference."""

    ref_kind: str
    ref_id: str
    label: Optional[str] = None


class KGGroundingStepOut(AgentInput):
    """One segment-level grounding step for a KG claim (Edge), from the shared
    Vespa knowledge graph (bound at dispatch). Complements the mem0 provenance
    walk — it carries the media-segment provenance the per-agent memory lacks."""

    source_doc_id: str
    segment_id: str
    ts_start: float
    ts_end: float
    modality: str
    evidence_span: str
    node_name: str = Field(..., description="The claim subject (source_node_id)")
    predicate: str = Field(..., description="The claim relation")


class CitationTracingOutput(AgentOutput):
    """Walked chain plus structured primary-source list for UI rendering."""

    root_memory_id: str = Field(..., description="The starting memory id")
    nodes: List[CitationNodeOut] = Field(
        default_factory=list,
        description="Chain nodes in BFS order (root at depth 0)",
    )
    primary_sources: List[CitationRefOut] = Field(
        default_factory=list,
        description=(
            "Terminal sources — non-memory refs (URLs/docs) and memory leaves "
            "without further provenance"
        ),
    )
    kg_primary_sources: List[KGGroundingStepOut] = Field(
        default_factory=list,
        description=(
            "Segment-level KG grounding for ``claim_id`` from the shared Vespa "
            "knowledge graph (bound at dispatch). Empty when no graph is bound "
            "or no claim_id is given. Complements the mem0 provenance walk."
        ),
    )
    truncated: bool = Field(
        False,
        description="True when traversal stopped at max_depth / max_nodes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Telemetry: nodes_visited, primary_source_count",
    )


class CitationTracingDeps(AgentDeps):
    """Tenant-agnostic deps; walker bounds are per-request via input."""

    pass


class CitationTracingAgent(
    GraphBindableMixin,
    MemoryAwareMixin,
    A2AAgent[CitationTracingInput, CitationTracingOutput, CitationTracingDeps],
):
    """Read-only A2A agent that surfaces citation chains via ProvenanceWalker."""

    def __init__(self, deps: CitationTracingDeps, port: int = 8019):
        config = A2AAgentConfig(
            agent_name="citation_tracing_agent",
            agent_description=(
                "Walks the citation/provenance chain of a memory back to its "
                "primary sources for audit and citation rendering."
            ),
            capabilities=[
                "citation_tracing",
                "provenance_walk",
                "audit",
            ],
            port=port,
        )
        super().__init__(deps=deps, config=config)

    def trace(self, claim_id: str) -> Dict[str, Any]:
        """Walk an Edge (claim) back to its grounding Mention(s).

        Returns ``{"chain": [<step>, ...]}`` where each step is a dict with
        keys: ``source_doc_id``, ``segment_id``, ``ts_start``, ``ts_end``,
        ``modality``, ``evidence_span``, ``node_name`` (subject), ``predicate``.

        ``claim_id`` is the Edge ``edge_id`` (the SHA1-16 prefix of the
        normalised triple+segment), matching ``edge_id_of(source, relation, target)``
        in the test fixtures.
        """
        graph_manager = self._require_graph_manager("trace")
        edge_fields = graph_manager.get_edge_by_id(claim_id)
        chain: List[Dict[str, Any]] = []
        if edge_fields:
            chain.append(
                {
                    "source_doc_id": str(edge_fields.get("source_doc_id") or ""),
                    "segment_id": str(edge_fields.get("segment_id") or ""),
                    "ts_start": float(edge_fields.get("ts_start") or 0.0),
                    "ts_end": float(edge_fields.get("ts_end") or 0.0),
                    "modality": str(edge_fields.get("modality") or ""),
                    "evidence_span": str(edge_fields.get("evidence_span") or ""),
                    "node_name": str(edge_fields.get("source_node_id") or ""),
                    "predicate": str(edge_fields.get("relation") or ""),
                }
            )
        return {"chain": chain}

    def _kg_primary_sources(
        self, input: CitationTracingInput
    ) -> List[KGGroundingStepOut]:
        """Ground ``claim_id`` to its KG segment provenance via the bound graph,
        complementary to the mem0 provenance walk. Empty when no graph is bound
        or no claim_id is given."""
        if self._graph_manager is None or not input.claim_id:
            return []
        try:
            traced = self.trace(input.claim_id)
        except Exception as exc:
            logger.debug(
                "citation: Vespa-KG complement skipped for claim %s: %s",
                input.claim_id,
                exc,
            )
            return []
        return [
            KGGroundingStepOut(
                source_doc_id=str(step.get("source_doc_id") or ""),
                segment_id=str(step.get("segment_id") or ""),
                ts_start=float(step.get("ts_start") or 0.0),
                ts_end=float(step.get("ts_end") or 0.0),
                modality=str(step.get("modality") or ""),
                evidence_span=str(step.get("evidence_span") or ""),
                node_name=str(step.get("node_name") or ""),
                predicate=str(step.get("predicate") or ""),
            )
            for step in traced.get("chain", [])
        ]

    async def _process_impl(self, input: CitationTracingInput) -> CitationTracingOutput:
        """Walk the chain and return a structured graph."""
        from cogniverse_core.memory.provenance import ProvenanceWalker

        kg_primary_sources = self._kg_primary_sources(input)

        if not self.is_memory_enabled() or self.memory_manager is None:
            logger.warning(
                "CitationTracingAgent invoked without a memory manager; "
                "returning KG-only grounding"
            )
            return CitationTracingOutput(
                root_memory_id=input.memory_id,
                nodes=[],
                primary_sources=[],
                kg_primary_sources=kg_primary_sources,
                truncated=False,
                metadata={
                    "reason": "memory_manager_unavailable",
                    "kg_primary_source_count": len(kg_primary_sources),
                },
            )

        walker = ProvenanceWalker(
            self.memory_manager,
            max_depth=input.max_depth,
            max_nodes=input.max_nodes,
        )
        # Use the agent's resolved tenant id, NOT input.tenant_id directly,
        # so the dispatcher's stamping always wins (input.tenant_id can be
        # missing when the orchestrator forwards an enrichment payload).
        tenant_id = getattr(self, "_memory_tenant_id", None) or input.tenant_id or ""
        graph = walker.walk(input.memory_id, tenant_id=tenant_id)

        nodes_out = [
            CitationNodeOut(
                memory_id=n.memory_id,
                depth=n.depth,
                content_excerpt=n.content_excerpt,
                written_by=(
                    n.provenance.written_by if n.provenance is not None else None
                ),
                derivation_kind=(
                    n.provenance.derivation_kind.value
                    if n.provenance is not None
                    else None
                ),
                confidence=(
                    n.provenance.confidence if n.provenance is not None else None
                ),
            )
            for n in graph.nodes
        ]
        sources_out = [
            CitationRefOut(
                ref_kind=r.ref_kind,
                ref_id=r.ref_id,
                label=r.label,
            )
            for r in graph.primary_sources
        ]

        return CitationTracingOutput(
            root_memory_id=graph.root_memory_id,
            nodes=nodes_out,
            primary_sources=sources_out,
            kg_primary_sources=kg_primary_sources,
            truncated=graph.truncated_at_max_depth,
            metadata={
                "nodes_visited": len(nodes_out),
                "primary_source_count": len(sources_out),
                "kg_primary_source_count": len(kg_primary_sources),
            },
        )
