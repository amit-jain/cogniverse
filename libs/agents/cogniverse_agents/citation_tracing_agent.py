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
    MemoryAwareMixin,
    A2AAgent[CitationTracingInput, CitationTracingOutput, CitationTracingDeps],
):
    """Read-only A2A agent that surfaces citation chains via ProvenanceWalker."""

    def __init__(
        self, deps: CitationTracingDeps, config_manager=None, port: int = 8019
    ):
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
        self._config_manager = config_manager

    async def _process_impl(self, input: CitationTracingInput) -> CitationTracingOutput:
        """Walk the chain and return a structured graph."""
        from cogniverse_core.memory.provenance import ProvenanceWalker

        if not self.is_memory_enabled() or self.memory_manager is None:
            logger.warning(
                "CitationTracingAgent invoked without a memory manager; "
                "returning empty chain"
            )
            return CitationTracingOutput(
                root_memory_id=input.memory_id,
                nodes=[],
                primary_sources=[],
                truncated=False,
                metadata={"reason": "memory_manager_unavailable"},
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
            truncated=graph.truncated_at_max_depth,
            metadata={
                "nodes_visited": len(nodes_out),
                "primary_source_count": len(sources_out),
            },
        )
