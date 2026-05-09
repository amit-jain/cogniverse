"""AuditExplanationAgent (C3.9).

Given an answer the system produced (identified by its memory id),
explains *why* — which memories it was derived from (via the
provenance walker), the trust score on each source, any open
contradiction sets touching the same subject, and a short structured
explanation block. Read-only.

Composes:
  * :class:`ProvenanceWalker` (A.2) for the citation chain
  * :func:`extract_trust` / :func:`apply_decay` (A.4) for trust scoring
  * :class:`ContradictionDetector` (A.3) for conflict awareness

This agent ships the audit-explanation surface that compliance
deployments need: every claim traceable to its sources, with the trust
weight that drove ranking and any disputes that were active at answer
time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.memory.contradiction import ContradictionDetector
from cogniverse_core.memory.provenance import ProvenanceWalker
from cogniverse_core.memory.trust import apply_decay, extract_trust

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8027


class AuditExplanationInput(AgentInput):
    tenant_id: Optional[str] = Field(None)
    answer_memory_id: str = Field(
        ...,
        min_length=1,
        description="ID of the answer memory whose derivation we explain",
    )
    include_trust: bool = Field(
        True, description="Compute decayed trust scores per source memory"
    )
    include_contradictions: bool = Field(
        True,
        description=(
            "Detect contradictions among the derivation set's subjects "
            "(uses ContradictionDetector)"
        ),
    )
    max_chain_depth: int = Field(10, ge=1, le=25)
    max_chain_nodes: int = Field(100, ge=1, le=500)


class SourceExplanationOut(AgentInput):
    """Per-source audit row."""

    memory_id: str
    depth: int
    excerpt: str
    written_by: Optional[str] = None
    derivation_kind: Optional[str] = None
    confidence: Optional[float] = None
    trust_score: Optional[float] = None
    trust_endorsements: Optional[int] = None


class ContradictionOut(AgentInput):
    subject_key: str
    conflicting_memory_ids: List[str]


class AuditExplanationOutput(AgentOutput):
    answer_memory_id: str
    sources: List[SourceExplanationOut] = Field(default_factory=list)
    primary_sources: List[Dict[str, Any]] = Field(default_factory=list)
    contradictions_touched: List[ContradictionOut] = Field(default_factory=list)
    explanation: str = Field(
        "",
        description=(
            "Human-readable structured explanation; usable as-is in audit "
            "UIs without further LLM post-processing."
        ),
    )
    truncated_chain: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditExplanationDeps(AgentDeps):
    pass


def _format_explanation(
    answer_memory_id: str,
    sources: List[SourceExplanationOut],
    contradictions: List[ContradictionOut],
    truncated: bool,
) -> str:
    lines: List[str] = [f"Answer memory: {answer_memory_id}"]
    if not sources:
        lines.append("(no derivation chain — answer has no provenance graph)")
    else:
        lines.append(f"Derivation chain ({len(sources)} sources):")
        for s in sources:
            indent = "  " * (s.depth + 1)
            trust = f", trust={s.trust_score:.3f}" if s.trust_score is not None else ""
            kind = f", kind={s.derivation_kind}" if s.derivation_kind else ""
            lines.append(
                f"{indent}- depth={s.depth} mem={s.memory_id}{kind}{trust}: "
                f"{s.excerpt[:120]}"
            )
    if contradictions:
        lines.append(f"Contradictions touched ({len(contradictions)}):")
        for c in contradictions:
            lines.append(
                f"  - subject={c.subject_key}: members={c.conflicting_memory_ids}"
            )
    if truncated:
        lines.append("[chain truncated at max_depth/max_nodes]")
    return "\n".join(lines)


class AuditExplanationAgent(
    MemoryAwareMixin,
    A2AAgent[AuditExplanationInput, AuditExplanationOutput, AuditExplanationDeps],
):
    """A2A agent that explains why a given answer memory was produced."""

    def __init__(
        self,
        deps: AuditExplanationDeps,
        memory_manager_factory=None,
        port: int = _DEFAULT_PORT,
    ):
        config = A2AAgentConfig(
            agent_name="audit_explanation_agent",
            agent_description=(
                "Explains why a system answer was produced: walks "
                "provenance, surfaces trust scores per source, and "
                "flags any contradictions touching the same subjects."
            ),
            capabilities=["audit_explanation", "audit", "provenance_consumer"],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        self._mm_factory = memory_manager_factory

    async def _process_impl(
        self, input: AuditExplanationInput
    ) -> AuditExplanationOutput:
        if self._mm_factory is None:
            from cogniverse_core.memory.manager import Mem0MemoryManager

            self._mm_factory = lambda tid: Mem0MemoryManager(tenant_id=tid)

        tenant_id = input.tenant_id or getattr(self.deps, "tenant_id", None)
        if not tenant_id:
            raise ValueError("AuditExplanationAgent: no tenant_id on input or deps")

        try:
            mm = self._mm_factory(tenant_id)
        except Exception as exc:
            raise ValueError(
                f"AuditExplanationAgent: factory({tenant_id}) failed: {exc}"
            ) from exc

        walker = ProvenanceWalker(
            mm,
            max_depth=input.max_chain_depth,
            max_nodes=input.max_chain_nodes,
        )
        graph = walker.walk(input.answer_memory_id, tenant_id)

        sources: List[SourceExplanationOut] = []
        for node in graph.nodes:
            mem = self._fetch_memory(mm, tenant_id, node.memory_id)
            trust_score: Optional[float] = None
            trust_endorsements: Optional[int] = None
            if input.include_trust and mem is not None:
                trust = extract_trust(mem)
                if trust is not None:
                    decayed = apply_decay(trust)
                    trust_score = decayed.score
                    trust_endorsements = decayed.endorsements

            written_by = None
            derivation_kind = None
            confidence = None
            if node.provenance is not None:
                p = node.provenance
                written_by = p.written_by
                derivation_kind = (
                    p.derivation_kind.value
                    if hasattr(p.derivation_kind, "value")
                    else str(p.derivation_kind)
                )
                confidence = p.confidence

            sources.append(
                SourceExplanationOut(
                    memory_id=node.memory_id,
                    depth=node.depth,
                    excerpt=node.content_excerpt,
                    written_by=written_by,
                    derivation_kind=derivation_kind,
                    confidence=confidence,
                    trust_score=trust_score,
                    trust_endorsements=trust_endorsements,
                )
            )

        contradictions_out: List[ContradictionOut] = []
        if input.include_contradictions and sources:
            candidates: List[Dict[str, Any]] = []
            for src in sources:
                m = self._fetch_memory(mm, tenant_id, src.memory_id)
                if m is not None:
                    candidates.append(m)
            detector = ContradictionDetector()
            for cs in detector.detect(candidates):
                contradictions_out.append(
                    ContradictionOut(
                        subject_key=cs.subject_key,
                        conflicting_memory_ids=cs.conflicting_memory_ids,
                    )
                )

        primary_sources = [
            {
                "ref_kind": ref.ref_kind,
                "ref_id": ref.ref_id,
                "label": ref.label,
            }
            for ref in graph.primary_sources
        ]

        explanation = _format_explanation(
            answer_memory_id=input.answer_memory_id,
            sources=sources,
            contradictions=contradictions_out,
            truncated=graph.truncated_at_max_depth,
        )

        return AuditExplanationOutput(
            answer_memory_id=input.answer_memory_id,
            sources=sources,
            primary_sources=primary_sources,
            contradictions_touched=contradictions_out,
            explanation=explanation,
            truncated_chain=graph.truncated_at_max_depth,
            metadata={
                "source_count": len(sources),
                "primary_source_count": len(primary_sources),
                "contradiction_count": len(contradictions_out),
                "tenant_id": tenant_id,
            },
        )

    @staticmethod
    def _fetch_memory(mm, tenant_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one memory dict by id; tolerates Mem0's varying return shapes."""
        try:
            getter = getattr(mm, "memory", None)
            if getter is None or not hasattr(getter, "get"):
                return None
            mem_obj = getter.get(memory_id)
        except Exception as exc:
            logger.debug("audit: get(%s) failed: %s", memory_id, exc)
            return None
        if mem_obj is None:
            return None
        if isinstance(mem_obj, dict):
            return mem_obj
        if isinstance(mem_obj, list) and mem_obj:
            return mem_obj[0] if isinstance(mem_obj[0], dict) else None
        return None
