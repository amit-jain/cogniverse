"""
Deep Research Agent — multi-step research with iterative evidence gathering.

Decomposes complex queries into sub-tasks, dispatches parallel searches,
evaluates evidence sufficiency, iterates if gaps remain, then synthesizes
a structured research report with citations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions

logger = logging.getLogger(__name__)


class DeepResearchInput(AgentInput):
    query: str = Field(..., description="Research question")
    max_iterations: int = Field(3, description="Maximum research iterations")
    tenant_id: str = Field(..., description="Tenant identifier (required)")
    rlm: Optional[RLMOptions] = Field(
        None,
        description="RLM configuration. None=disabled, set RLMOptions to enable RLM synthesis over accumulated evidence",
    )


class Citation(AgentOutput):
    source: str = Field(..., description="Source identifier (span_id, doc_id, etc.)")
    text: str = Field(..., description="Cited passage or evidence")
    relevance: float = Field(0.0, description="Relevance score 0-1")


class DeepResearchOutput(AgentOutput):
    summary: str = Field(..., description="Synthesized research summary")
    sub_questions: List[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list, description="Collected evidence per sub-question"
    )
    citations: List[Dict[str, str]] = Field(
        default_factory=list, description="Citations with source and text"
    )
    iterations_used: int = Field(0, description="Number of research iterations used")
    gaps_remaining: List[str] = Field(
        default_factory=list, description="Unanswered sub-questions"
    )
    confidence: float = Field(0.0, description="Overall research confidence")
    rlm_synthesis: Optional[str] = Field(
        None,
        description="RLM-synthesized answer from accumulated evidence (only when RLM enabled)",
    )
    rlm_telemetry: Optional[Dict[str, Any]] = Field(
        None, description="RLM telemetry metrics for A/B testing"
    )


class DeepResearchDeps(AgentDeps):
    tenant_id: str


class TaskDecompositionSignature(dspy.Signature):
    """Decompose a complex research query into focused sub-questions."""

    query: str = dspy.InputField(desc="Complex research question")
    sub_questions: list[str] = dspy.OutputField(
        desc="3-5 focused sub-questions that together answer the main query"
    )


class EvidenceEvaluationSignature(dspy.Signature):
    """Evaluate whether collected evidence sufficiently answers the query."""

    query: str = dspy.InputField(desc="Original research question")
    sub_questions: str = dspy.InputField(desc="Sub-questions being researched")
    evidence_summary: str = dspy.InputField(desc="Summary of evidence collected so far")
    has_sufficient_evidence: bool = dspy.OutputField(
        desc="True if evidence is sufficient to answer the query"
    )
    gaps: list[str] = dspy.OutputField(
        desc="Sub-questions still lacking evidence (empty if sufficient)"
    )
    confidence: float = dspy.OutputField(desc="Confidence in evidence 0.0-1.0")


class SynthesisSignature(dspy.Signature):
    """Synthesize collected evidence into a research report."""

    query: str = dspy.InputField(desc="Original research question")
    evidence: str = dspy.InputField(desc="All collected evidence with sources")
    summary: str = dspy.OutputField(
        desc="Comprehensive research summary with inline citations"
    )


class DeepResearchAgent(
    MemoryAwareMixin,
    RLMAwareMixin,
    A2AAgent[DeepResearchInput, DeepResearchOutput, DeepResearchDeps],
):
    """
    Multi-step research agent with iterative evidence gathering.

    Flow:
    1. Decompose query into sub-questions (DSPy TaskDecomposition)
    2. Dispatch parallel searches for each sub-question
    3. Evaluate evidence sufficiency (DSPy EvidenceEvaluation)
    4. If gaps remain and iterations left → refine and search again
    5. Synthesize final report (DSPy Synthesis)
    """

    def __init__(
        self,
        deps: DeepResearchDeps,
        config: A2AAgentConfig | None = None,
        search_fn: Any = None,
    ):
        if config is None:
            config = A2AAgentConfig(
                agent_name="deep_research_agent",
                agent_description="Multi-step deep research with iterative evidence gathering",
                capabilities=["deep_research", "analysis"],
            )
        super().__init__(deps=deps, config=config)

        self._search_fn = search_fn
        self._decomposer = dspy.ChainOfThought(TaskDecompositionSignature)
        self._evaluator = dspy.ChainOfThought(EvidenceEvaluationSignature)
        self._synthesizer = dspy.ChainOfThought(SynthesisSignature)

    async def _process_impl(self, input: DeepResearchInput) -> DeepResearchOutput:
        # Set tenant for memory/instructions injection and enrich the query
        # with the full context stack (instructions + learned strategies +
        # tenant memories) before decomposition. Mirrors the pattern used
        # by SearchAgent and CodingAgent. No-ops gracefully when memory
        # isn't initialized.
        self.set_tenant_for_context(input.tenant_id)
        enriched_query = self.inject_context_into_prompt(input.query, input.query)

        self.emit_progress("decompose", "Decomposing research query...")

        sub_questions = await self._decompose(enriched_query)

        all_evidence: List[Dict[str, Any]] = []
        all_citations: List[Dict[str, str]] = []
        gaps = list(sub_questions)
        iteration = 0

        while gaps and iteration < input.max_iterations:
            iteration += 1
            self.emit_progress(
                "search",
                f"Iteration {iteration}: searching {len(gaps)} sub-questions...",
            )

            new_evidence = await self._search_parallel(gaps, input.tenant_id)
            all_evidence.extend(new_evidence)
            all_citations.extend(self._extract_citations(new_evidence))

            self.emit_progress(
                "evaluate", f"Evaluating evidence (iteration {iteration})..."
            )
            sufficient, gaps, confidence = await self._evaluate_evidence(
                input.query, sub_questions, all_evidence
            )

            if sufficient:
                break

        if iteration == 0:
            raise ValueError(
                "DeepResearchAgent._process_impl: no iterations ran — "
                "sub_questions list was empty after decomposition. "
                "The decomposer must return at least one sub-question."
            )

        self.emit_progress("synthesize", "Synthesizing research report...")
        summary = await self._synthesize(input.query, all_evidence)

        # Build flat evidence text for optional RLM synthesis over accumulated evidence
        evidence_context = "\n\n".join(
            f"## {e['question']}\n{e.get('results', 'No results')}"
            for e in all_evidence
        )

        rlm_synthesis = None
        rlm_telemetry = None

        if self.should_use_rlm_for_query(input.rlm, evidence_context):
            self.emit_progress("rlm_synthesis", "Synthesizing evidence with RLM...")
            logger.info(f"RLM enabled for research query: {input.query[:50]}...")
            try:
                rlm_result = self.process_with_rlm(
                    query=input.query,
                    context=evidence_context,
                    rlm_options=input.rlm,
                )
                rlm_synthesis = rlm_result.answer
                rlm_telemetry = self.get_rlm_telemetry(
                    rlm_result, len(evidence_context)
                )
                logger.info(
                    f"RLM synthesis complete: depth={rlm_result.depth_reached}, "
                    f"calls={rlm_result.total_calls}, latency={rlm_result.latency_ms:.0f}ms"
                )
            except Exception as e:
                logger.error(f"RLM processing failed: {e}")
                rlm_telemetry = {
                    "rlm_enabled": False,
                    "rlm_attempted": True,
                    "rlm_error": str(e),
                }

        return DeepResearchOutput(
            summary=summary,
            sub_questions=sub_questions,
            evidence=all_evidence,
            citations=all_citations,
            iterations_used=iteration,
            gaps_remaining=gaps,
            confidence=confidence,
            rlm_synthesis=rlm_synthesis,
            rlm_telemetry=rlm_telemetry,
        )

    async def _decompose(self, query: str) -> List[str]:
        """Decompose query into sub-questions using DSPy."""
        result = await self.call_dspy(
            self._decomposer,
            output_field="sub_questions",
            query=query,
        )
        sub_qs = result.sub_questions
        if isinstance(sub_qs, str):
            sub_qs = [q.strip() for q in sub_qs.split("\n") if q.strip()]
        return sub_qs[:5]

    async def _search_parallel(
        self, questions: List[str], tenant_id: str
    ) -> List[Dict[str, Any]]:
        """Search for evidence for each sub-question."""
        if not self._search_fn:
            raise ValueError(
                "DeepResearchAgent._search_parallel: no search_fn provided. "
                "Pass a search_fn callable to DeepResearchAgent.__init__ "
                "before calling process()."
            )

        import asyncio

        async def search_one(q: str) -> Dict[str, Any]:
            results = await self._search_fn(query=q, tenant_id=tenant_id)
            return {"question": q, "results": results, "source": "search"}

        tasks = [search_one(q) for q in questions]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    async def _evaluate_evidence(
        self,
        query: str,
        sub_questions: List[str],
        evidence: List[Dict[str, Any]],
    ) -> tuple[bool, List[str], float]:
        """Evaluate if evidence is sufficient."""
        evidence_summary = "\n".join(
            f"Q: {e['question']} → {len(e.get('results', []))} results"
            for e in evidence
        )
        result = await self.call_dspy(
            self._evaluator,
            output_field="gaps",
            query=query,
            sub_questions="\n".join(sub_questions),
            evidence_summary=evidence_summary,
        )

        sufficient = bool(result.has_sufficient_evidence)
        gaps = result.gaps
        if isinstance(gaps, str):
            gaps = [g.strip() for g in gaps.split("\n") if g.strip()]
        confidence = float(result.confidence) if result.confidence else 0.5

        return sufficient, gaps, confidence

    async def _synthesize(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """Synthesize evidence into a research report."""
        evidence_text = "\n\n".join(
            f"## {e['question']}\n{e.get('results', 'No results')}" for e in evidence
        )
        result = await self.call_dspy(
            self._synthesizer,
            output_field="summary",
            query=query,
            evidence=evidence_text,
        )
        return str(result.summary)

    @staticmethod
    def _extract_citations(evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract citations from evidence results."""
        citations = []
        for e in evidence:
            results = e.get("results", [])
            if isinstance(results, list):
                for r in results[:3]:
                    if isinstance(r, dict):
                        citations.append(
                            {
                                "source": r.get(
                                    "document_id", r.get("video_id", "unknown")
                                ),
                                "text": r.get("description", r.get("transcript", "")),
                                "question": e.get("question", ""),
                            }
                        )
        return citations
