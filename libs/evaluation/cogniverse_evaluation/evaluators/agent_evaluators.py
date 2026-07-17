"""Per-agent-type span evaluators behind one registry.

Each agent type registers:

- ``judge_prompt`` — builds the LLM-judge prompt from an extracted
  ``(query, payload)`` pair. QualityMonitor's live eval scores every agent
  type through this.
- ``extract_judge_payload`` — pulls the judge payload out of a normalized
  SpanEvaluator row (``attributes``/``outputs`` dicts), per the agent's own
  output shape (search-result list, summary/report string, domain dict).
- ``structural`` — named no-LLM evaluators over a raw Phoenix span row
  (dotted attribute columns). The online span-eval cycle persists one
  ``online_eval.<name>`` annotation per structural result.

Adding an agent type to the optimization loop = one entry here plus an
``AgentType`` member in ``quality_monitor``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cogniverse_evaluation.evaluators.routing_evaluator import (
    RoutingOutcome,
    classify_routing_outcome,
)
from cogniverse_foundation.confidence import parse_confidence
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_ENTITY_EXTRACTION,
    SPAN_NAME_PROFILE_SELECTION,
    SPAN_NAME_QUERY_ENHANCEMENT,
    SPAN_NAME_ROUTING,
)
from cogniverse_foundation.telemetry.span_contract import read_span_io

logger = logging.getLogger(__name__)


@dataclass
class StructuralEvalResult:
    """One no-LLM evaluation of a span."""

    evaluator_name: str
    score: float  # 0-1
    label: str
    explanation: str


# ── judge prompts ────────────────────────────────────────────────────────
# The first four moved verbatim from QualityMonitor — their exact text is
# pinned by tests; do not reword without re-baselining stored judge scores.


def build_search_judge_prompt(query: str, results: List[Dict[str, Any]]) -> str:
    top_results = results[:5]
    results_text = "\n".join(
        f"  {i + 1}. {r.get('video_id', r.get('source_id', 'unknown'))} "
        f"(score: {r.get('score', 'N/A')})"
        for i, r in enumerate(top_results)
    )
    return (
        f"Query: {query}\n\n"
        f"Search Results:\n{results_text}\n\n"
        f"Rate the relevance of these search results to the query. "
        f"Consider: Are the results topically relevant? Is the ranking order "
        f"sensible? Would a user find these results helpful?\n"
        f"Score: X/10"
    )


def build_summary_judge_prompt(query: str, summary: str) -> str:
    return (
        f"Original Query: {query}\n\n"
        f"Generated Summary:\n{summary}\n\n"
        f"Rate the quality of this summary. Consider: Is it accurate? "
        f"Is it concise? Does it address the query? Is it coherent?\n"
        f"Score: X/10"
    )


def build_report_judge_prompt(query: str, report: str) -> str:
    return (
        f"Original Query: {query}\n\n"
        f"Generated Report:\n{report[:2000]}\n\n"
        f"Rate the quality of this report. Consider: Is it comprehensive? "
        f"Are the findings well-supported? Are recommendations actionable? "
        f"Is the structure logical?\n"
        f"Score: X/10"
    )


def build_routing_judge_prompt(query: str, routing: Dict[str, Any]) -> str:
    return (
        f"Query: {query}\n\n"
        f"Routing Decision: {json.dumps(routing, default=str)}\n\n"
        f"Rate the routing decision. Consider: Was the right agent selected? "
        f"Is the confidence calibrated? Does the workflow make sense for "
        f"this query type?\n"
        f"Score: X/10"
    )


def build_query_enhancement_judge_prompt(
    query: str, enhancement: Dict[str, Any]
) -> str:
    return (
        f"Original Query: {query}\n\n"
        f"Enhanced Query: {enhancement.get('enhanced_query', '')}\n"
        f"Variants Produced: {enhancement.get('variant_count', 0)}\n\n"
        f"Rate the query enhancement. Consider: Does it preserve the original "
        f"intent? Does it add retrieval-useful specificity (entities, synonyms, "
        f"context)? Would it retrieve better results than the original query?\n"
        f"Score: X/10"
    )


def build_entity_extraction_judge_prompt(query: str, extraction: Dict[str, Any]) -> str:
    entities = extraction.get("entities", [])
    relationships = extraction.get("relationships", [])
    return (
        f"Source Text: {query}\n\n"
        f"Extracted Entities: {json.dumps(entities, default=str)}\n"
        f"Extracted Relationships: {json.dumps(relationships, default=str)}\n\n"
        f"Rate the entity extraction. Consider: Are all salient entities "
        f"captured (completeness)? Are the extracted entities actually present "
        f"in the text (precision)? Are the entity types sensible?\n"
        f"Score: X/10"
    )


def build_profile_selection_judge_prompt(query: str, selection: Dict[str, Any]) -> str:
    return (
        f"Query: {query}\n\n"
        f"Selected Profile: {selection.get('selected_profile', '')}\n"
        f"Detected Modality: {selection.get('modality', '')}\n"
        f"Detected Intent: {selection.get('intent', '')}\n\n"
        f"Rate the profile selection. Consider: Does the profile match the "
        f"query's modality and intent? Is it a sensible retrieval strategy "
        f"for this query?\n"
        f"Score: X/10"
    )


# ── judge payload extraction (normalized SpanEvaluator rows) ─────────────


def _payload_search(attributes: dict, outputs: dict) -> Any:
    return outputs.get("results", [])


def _payload_str(attributes: dict, outputs: dict) -> Any:
    value = outputs.get("value")
    return value if isinstance(value, str) else str(value or "")


def _payload_dict(attributes: dict, outputs: dict) -> Any:
    value = outputs.get("value")
    return value if isinstance(value, dict) else {}


# ── structural evaluators (raw Phoenix span rows) ────────────────────────


def _span_success(span_data: Dict[str, Any]) -> bool:
    status = span_data.get("status_code", span_data.get("status", ""))
    return status == "OK" if status else True


def _calibration_bands(score: float) -> str:
    if score >= 0.8:
        return "well_calibrated"
    if score >= 0.5:
        return "moderately_calibrated"
    return "poorly_calibrated"


def _confidence_from(io: dict, span_data: Dict[str, Any]) -> float:
    """Confidence from the canonical ``output.value`` JSON, falling back to
    the legacy ``attributes.routing`` dict. The emitter writes confidence
    inside ``output.value``; reading only the legacy attribute made every
    real span score the 0.5 default."""
    output = io["output"] if isinstance(io.get("output"), dict) else {}
    raw = output.get("confidence")
    if raw is None:
        legacy = span_data.get("attributes.routing")
        if isinstance(legacy, dict):
            raw = legacy.get("confidence")
    return parse_confidence(raw, default=0.5)


def _eval_routing_outcome(span_data: Dict[str, Any], io: dict) -> StructuralEvalResult:
    outcome, description = classify_routing_outcome(span_data)
    score_map = {
        RoutingOutcome.SUCCESS: 1.0,
        RoutingOutcome.FAILURE: 0.0,
        RoutingOutcome.AMBIGUOUS: 0.5,
    }
    return StructuralEvalResult(
        evaluator_name="routing_outcome",
        score=score_map.get(outcome, 0.5),
        label=outcome.value,
        explanation=description,
    )


def _eval_confidence_calibration(
    span_data: Dict[str, Any], io: dict
) -> StructuralEvalResult:
    confidence = _confidence_from(io, span_data)
    actual_success = _span_success(span_data)
    score = confidence if actual_success else 1.0 - confidence
    return StructuralEvalResult(
        evaluator_name="confidence_calibration",
        score=score,
        label=_calibration_bands(score),
        explanation=(
            f"confidence={confidence:.2f}, "
            f"actual_success={actual_success}, "
            f"calibration={score:.2f}"
        ),
    )


def _eval_enhancement_effect(
    span_data: Dict[str, Any], io: dict
) -> StructuralEvalResult:
    output = io["output"] if isinstance(io.get("output"), dict) else {}
    original = io.get("input") or ""
    enhanced = output.get("enhanced_query") or ""
    if enhanced.strip().casefold() == str(original).strip().casefold():
        return StructuralEvalResult(
            evaluator_name="enhancement_effect",
            score=0.0,
            label="no_enhancement",
            explanation="enhanced query is identical to the original",
        )
    confidence = parse_confidence(output.get("confidence"), default=0.5)
    return StructuralEvalResult(
        evaluator_name="enhancement_effect",
        score=confidence,
        label="enhanced",
        explanation=(
            f"query changed, module confidence={confidence:.2f}, "
            f"variants={output.get('variant_count', 0)}"
        ),
    )


def _eval_extraction_yield(span_data: Dict[str, Any], io: dict) -> StructuralEvalResult:
    output = io["output"] if isinstance(io.get("output"), dict) else {}
    entities = output.get("entities") or []
    entity_count = output.get("entity_count")
    if entity_count is None:
        entity_count = len(entities)
    text = str(io.get("input") or "")
    if entity_count > 0:
        return StructuralEvalResult(
            evaluator_name="extraction_yield",
            score=1.0,
            label="extracted",
            explanation=(
                f"{entity_count} entities, "
                f"{output.get('relationship_count', 0)} relationships"
            ),
        )
    if len(text.split()) < 3:
        return StructuralEvalResult(
            evaluator_name="extraction_yield",
            score=0.5,
            label="trivial_text",
            explanation="no entities, but the source text is trivially short",
        )
    return StructuralEvalResult(
        evaluator_name="extraction_yield",
        score=0.0,
        label="no_entities",
        explanation="no entities extracted from substantive text",
    )


def _eval_profile_calibration(
    span_data: Dict[str, Any], io: dict
) -> StructuralEvalResult:
    output = io["output"] if isinstance(io.get("output"), dict) else {}
    if not output.get("selected_profile"):
        return StructuralEvalResult(
            evaluator_name="profile_confidence_calibration",
            score=0.0,
            label="no_profile",
            explanation="no profile selected",
        )
    confidence = parse_confidence(output.get("confidence"), default=0.5)
    actual_success = _span_success(span_data)
    score = confidence if actual_success else 1.0 - confidence
    return StructuralEvalResult(
        evaluator_name="profile_confidence_calibration",
        score=score,
        label=_calibration_bands(score),
        explanation=(
            f"profile={output.get('selected_profile')}, "
            f"confidence={confidence:.2f}, actual_success={actual_success}"
        ),
    )


# ── registry ─────────────────────────────────────────────────────────────

StructuralFn = Callable[[Dict[str, Any], dict], StructuralEvalResult]


@dataclass(frozen=True)
class AgentEvaluator:
    """Everything needed to score one agent type's spans."""

    agent_type: str
    span_name: str
    judge_prompt: Callable[[str, Any], str]
    extract_judge_payload: Callable[[dict, dict], Any]
    structural: Dict[str, StructuralFn] = field(default_factory=dict)

    def structural_evaluations(
        self, span_data: Dict[str, Any]
    ) -> List[StructuralEvalResult]:
        """Run every structural evaluator against a raw Phoenix span row."""
        io = read_span_io(span_data)
        results: List[StructuralEvalResult] = []
        for name, fn in self.structural.items():
            try:
                results.append(fn(span_data, io))
            except Exception as e:
                logger.warning(
                    "Structural evaluator %s failed for %s span: %s",
                    name,
                    self.agent_type,
                    e,
                )
        return results

    def run_structural(
        self, name: str, span_data: Dict[str, Any]
    ) -> Optional[StructuralEvalResult]:
        """Run one named structural evaluator; None for an unknown name."""
        fn = self.structural.get(name)
        if fn is None:
            return None
        return fn(span_data, read_span_io(span_data))


_ROUTING_STRUCTURAL: Dict[str, StructuralFn] = {
    "routing_outcome": _eval_routing_outcome,
    "confidence_calibration": _eval_confidence_calibration,
}

AGENT_EVALUATORS: Dict[str, AgentEvaluator] = {
    "search": AgentEvaluator(
        agent_type="search",
        span_name="SearchAgent.process",
        judge_prompt=build_search_judge_prompt,
        extract_judge_payload=_payload_search,
    ),
    "summary": AgentEvaluator(
        agent_type="summary",
        span_name="SummarizerAgent.process",
        judge_prompt=build_summary_judge_prompt,
        extract_judge_payload=_payload_str,
    ),
    "report": AgentEvaluator(
        agent_type="report",
        span_name="DetailedReportAgent.process",
        judge_prompt=build_report_judge_prompt,
        extract_judge_payload=_payload_str,
    ),
    "gateway": AgentEvaluator(
        agent_type="gateway",
        span_name="GatewayAgent.process",
        judge_prompt=build_routing_judge_prompt,
        extract_judge_payload=_payload_dict,
        structural=_ROUTING_STRUCTURAL,
    ),
    "routing": AgentEvaluator(
        agent_type="routing",
        span_name=SPAN_NAME_ROUTING,
        judge_prompt=build_routing_judge_prompt,
        extract_judge_payload=_payload_dict,
        structural=_ROUTING_STRUCTURAL,
    ),
    "query_enhancement": AgentEvaluator(
        agent_type="query_enhancement",
        span_name=SPAN_NAME_QUERY_ENHANCEMENT,
        judge_prompt=build_query_enhancement_judge_prompt,
        extract_judge_payload=_payload_dict,
        structural={"enhancement_effect": _eval_enhancement_effect},
    ),
    "entity_extraction": AgentEvaluator(
        agent_type="entity_extraction",
        span_name=SPAN_NAME_ENTITY_EXTRACTION,
        judge_prompt=build_entity_extraction_judge_prompt,
        extract_judge_payload=_payload_dict,
        structural={"extraction_yield": _eval_extraction_yield},
    ),
    "profile_selection": AgentEvaluator(
        agent_type="profile_selection",
        span_name=SPAN_NAME_PROFILE_SELECTION,
        judge_prompt=build_profile_selection_judge_prompt,
        extract_judge_payload=_payload_dict,
        structural={"profile_confidence_calibration": _eval_profile_calibration},
    ),
}


def get_agent_evaluator(agent_type: str) -> Optional[AgentEvaluator]:
    """Registry lookup by agent-type value; None for an unknown type."""
    return AGENT_EVALUATORS.get(agent_type)
