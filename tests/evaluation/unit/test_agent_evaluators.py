"""Per-agent-type span evaluator registry.

One registry entry per agent type: a judge-prompt builder (LLM-scored quality,
used by QualityMonitor's live eval) plus structural evaluators (no LLM, used by
the online span-eval cycle). Pins:

- every agent type has a registry entry (search/summary/report/gateway/routing/
  query_enhancement/entity_extraction/profile_selection);
- the four pre-existing judge prompts are byte-identical after moving into the
  registry;
- routing confidence calibration reads confidence from the canonical
  ``output.value`` JSON the gateway actually writes (it used to read a
  ``attributes.routing`` attribute that the emitter never sets, so every span
  scored the 0.5 default), with the legacy attribute still honored;
- the new structural evaluators score synthetic spans to exact values.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from cogniverse_evaluation.evaluators.agent_evaluators import (
    AGENT_EVALUATORS,
    build_entity_extraction_judge_prompt,
    build_profile_selection_judge_prompt,
    build_query_enhancement_judge_prompt,
    build_report_judge_prompt,
    build_routing_judge_prompt,
    build_search_judge_prompt,
    build_summary_judge_prompt,
    get_agent_evaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


ALL_AGENT_TYPES = {
    "search",
    "summary",
    "report",
    "gateway",
    "routing",
    "query_enhancement",
    "entity_extraction",
    "profile_selection",
}


def _routing_span(
    *,
    confidence=None,
    legacy_confidence=None,
    status="OK",
    parent="parent-1",
    chosen="search_agent",
):
    """Raw Phoenix-row shape: dotted attribute columns, output.value JSON."""
    output = {}
    if chosen is not None:
        output["chosen_agent"] = chosen
    if confidence is not None:
        output["confidence"] = confidence
    span = {
        "context.span_id": "span-1",
        "status_code": status,
        "attributes.output.value": json.dumps(output),
        "attributes.input.value": "find cat videos",
    }
    if parent is not None:
        span["parent_id"] = parent
    if legacy_confidence is not None:
        span["attributes.routing"] = {"confidence": legacy_confidence}
    return span


class TestRegistryCoverage:
    def test_registry_covers_every_agent_type(self):
        assert set(AGENT_EVALUATORS) == ALL_AGENT_TYPES

    def test_get_agent_evaluator_unknown_returns_none(self):
        assert get_agent_evaluator("nonexistent") is None

    def test_every_entry_has_judge_prompt_and_span_name(self):
        for agent_type, entry in AGENT_EVALUATORS.items():
            assert entry.agent_type == agent_type
            assert entry.span_name, f"{agent_type} has no span name"
            assert callable(entry.judge_prompt), f"{agent_type} has no judge"

    def test_quality_monitor_agent_types_align_with_registry(self):
        from cogniverse_evaluation.quality_monitor import (
            SPAN_NAME_BY_AGENT,
            AgentType,
        )

        assert {a.value for a in AgentType} == ALL_AGENT_TYPES
        for agent_type in AgentType:
            entry = get_agent_evaluator(agent_type.value)
            assert SPAN_NAME_BY_AGENT[agent_type] == entry.span_name

    def test_new_agent_types_map_to_domain_span_names(self):
        from cogniverse_evaluation.quality_monitor import (
            SPAN_NAME_BY_AGENT,
            AgentType,
        )

        assert SPAN_NAME_BY_AGENT[AgentType.ROUTING] == "cogniverse.routing"
        assert (
            SPAN_NAME_BY_AGENT[AgentType.QUERY_ENHANCEMENT]
            == "cogniverse.query_enhancement"
        )
        assert (
            SPAN_NAME_BY_AGENT[AgentType.ENTITY_EXTRACTION]
            == "cogniverse.entity_extraction"
        )
        assert (
            SPAN_NAME_BY_AGENT[AgentType.PROFILE_SELECTION]
            == "cogniverse.profile_selection"
        )


class TestLegacyJudgePromptsUnchanged:
    """The four pre-registry prompts are the contract — pin them exactly."""

    def test_search_prompt_exact(self):
        results = [
            {"video_id": "v1", "score": 0.9},
            {"source_id": "s2", "score": 0.7},
        ]
        expected = (
            "Query: cats\n\n"
            "Search Results:\n"
            "  1. v1 (score: 0.9)\n"
            "  2. s2 (score: 0.7)\n\n"
            "Rate the relevance of these search results to the query. "
            "Consider: Are the results topically relevant? Is the ranking order "
            "sensible? Would a user find these results helpful?\n"
            "Score: X/10"
        )
        assert build_search_judge_prompt("cats", results) == expected

    def test_summary_prompt_exact(self):
        expected = (
            "Original Query: cats\n\n"
            "Generated Summary:\nA summary.\n\n"
            "Rate the quality of this summary. Consider: Is it accurate? "
            "Is it concise? Does it address the query? Is it coherent?\n"
            "Score: X/10"
        )
        assert build_summary_judge_prompt("cats", "A summary.") == expected

    def test_report_prompt_truncates_at_2000(self):
        long_report = "r" * 3000
        prompt = build_report_judge_prompt("cats", long_report)
        assert "r" * 2000 in prompt
        assert "r" * 2001 not in prompt

    def test_routing_prompt_exact(self):
        routing = {"chosen_agent": "search_agent", "confidence": 0.9}
        expected = (
            "Query: cats\n\n"
            f"Routing Decision: {json.dumps(routing, default=str)}\n\n"
            "Rate the routing decision. Consider: Was the right agent selected? "
            "Is the confidence calibrated? Does the workflow make sense for "
            "this query type?\n"
            "Score: X/10"
        )
        assert build_routing_judge_prompt("cats", routing) == expected


class TestRoutingStructuralEvaluators:
    def test_confidence_read_from_canonical_output_value(self):
        """The gateway writes confidence inside output.value JSON; the
        calibration evaluator must read it there — not default to 0.5."""
        entry = get_agent_evaluator("routing")
        span = _routing_span(confidence=0.9)

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        cal = results["confidence_calibration"]

        assert cal.score == pytest.approx(0.9)
        assert cal.label == "well_calibrated"

    def test_confidence_falls_back_to_legacy_attribute(self):
        entry = get_agent_evaluator("routing")
        span = _routing_span(confidence=None, legacy_confidence="high")

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}

        assert results["confidence_calibration"].score == pytest.approx(0.9)

    def test_confidence_missing_everywhere_defaults(self):
        entry = get_agent_evaluator("routing")
        span = _routing_span(confidence=None)

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}

        assert results["confidence_calibration"].score == pytest.approx(0.5)

    def test_failed_span_inverts_calibration(self):
        entry = get_agent_evaluator("routing")
        span = _routing_span(confidence=0.9, status="ERROR")

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        cal = results["confidence_calibration"]

        assert cal.score == pytest.approx(0.1)
        assert cal.label == "poorly_calibrated"

    def test_routing_outcome_success(self):
        entry = get_agent_evaluator("routing")
        span = _routing_span(confidence=0.9)

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        outcome = results["routing_outcome"]

        assert outcome.score == pytest.approx(1.0)
        assert outcome.label == "success"

    def test_routing_outcome_no_parent_is_ambiguous(self):
        entry = get_agent_evaluator("routing")
        span = _routing_span(confidence=0.9, parent=None)

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        outcome = results["routing_outcome"]

        assert outcome.score == pytest.approx(0.5)
        assert outcome.label == "ambiguous"

    def test_online_evaluator_scores_canonical_confidence(self):
        """End-to-end through OnlineEvaluator — the path the span-eval cycle
        uses. Before the fix this returned the 0.5 default for every real
        gateway span."""
        from cogniverse_evaluation.online_evaluator import OnlineEvaluator

        ev = OnlineEvaluator(provider=Mock(), project_name="p")
        out = ev._eval_confidence_calibration(_routing_span(confidence=0.9), "s1")

        assert out.score == pytest.approx(0.9)
        assert out.label == "well_calibrated"


class TestQueryEnhancementEvaluator:
    def _span(self, original, enhanced, confidence=0.8, variants=2):
        return {
            "context.span_id": "qe-1",
            "status_code": "OK",
            "attributes.input.value": original,
            "attributes.output.value": json.dumps(
                {
                    "enhanced_query": enhanced,
                    "variant_count": variants,
                    "confidence": confidence,
                }
            ),
        }

    def test_identity_enhancement_scores_zero(self):
        entry = get_agent_evaluator("query_enhancement")
        span = self._span("find cats", "find cats")

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        effect = results["enhancement_effect"]

        assert effect.score == pytest.approx(0.0)
        assert effect.label == "no_enhancement"

    def test_changed_query_scores_module_confidence(self):
        entry = get_agent_evaluator("query_enhancement")
        span = self._span("find cats", "find domestic cat videos", confidence=0.8)

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        effect = results["enhancement_effect"]

        assert effect.score == pytest.approx(0.8)
        assert effect.label == "enhanced"

    def test_judge_prompt_carries_both_queries(self):
        prompt = build_query_enhancement_judge_prompt(
            "find cats", {"enhanced_query": "find domestic cats", "variant_count": 3}
        )
        assert "Original Query: find cats" in prompt
        assert "Enhanced Query: find domestic cats" in prompt
        assert "Variants Produced: 3" in prompt
        assert "Score: X/10" in prompt


class TestEntityExtractionEvaluator:
    def _span(self, query, entities, relationships=()):
        return {
            "context.span_id": "ee-1",
            "status_code": "OK",
            "attributes.input.value": query,
            "attributes.output.value": json.dumps(
                {
                    "entities": list(entities),
                    "relationships": list(relationships),
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                }
            ),
        }

    def test_zero_entities_on_substantive_text_scores_zero(self):
        entry = get_agent_evaluator("entity_extraction")
        span = self._span("show me the eiffel tower at night", [])

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        yield_result = results["extraction_yield"]

        assert yield_result.score == pytest.approx(0.0)
        assert yield_result.label == "no_entities"

    def test_entities_present_scores_one(self):
        entry = get_agent_evaluator("entity_extraction")
        span = self._span(
            "show me the eiffel tower at night",
            [{"text": "eiffel tower", "type": "landmark"}],
        )

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}

        assert results["extraction_yield"].score == pytest.approx(1.0)
        assert results["extraction_yield"].label == "extracted"

    def test_trivial_text_with_no_entities_is_neutral(self):
        entry = get_agent_evaluator("entity_extraction")
        span = self._span("hi", [])

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}

        assert results["extraction_yield"].score == pytest.approx(0.5)
        assert results["extraction_yield"].label == "trivial_text"

    def test_judge_prompt_carries_text_and_entities(self):
        prompt = build_entity_extraction_judge_prompt(
            "the eiffel tower", {"entities": [{"text": "eiffel tower"}]}
        )
        assert "Source Text: the eiffel tower" in prompt
        assert "eiffel tower" in prompt
        assert "Score: X/10" in prompt


class TestProfileSelectionEvaluator:
    def _span(self, profile, confidence, status="OK"):
        output = {"modality": "video", "intent": "search", "confidence": confidence}
        if profile is not None:
            output["selected_profile"] = profile
        return {
            "context.span_id": "ps-1",
            "status_code": status,
            "attributes.input.value": "find cat videos",
            "attributes.output.value": json.dumps(output),
        }

    def test_confident_selection_on_ok_span_scores_confidence(self):
        entry = get_agent_evaluator("profile_selection")
        span = self._span("video_colpali_smol500_mv_frame", 0.9)

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        cal = results["profile_confidence_calibration"]

        assert cal.score == pytest.approx(0.9)
        assert cal.label == "well_calibrated"

    def test_missing_profile_scores_zero(self):
        entry = get_agent_evaluator("profile_selection")
        span = self._span(None, 0.9)

        results = {r.evaluator_name: r for r in entry.structural_evaluations(span)}
        cal = results["profile_confidence_calibration"]

        assert cal.score == pytest.approx(0.0)
        assert cal.label == "no_profile"

    def test_judge_prompt_carries_selection(self):
        prompt = build_profile_selection_judge_prompt(
            "find cat videos",
            {"selected_profile": "video_colpali", "modality": "video", "intent": "s"},
        )
        assert "Selected Profile: video_colpali" in prompt
        assert "Score: X/10" in prompt


class TestQualityMonitorDispatchesThroughRegistry:
    """QM's live judging routes every agent type through the registry —
    extraction parity for the legacy four, coverage for the new four."""

    def _monitor(self, judge):
        from cogniverse_evaluation.quality_monitor import QualityMonitor

        monitor = QualityMonitor(
            tenant_id="t",
            runtime_url="http://r",
            phoenix_http_endpoint="http://p",
            llm_base_url="http://l",
            llm_model="m",
            golden_dataset_path="/nonexistent.json",
        )
        monitor._llm_judge = judge

        async def _no_baseline(agent_type):
            return None

        monitor._get_agent_baseline = _no_baseline
        return monitor

    @pytest.mark.asyncio
    async def test_query_enhancement_span_judged_with_qe_prompt(self):
        import pandas as pd

        from cogniverse_evaluation.quality_monitor import AgentType

        captured = {}

        class _Judge:
            async def _call_llm(self, prompt, system_prompt):
                captured["prompt"] = prompt
                return "Score: 8/10"

            def _extract_score_from_response(self, response):
                return 0.8, "ok"

        monitor = self._monitor(_Judge())
        spans_df = pd.DataFrame(
            [
                {
                    "span_id": "qe-1",
                    "attributes": {"query": "find cats"},
                    "outputs": {
                        "results": [],
                        "value": {
                            "enhanced_query": "find domestic cats",
                            "variant_count": 2,
                            "confidence": 0.8,
                        },
                    },
                }
            ]
        )

        result = await monitor._evaluate_agent_spans(
            AgentType.QUERY_ENHANCEMENT, spans_df
        )

        assert result.sample_count == 1
        assert result.score == pytest.approx(0.8)
        assert "Enhanced Query: find domestic cats" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_search_span_judged_with_exact_legacy_prompt(self):
        import pandas as pd

        from cogniverse_evaluation.quality_monitor import AgentType

        captured = {}

        class _Judge:
            async def _call_llm(self, prompt, system_prompt):
                captured["prompt"] = prompt
                return "Score: 9/10"

            def _extract_score_from_response(self, response):
                return 0.9, "ok"

        monitor = self._monitor(_Judge())
        spans_df = pd.DataFrame(
            [
                {
                    "span_id": "s-1",
                    "attributes": {"query": "cats"},
                    "outputs": {"results": [{"video_id": "v1", "score": 0.9}]},
                }
            ]
        )

        await monitor._evaluate_agent_spans(AgentType.SEARCH, spans_df)

        assert captured["prompt"] == build_search_judge_prompt(
            "cats", [{"video_id": "v1", "score": 0.9}]
        )
