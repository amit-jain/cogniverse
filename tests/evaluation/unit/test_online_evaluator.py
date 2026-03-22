"""
Unit tests for OnlineEvaluator — real-time routing span scoring.

Tests:
1. Evaluate span produces correct routing_outcome and confidence_calibration scores
2. Sampling rate controls evaluation frequency
3. Disabled evaluator skips all spans
4. Score persistence calls annotation API
5. Config propagation from OnlineEvaluationConfig
6. Unknown evaluator names are handled gracefully
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_agents.routing.config import (
    AutomationRulesConfig,
    OnlineEvaluationConfig,
)
from cogniverse_evaluation.online_evaluator import OnlineEvalResult, OnlineEvaluator


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.annotations = MagicMock()
    provider.annotations.add_annotation = AsyncMock()
    return provider


@pytest.fixture
def default_evaluator(mock_provider):
    return OnlineEvaluator(
        provider=mock_provider,
        project_name="test-project",
    )


@pytest.fixture
def configured_evaluator(mock_provider):
    config = OnlineEvaluationConfig(
        enabled=True,
        sampling_rate=1.0,
        evaluators=["routing_outcome", "confidence_calibration"],
        persist_scores=True,
        score_annotation_name="test_eval",
    )
    return OnlineEvaluator(
        provider=mock_provider,
        project_name="test-project",
        config=config,
    )


class TestOnlineEvaluatorBasicScoring:
    """Verify evaluators produce correct scores."""

    @pytest.mark.asyncio
    async def test_routing_outcome_success(self, default_evaluator):
        span_data = {
            "context.span_id": "span-1",
            "parent_id": "parent-1",
            "status_code": "OK",
            "attributes": {"routing.chosen_agent": "search_agent"},
            "events": [],
        }
        results = await default_evaluator.evaluate_span(span_data)

        outcome_results = [r for r in results if r.evaluator_name == "routing_outcome"]
        assert len(outcome_results) == 1
        assert outcome_results[0].score == 1.0
        assert outcome_results[0].label == "success"

    @pytest.mark.asyncio
    async def test_routing_outcome_failure(self, default_evaluator):
        span_data = {
            "context.span_id": "span-2",
            "parent_id": "parent-1",
            "status_code": "ERROR",
            "attributes": {"routing.chosen_agent": "search_agent"},
            "events": [],
        }
        results = await default_evaluator.evaluate_span(span_data)

        outcome_results = [r for r in results if r.evaluator_name == "routing_outcome"]
        assert len(outcome_results) == 1
        assert outcome_results[0].score == 0.0
        assert outcome_results[0].label == "failure"

    @pytest.mark.asyncio
    async def test_routing_outcome_ambiguous(self, default_evaluator):
        span_data = {
            "context.span_id": "span-3",
            "status_code": "OK",
            "attributes": {},
            "events": [],
        }
        results = await default_evaluator.evaluate_span(span_data)

        outcome_results = [r for r in results if r.evaluator_name == "routing_outcome"]
        assert len(outcome_results) == 1
        assert outcome_results[0].score == 0.5
        assert outcome_results[0].label == "ambiguous"

    @pytest.mark.asyncio
    async def test_confidence_calibration_well_calibrated(self, default_evaluator):
        span_data = {
            "context.span_id": "span-4",
            "status_code": "OK",
            "attributes.routing": {"confidence": 0.95},
        }
        results = await default_evaluator.evaluate_span(span_data)

        cal_results = [
            r for r in results if r.evaluator_name == "confidence_calibration"
        ]
        assert len(cal_results) == 1
        assert cal_results[0].score == 0.95
        assert cal_results[0].label == "well_calibrated"

    @pytest.mark.asyncio
    async def test_confidence_calibration_poorly_calibrated(self, default_evaluator):
        """High confidence but actual failure → poorly calibrated."""
        span_data = {
            "context.span_id": "span-5",
            "status_code": "ERROR",
            "attributes.routing": {"confidence": 0.9},
        }
        results = await default_evaluator.evaluate_span(span_data)

        cal_results = [
            r for r in results if r.evaluator_name == "confidence_calibration"
        ]
        assert len(cal_results) == 1
        # calibration = 1 - 0.9 = 0.1 (high confidence predicted success but got failure)
        assert cal_results[0].score == pytest.approx(0.1)
        assert cal_results[0].label == "poorly_calibrated"


class TestOnlineEvaluatorSampling:
    """Verify sampling rate controls evaluation frequency."""

    @pytest.mark.asyncio
    async def test_sampling_rate_zero_skips_all(self, mock_provider):
        config = OnlineEvaluationConfig(sampling_rate=0.0)
        evaluator = OnlineEvaluator(
            provider=mock_provider, project_name="test", config=config
        )

        span_data = {"context.span_id": "span-1", "status_code": "OK"}
        results = await evaluator.evaluate_span(span_data)
        assert results == []
        assert evaluator._total_skipped == 1

    @pytest.mark.asyncio
    async def test_sampling_rate_one_evaluates_all(self, mock_provider):
        config = OnlineEvaluationConfig(
            sampling_rate=1.0, evaluators=["routing_outcome"]
        )
        evaluator = OnlineEvaluator(
            provider=mock_provider, project_name="test", config=config
        )

        span_data = {
            "context.span_id": "span-1",
            "parent_id": "p-1",
            "status_code": "OK",
            "attributes": {"routing.chosen_agent": "search_agent"},
            "events": [],
        }
        results = await evaluator.evaluate_span(span_data)
        assert len(results) == 1
        assert evaluator._total_evaluated == 1

    @pytest.mark.asyncio
    async def test_disabled_evaluator_skips(self, mock_provider):
        config = OnlineEvaluationConfig(enabled=False)
        evaluator = OnlineEvaluator(
            provider=mock_provider, project_name="test", config=config
        )

        span_data = {"context.span_id": "span-1"}
        results = await evaluator.evaluate_span(span_data)
        assert results == []


class TestOnlineEvaluatorPersistence:
    """Verify scores are persisted as annotations."""

    @pytest.mark.asyncio
    async def test_persist_scores_calls_annotation_api(self, configured_evaluator):
        span_data = {
            "context.span_id": "span-persist",
            "parent_id": "parent-1",
            "status_code": "OK",
            "attributes": {"routing.chosen_agent": "search_agent"},
            "attributes.routing": {"confidence": 0.8},
            "events": [],
        }
        results = await configured_evaluator.evaluate_span(span_data)

        provider = configured_evaluator.provider
        assert provider.annotations.add_annotation.call_count == len(results)

        # Verify the annotation call parameters
        calls = provider.annotations.add_annotation.call_args_list
        annotation_names = {call.kwargs.get("name") for call in calls}
        assert "test_eval.routing_outcome" in annotation_names
        assert "test_eval.confidence_calibration" in annotation_names

    @pytest.mark.asyncio
    async def test_persist_disabled(self, mock_provider):
        config = OnlineEvaluationConfig(
            persist_scores=False, evaluators=["routing_outcome"]
        )
        evaluator = OnlineEvaluator(
            provider=mock_provider, project_name="test", config=config
        )

        span_data = {
            "context.span_id": "span-np",
            "parent_id": "p-1",
            "status_code": "OK",
            "attributes": {"routing.chosen_agent": "search_agent"},
            "events": [],
        }
        results = await evaluator.evaluate_span(span_data)
        assert len(results) == 1
        mock_provider.annotations.add_annotation.assert_not_called()


class TestOnlineEvaluatorConfig:
    """Verify config propagation from AutomationRulesConfig."""

    def test_config_propagates_from_automation_rules(self, mock_provider):
        rules = AutomationRulesConfig(
            online_evaluation={
                "enabled": True,
                "sampling_rate": 0.5,
                "evaluators": ["routing_outcome"],
                "persist_scores": False,
                "score_annotation_name": "custom_eval",
            }
        )

        evaluator = OnlineEvaluator(
            provider=mock_provider,
            project_name="test",
            config=rules.online_evaluation,
        )
        assert evaluator.sampling_rate == 0.5
        assert evaluator.evaluator_names == ["routing_outcome"]
        assert evaluator.persist_scores is False
        assert evaluator.annotation_name == "custom_eval"

    def test_default_config_values(self):
        config = OnlineEvaluationConfig()
        assert config.enabled is True
        assert config.sampling_rate == 1.0
        assert "routing_outcome" in config.evaluators
        assert "confidence_calibration" in config.evaluators
        assert config.persist_scores is True
        assert config.score_annotation_name == "online_eval"


class TestOnlineEvaluatorEdgeCases:
    """Verify graceful handling of edge cases."""

    @pytest.mark.asyncio
    async def test_unknown_evaluator_name(self, mock_provider):
        config = OnlineEvaluationConfig(evaluators=["nonexistent_evaluator"])
        evaluator = OnlineEvaluator(
            provider=mock_provider, project_name="test", config=config
        )

        span_data = {"context.span_id": "span-1"}
        results = await evaluator.evaluate_span(span_data)
        assert results == []

    @pytest.mark.asyncio
    async def test_missing_routing_attributes(self, default_evaluator):
        """Confidence calibration works with missing attributes."""
        span_data = {
            "context.span_id": "span-missing",
            "status_code": "OK",
        }
        results = await default_evaluator.evaluate_span(span_data)

        cal_results = [
            r for r in results if r.evaluator_name == "confidence_calibration"
        ]
        assert len(cal_results) == 1
        # Default confidence 0.5, success=True → score=0.5
        assert cal_results[0].score == 0.5

    def test_statistics(self, default_evaluator):
        stats = default_evaluator.get_statistics()
        assert stats["total_evaluated"] == 0
        assert stats["total_skipped"] == 0
        assert stats["sampling_rate"] == 1.0
        assert "routing_outcome" in stats["evaluators"]

    def test_eval_result_to_dict(self):
        result = OnlineEvalResult(
            span_id="s1",
            evaluator_name="routing_outcome",
            score=1.0,
            label="success",
            explanation="ok",
            timestamp=datetime(2026, 1, 1),
        )
        d = result.to_dict()
        assert d["span_id"] == "s1"
        assert d["score"] == 1.0
        assert d["timestamp"] == "2026-01-01T00:00:00"
