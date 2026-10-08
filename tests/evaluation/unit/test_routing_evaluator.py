"""
Unit tests for RoutingEvaluator.

Tests the routing evaluation metrics calculation without requiring
actual telemetry infrastructure.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from cogniverse_evaluation.evaluators.routing_evaluator import (
    RoutingEvaluator,
    RoutingMetrics,
    RoutingOutcome,
)

pytestmark = pytest.mark.unit

TEST_PROJECT_NAME = "cogniverse-test-routing-optimization"


def _routing_evaluator(provider):
    return RoutingEvaluator(provider=provider, project_name=TEST_PROJECT_NAME)


@pytest.fixture
def mock_provider():
    """Create a mock telemetry provider for testing"""
    provider = MagicMock()
    provider.traces = MagicMock()
    return provider


class TestRoutingEvaluator:
    """Test RoutingEvaluator initialization and basic functionality"""

    def test_evaluator_requires_project_name(self, mock_provider):
        """Test RoutingEvaluator rejects omission of the project name."""
        with pytest.raises(TypeError) as excinfo:
            RoutingEvaluator(provider=mock_provider)

        assert (
            str(excinfo.value)
            == "RoutingEvaluator.__init__() missing 1 required keyword-only argument: 'project_name'"
        )

    def test_evaluator_initialization(self, mock_provider):
        """Test RoutingEvaluator initializes correctly"""
        evaluator = _routing_evaluator(mock_provider)
        assert evaluator.provider is not None
        assert evaluator.logger is not None

    def test_evaluator_with_custom_provider(self, mock_provider):
        """Test RoutingEvaluator accepts custom provider"""
        custom_provider = MagicMock()
        evaluator = _routing_evaluator(custom_provider)
        assert evaluator.provider is custom_provider


class TestRoutingDecisionEvaluation:
    """Test evaluation of individual routing decisions"""

    def test_evaluate_successful_routing_decision(self, mock_provider):
        """Test evaluation of a successful routing decision"""
        evaluator = _routing_evaluator(mock_provider)

        span_data = {
            "name": "cogniverse.routing",
            "parent_id": "parent-123",
            "status_code": "OK",
            "attributes": {
                "routing.chosen_agent": "video_search",
                "routing.confidence": 0.95,
                "routing.processing_time": 50.0,
            },
            "events": [],
        }

        outcome, metrics = evaluator.evaluate_routing_decision(span_data)

        assert outcome == RoutingOutcome.SUCCESS
        assert metrics["chosen_agent"] == "video_search"
        assert metrics["confidence"] == 0.95
        assert metrics["latency_ms"] == 50.0
        assert metrics["success"] is True
        assert metrics["downstream_status"] == "completed_successfully"

    def test_evaluate_failed_routing_decision(self, mock_provider):
        """Test evaluation of a failed routing decision"""
        evaluator = _routing_evaluator(mock_provider)

        span_data = {
            "name": "cogniverse.routing",
            "parent_id": "parent-123",
            "status_code": "ERROR",
            "attributes": {
                "routing.chosen_agent": "text_search",
                "routing.confidence": 0.45,
                "routing.processing_time": 75.0,
            },
            "events": [],
        }

        outcome, metrics = evaluator.evaluate_routing_decision(span_data)

        assert outcome == RoutingOutcome.FAILURE
        assert metrics["chosen_agent"] == "text_search"
        assert metrics["success"] is False
        assert metrics["downstream_status"] == "routing_error"

    def test_evaluate_ambiguous_routing_decision(self, mock_provider):
        """Test evaluation of ambiguous routing decision (no parent span)"""
        evaluator = _routing_evaluator(mock_provider)

        span_data = {
            "name": "cogniverse.routing",
            "status_code": "OK",
            "attributes": {
                "routing.chosen_agent": "video_search",
                "routing.confidence": 0.60,
                "routing.processing_time": 100.0,
            },
            "events": [],
        }

        outcome, metrics = evaluator.evaluate_routing_decision(span_data)

        assert outcome == RoutingOutcome.AMBIGUOUS
        assert metrics["downstream_status"] == "no_parent_span"

    def test_a_routing_span_left_unset_with_a_chosen_agent_succeeded(
        self, mock_provider
    ):
        """The span writer leaves a successful span UNSET; only ERROR fails."""
        evaluator = _routing_evaluator(mock_provider)
        span_data = {
            "name": "cogniverse.routing",
            "parent_id": "parent-123",
            "status_code": "UNSET",
            "attributes.output.value": '{"chosen_agent": "search_agent", '
            '"confidence": 0.9}',
        }

        outcome, metrics = evaluator.evaluate_routing_decision(span_data)

        assert (outcome, metrics["downstream_status"]) == (
            RoutingOutcome.SUCCESS,
            "completed_successfully",
        )

    def test_a_root_routing_span_read_from_a_frame_is_ambiguous(self, mock_provider):
        """A span frame gives a root span a NaN parent."""
        evaluator = _routing_evaluator(mock_provider)
        span_data = {
            "name": "cogniverse.routing",
            "parent_id": float("nan"),
            "status_code": "UNSET",
            "attributes.output.value": '{"chosen_agent": "search_agent", '
            '"confidence": 0.9}',
        }

        outcome, metrics = evaluator.evaluate_routing_decision(span_data)

        assert (outcome, metrics["downstream_status"]) == (
            RoutingOutcome.AMBIGUOUS,
            "no_parent_span",
        )

    def test_evaluate_invalid_span_name(self, mock_provider):
        """Test that evaluator raises error for non-routing spans"""
        evaluator = _routing_evaluator(mock_provider)

        span_data = {
            "name": "cogniverse.search",  # Wrong span name
            "attributes": {},
        }

        with pytest.raises(ValueError, match="Expected cogniverse.routing span"):
            evaluator.evaluate_routing_decision(span_data)

    def test_evaluate_missing_required_attributes(self, mock_provider):
        """Test that evaluator raises error when required attributes are missing"""
        evaluator = _routing_evaluator(mock_provider)

        span_data = {
            "name": "cogniverse.routing",
            "attributes": {
                # Missing routing.chosen_agent and routing.confidence
            },
        }

        with pytest.raises(ValueError, match="missing required attributes"):
            evaluator.evaluate_routing_decision(span_data)


class TestMetricsCalculation:
    """Test calculation of aggregated routing metrics"""

    def test_calculate_metrics_with_all_successful(self, mock_provider):
        """Test metrics calculation when all routing decisions succeed"""
        evaluator = _routing_evaluator(mock_provider)

        routing_spans = [
            {
                "name": "cogniverse.routing",
                "parent_id": "parent-1",
                "status_code": "OK",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.95,
                    "routing.processing_time": 50.0,
                },
                "events": [],
            },
            {
                "name": "cogniverse.routing",
                "parent_id": "parent-2",
                "status_code": "OK",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.90,
                    "routing.processing_time": 60.0,
                },
                "events": [],
            },
        ]

        metrics = evaluator.calculate_metrics(routing_spans)

        assert isinstance(metrics, RoutingMetrics)
        assert metrics.routing_accuracy == 1.0  # 100% success
        assert metrics.total_decisions == 2
        assert metrics.ambiguous_count == 0
        assert metrics.avg_routing_latency == 55.0  # (50 + 60) / 2
        assert "video_search" in metrics.per_agent_precision

    def test_calculate_metrics_with_mixed_outcomes(self, mock_provider):
        """Test metrics calculation with mixed success/failure"""
        evaluator = _routing_evaluator(mock_provider)

        routing_spans = [
            {
                "name": "cogniverse.routing",
                "parent_id": "parent-1",
                "status_code": "OK",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.95,
                    "routing.processing_time": 50.0,
                },
                "events": [],
            },
            {
                "name": "cogniverse.routing",
                "parent_id": "parent-2",
                "status_code": "ERROR",
                "attributes": {
                    "routing.chosen_agent": "text_search",
                    "routing.confidence": 0.40,
                    "routing.processing_time": 75.0,
                },
                "events": [],
            },
            {
                "name": "cogniverse.routing",
                "parent_id": "parent-3",
                "status_code": "OK",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.85,
                    "routing.processing_time": 55.0,
                },
                "events": [],
            },
        ]

        metrics = evaluator.calculate_metrics(routing_spans)

        assert metrics.routing_accuracy == pytest.approx(2 / 3)  # 2 out of 3 succeeded
        assert metrics.total_decisions == 3
        assert metrics.ambiguous_count == 0
        assert metrics.avg_routing_latency == pytest.approx(60.0)  # (50 + 75 + 55) / 3

    def test_calculate_metrics_empty_list_raises_error(self, mock_provider):
        """Test that empty span list raises ValueError"""
        evaluator = _routing_evaluator(mock_provider)

        with pytest.raises(ValueError, match="Cannot calculate metrics from empty"):
            evaluator.calculate_metrics([])

    def test_calculate_metrics_all_invalid_spans_raises_error(self, mock_provider):
        """Test that list with only invalid spans raises ValueError"""
        evaluator = _routing_evaluator(mock_provider)

        invalid_spans = [
            {
                "name": "wrong_span_name",
                "attributes": {},
            }
        ]

        with pytest.raises(ValueError, match="No valid routing spans found"):
            evaluator.calculate_metrics(invalid_spans)


class TestConfidenceCalibration:
    """Test confidence calibration metric calculation"""

    def test_perfect_calibration(self, mock_provider):
        """Test confidence calibration with perfect correlation"""
        evaluator = _routing_evaluator(mock_provider)

        # High confidence => success, low confidence => failure
        routing_spans = [
            {
                "name": "cogniverse.routing",
                "parent_id": "p1",
                "status_code": "OK",  # Success
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.95,  # High confidence
                    "routing.processing_time": 50.0,
                },
                "events": [],
            },
            {
                "name": "cogniverse.routing",
                "parent_id": "p2",
                "status_code": "ERROR",  # Failure
                "attributes": {
                    "routing.chosen_agent": "text_search",
                    "routing.confidence": 0.30,  # Low confidence
                    "routing.processing_time": 60.0,
                },
                "events": [],
            },
        ]

        metrics = evaluator.calculate_metrics(routing_spans)

        # Perfect positive correlation should be close to 1.0
        assert metrics.confidence_calibration > 0.5


class TestPerAgentMetrics:
    """Test per-agent precision/recall/F1 calculation"""

    def test_per_agent_precision(self, mock_provider):
        """Test precision calculation for different agents"""
        evaluator = _routing_evaluator(mock_provider)

        routing_spans = [
            # Video search: 2 success, 1 failure
            {
                "name": "cogniverse.routing",
                "parent_id": "p1",
                "status_code": "OK",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.95,
                    "routing.processing_time": 50.0,
                },
                "events": [],
            },
            {
                "name": "cogniverse.routing",
                "parent_id": "p2",
                "status_code": "OK",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.90,
                    "routing.processing_time": 55.0,
                },
                "events": [],
            },
            {
                "name": "cogniverse.routing",
                "parent_id": "p3",
                "status_code": "ERROR",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.60,
                    "routing.processing_time": 70.0,
                },
                "events": [],
            },
            # Text search: 1 success
            {
                "name": "cogniverse.routing",
                "parent_id": "p4",
                "status_code": "OK",
                "attributes": {
                    "routing.chosen_agent": "text_search",
                    "routing.confidence": 0.85,
                    "routing.processing_time": 45.0,
                },
                "events": [],
            },
        ]

        metrics = evaluator.calculate_metrics(routing_spans)

        # Video search precision: 2 / (2 + 1) = 0.667
        assert "video_search" in metrics.per_agent_precision
        assert metrics.per_agent_precision["video_search"] == pytest.approx(2 / 3)

        # Text search precision: 1 / (1 + 0) = 1.0
        assert "text_search" in metrics.per_agent_precision
        assert metrics.per_agent_precision["text_search"] == 1.0


class TestProviderQuery:
    """Test telemetry provider span querying functionality"""

    @pytest.mark.asyncio
    async def test_query_routing_spans_success(self, mock_provider):
        """Test successful query of routing spans from telemetry provider"""
        from unittest.mock import AsyncMock

        import pandas as pd

        # Mock provider response
        mock_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "parent_id": "p1",
                    "status_code": "OK",
                    "start_time": pd.Timestamp("2024-01-01 10:00:00"),
                    "attributes": {"routing.chosen_agent": "video_search"},
                }
            ]
        )
        mock_provider.traces.get_spans = AsyncMock(return_value=mock_df)

        evaluator = _routing_evaluator(mock_provider)
        spans = await evaluator.query_routing_spans(limit=10)

        assert len(spans) == 1
        assert spans[0]["name"] == "cogniverse.routing"

    @pytest.mark.asyncio
    async def test_query_routing_spans_empty_result(self, mock_provider):
        """Test query with no matching spans"""
        from unittest.mock import AsyncMock

        import pandas as pd

        mock_provider.traces.get_spans = AsyncMock(return_value=pd.DataFrame())

        evaluator = _routing_evaluator(mock_provider)
        spans = await evaluator.query_routing_spans()

        assert spans == []

    @pytest.mark.asyncio
    async def test_query_routing_spans_with_time_range(self, mock_provider):
        """Test query with time range filters"""
        from unittest.mock import AsyncMock

        import pandas as pd

        mock_provider.traces.get_spans = AsyncMock(return_value=pd.DataFrame())

        evaluator = _routing_evaluator(mock_provider)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        await evaluator.query_routing_spans(start_time=start, end_time=end)

        # Verify get_spans was called with time range and project name
        mock_provider.traces.get_spans.assert_called_once_with(
            project=TEST_PROJECT_NAME,
            start_time=start,
            end_time=end,
            filters={"name": "cogniverse.routing"},
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_query_routing_spans_failure_raises_error(self, mock_provider):
        """Test that query failure raises RuntimeError"""
        from unittest.mock import AsyncMock

        mock_provider.traces.get_spans = AsyncMock(
            side_effect=Exception("Telemetry provider connection failed")
        )

        evaluator = _routing_evaluator(mock_provider)

        with pytest.raises(RuntimeError, match="Failed to query routing spans"):
            await evaluator.query_routing_spans()


class TestQueryRoutingSpansAwaited:
    """query_routing_spans is async; the optimization dashboard tab must await
    it (via run_async_in_streamlit) before passing the result to
    calculate_metrics, which iterates it. This guards the await -> list ->
    metrics sequence the tab performs — the pre-fix bug passed the raw coroutine
    to calculate_metrics, which then failed iterating a coroutine.
    """

    def test_awaited_query_yields_list_consumable_by_calculate_metrics(self):
        import asyncio
        from unittest.mock import AsyncMock

        import pandas as pd

        spans = [
            {
                "name": "cogniverse.routing",
                "parent_id": "p1",
                "status_code": "OK",
                "start_time": datetime(2026, 5, 1, 0, 0, 1),
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.95,
                    "routing.processing_time": 50.0,
                },
                "events": [],
            },
            {
                "name": "cogniverse.routing",
                "parent_id": "p2",
                "status_code": "OK",
                "start_time": datetime(2026, 5, 1, 0, 0, 2),
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.90,
                    "routing.processing_time": 60.0,
                },
                "events": [],
            },
        ]
        provider = MagicMock()
        provider.traces = MagicMock()
        provider.traces.get_spans = AsyncMock(return_value=pd.DataFrame(spans))
        evaluator = _routing_evaluator(provider)

        result = asyncio.run(evaluator.query_routing_spans(limit=1000))
        # Must be a concrete list, never a coroutine.
        assert isinstance(result, list)
        assert len(result) == 2

        metrics = evaluator.calculate_metrics(result)
        assert metrics.total_decisions == 2
        assert metrics.routing_accuracy == 1.0


class TestConfidenceCoercion:
    """Routers emit confidence as floats, labels ("high") or percents ("85%").
    Every form must coerce through parse_confidence — a bare float() raised
    ValueError on labels and the caller dropped the span, so a label-emitting
    router lost ALL routing metrics."""

    def test_label_confidence_is_coerced(self, mock_provider):
        evaluator = _routing_evaluator(mock_provider)
        span_data = {
            "name": "cogniverse.routing",
            "status_code": "OK",
            "attributes": {
                "routing.chosen_agent": "video_search",
                "routing.confidence": "high",
                "routing.processing_time": 10.0,
            },
            "events": [],
        }

        _, metrics = evaluator.evaluate_routing_decision(span_data)

        assert 0.0 < metrics["confidence"] <= 1.0

    def test_percent_confidence_is_coerced(self, mock_provider):
        evaluator = _routing_evaluator(mock_provider)
        span_data = {
            "name": "cogniverse.routing",
            "status_code": "OK",
            "attributes": {
                "routing.chosen_agent": "video_search",
                "routing.confidence": "85%",
                "routing.processing_time": 10.0,
            },
            "events": [],
        }

        _, metrics = evaluator.evaluate_routing_decision(span_data)

        assert metrics["confidence"] == 0.85


class TestLatencyAndNumpyCoercion:
    def test_non_numeric_latency_does_not_abort_batch(self, mock_provider):
        """A span with a None/list processing_time must not raise a TypeError
        that aborts the whole calculate_metrics batch — the ValueError-only
        caller guard didn't catch it."""
        evaluator = _routing_evaluator(mock_provider)
        span = {
            "name": "cogniverse.routing",
            "status_code": "OK",
            "attributes": {
                "routing.chosen_agent": "video_search",
                "routing.confidence": 0.9,
                "routing.processing_time": None,
            },
            "events": [],
        }
        _, metrics = evaluator.evaluate_routing_decision(span)
        # Neither a usable processing_time nor a start and end: no timing,
        # rather than a 0 ms decision.
        assert metrics["latency_ms"] is None

    def test_numpy_bool_confidence_is_not_zero(self, mock_provider):
        """A confidence read as np.bool_(True) from a pandas row must map to
        1.0, not silently to 0.0."""
        import numpy as np

        evaluator = _routing_evaluator(mock_provider)
        span = {
            "name": "cogniverse.routing",
            "status_code": "OK",
            "attributes": {
                "routing.chosen_agent": "video_search",
                "routing.confidence": np.bool_(True),
                "routing.processing_time": 10.0,
            },
            "events": [],
        }
        _, metrics = evaluator.evaluate_routing_decision(span)
        assert metrics["confidence"] == 1.0


def _gateway_routing_span(parent_id, agent, start, duration_ms):
    """A routing span as Phoenix returns the gateway's: the decision on
    ``output.value`` with no ``processing_time``, timed by its start and end."""
    import json

    import pandas as pd

    begin = pd.Timestamp(start, tz="UTC")
    return {
        "name": "cogniverse.routing",
        "parent_id": parent_id,
        "status_code": "UNSET",
        "attributes.output.value": json.dumps(
            {"chosen_agent": agent, "recommended_agent": agent, "confidence": 0.9}
        ),
        "start_time": begin,
        "end_time": begin + pd.Timedelta(milliseconds=duration_ms),
        "events": [],
    }


class TestDecisionTimeFromTheSpan:
    def test_a_decision_without_processing_time_takes_its_span_duration(
        self, mock_provider
    ):
        span = _gateway_routing_span("p-1", "search_agent", "2026-10-09 10:00", 412.5)

        _, metrics = _routing_evaluator(mock_provider).evaluate_routing_decision(span)

        assert metrics["latency_ms"] == 412.5

    def test_a_recorded_processing_time_wins_over_the_span_duration(
        self, mock_provider
    ):
        import json

        span = _gateway_routing_span("p-1", "search_agent", "2026-10-09 10:00", 412.5)
        decision = json.loads(span["attributes.output.value"])
        span["attributes.output.value"] = json.dumps(
            {**decision, "processing_time": 37.0}
        )

        _, metrics = _routing_evaluator(mock_provider).evaluate_routing_decision(span)

        assert metrics["latency_ms"] == 37.0

    def test_the_mean_leaves_out_decisions_with_no_timing(self, mock_provider):
        untimed = _gateway_routing_span("p-3", "search_agent", "2026-10-09 10:02", 0)
        del untimed["end_time"]
        spans = [
            _gateway_routing_span("p-1", "search_agent", "2026-10-09 10:00", 300.0),
            _gateway_routing_span("p-2", "summarizer_agent", "2026-10-09 10:01", 500.0),
            untimed,
        ]

        metrics = _routing_evaluator(mock_provider).calculate_metrics(spans)

        assert (metrics.total_decisions, metrics.avg_routing_latency) == (3, 400.0)

    def test_no_timed_decision_leaves_the_mean_unset(self, mock_provider):
        span = _gateway_routing_span("p-1", "search_agent", "2026-10-09 10:00", 0)
        del span["start_time"]

        metrics = _routing_evaluator(mock_provider).calculate_metrics([span])

        assert (metrics.total_decisions, metrics.avg_routing_latency) == (1, None)
