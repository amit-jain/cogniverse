"""
Unit tests for RoutingEvaluator.

Tests the routing evaluation metrics calculation without requiring
actual Phoenix infrastructure.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.evaluation.evaluators.routing_evaluator import (
    RoutingEvaluator,
    RoutingMetrics,
    RoutingOutcome,
)


@pytest.mark.unit
class TestRoutingEvaluator:
    """Test RoutingEvaluator initialization and basic functionality"""

    def test_evaluator_initialization(self):
        """Test RoutingEvaluator initializes correctly"""
        evaluator = RoutingEvaluator()
        assert evaluator.client is not None
        assert evaluator.logger is not None

    def test_evaluator_with_custom_client(self):
        """Test RoutingEvaluator accepts custom Phoenix client"""
        mock_client = MagicMock()
        evaluator = RoutingEvaluator(phoenix_client=mock_client)
        assert evaluator.client is mock_client


@pytest.mark.unit
class TestRoutingDecisionEvaluation:
    """Test evaluation of individual routing decisions"""

    def test_evaluate_successful_routing_decision(self):
        """Test evaluation of a successful routing decision"""
        evaluator = RoutingEvaluator()

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

    def test_evaluate_failed_routing_decision(self):
        """Test evaluation of a failed routing decision"""
        evaluator = RoutingEvaluator()

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

    def test_evaluate_ambiguous_routing_decision(self):
        """Test evaluation of ambiguous routing decision (no parent span)"""
        evaluator = RoutingEvaluator()

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

    def test_evaluate_invalid_span_name(self):
        """Test that evaluator raises error for non-routing spans"""
        evaluator = RoutingEvaluator()

        span_data = {
            "name": "cogniverse.search",  # Wrong span name
            "attributes": {},
        }

        with pytest.raises(ValueError, match="Expected cogniverse.routing span"):
            evaluator.evaluate_routing_decision(span_data)

    def test_evaluate_missing_required_attributes(self):
        """Test that evaluator raises error when required attributes are missing"""
        evaluator = RoutingEvaluator()

        span_data = {
            "name": "cogniverse.routing",
            "attributes": {
                # Missing routing.chosen_agent and routing.confidence
            },
        }

        with pytest.raises(ValueError, match="missing required attributes"):
            evaluator.evaluate_routing_decision(span_data)


@pytest.mark.unit
class TestMetricsCalculation:
    """Test calculation of aggregated routing metrics"""

    def test_calculate_metrics_with_all_successful(self):
        """Test metrics calculation when all routing decisions succeed"""
        evaluator = RoutingEvaluator()

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

    def test_calculate_metrics_with_mixed_outcomes(self):
        """Test metrics calculation with mixed success/failure"""
        evaluator = RoutingEvaluator()

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

    def test_calculate_metrics_empty_list_raises_error(self):
        """Test that empty span list raises ValueError"""
        evaluator = RoutingEvaluator()

        with pytest.raises(ValueError, match="Cannot calculate metrics from empty"):
            evaluator.calculate_metrics([])

    def test_calculate_metrics_all_invalid_spans_raises_error(self):
        """Test that list with only invalid spans raises ValueError"""
        evaluator = RoutingEvaluator()

        invalid_spans = [
            {
                "name": "wrong_span_name",
                "attributes": {},
            }
        ]

        with pytest.raises(ValueError, match="No valid routing spans found"):
            evaluator.calculate_metrics(invalid_spans)


@pytest.mark.unit
class TestConfidenceCalibration:
    """Test confidence calibration metric calculation"""

    def test_perfect_calibration(self):
        """Test confidence calibration with perfect correlation"""
        evaluator = RoutingEvaluator()

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


@pytest.mark.unit
class TestPerAgentMetrics:
    """Test per-agent precision/recall/F1 calculation"""

    def test_per_agent_precision(self):
        """Test precision calculation for different agents"""
        evaluator = RoutingEvaluator()

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


@pytest.mark.unit
class TestPhoenixQuery:
    """Test Phoenix span querying functionality"""

    @patch("src.evaluation.evaluators.routing_evaluator.px.Client")
    def test_query_routing_spans_success(self, mock_client_class):
        """Test successful query of routing spans from Phoenix"""
        import pandas as pd

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock Phoenix response
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
        mock_client.get_spans_dataframe.return_value = mock_df

        evaluator = RoutingEvaluator()
        spans = evaluator.query_routing_spans(limit=10)

        assert len(spans) == 1
        assert spans[0]["name"] == "cogniverse.routing"

    @patch("src.evaluation.evaluators.routing_evaluator.px.Client")
    def test_query_routing_spans_empty_result(self, mock_client_class):
        """Test query with no matching spans"""
        import pandas as pd

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_spans_dataframe.return_value = pd.DataFrame()

        evaluator = RoutingEvaluator()
        spans = evaluator.query_routing_spans()

        assert spans == []

    @patch("src.evaluation.evaluators.routing_evaluator.px.Client")
    def test_query_routing_spans_with_time_range(self, mock_client_class):
        """Test query with time range filters"""
        import pandas as pd

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_spans_dataframe.return_value = pd.DataFrame()

        evaluator = RoutingEvaluator()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        evaluator.query_routing_spans(start_time=start, end_time=end)

        # Verify get_spans_dataframe was called with time range and project name
        mock_client.get_spans_dataframe.assert_called_once_with(
            project_name="cogniverse-default-routing-optimization",
            start_time=start,
            end_time=end
        )

    @patch("src.evaluation.evaluators.routing_evaluator.px.Client")
    def test_query_routing_spans_failure_raises_error(self, mock_client_class):
        """Test that query failure raises RuntimeError"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_spans_dataframe.side_effect = Exception("Phoenix connection failed")

        evaluator = RoutingEvaluator()

        with pytest.raises(RuntimeError, match="Failed to query routing spans"):
            evaluator.query_routing_spans()
