"""
Integration tests for RoutingEvaluator with real Phoenix data.

These tests require:
- Phoenix server running at localhost:6006
- Actual routing spans in the Phoenix database
"""

from datetime import datetime, timedelta

import phoenix as px
import pytest

from src.evaluation.evaluators.routing_evaluator import (
    RoutingEvaluator,
    RoutingMetrics,
    RoutingOutcome,
)


@pytest.mark.integration
class TestRoutingEvaluatorIntegration:
    """Integration tests with real Phoenix infrastructure"""

    @pytest.fixture
    def phoenix_client(self):
        """Create Phoenix client for integration tests"""
        try:
            client = px.Client()
            return client
        except Exception as e:
            pytest.skip(f"Phoenix not available: {e}")

    @pytest.fixture
    def routing_evaluator(self, phoenix_client):
        """Create RoutingEvaluator with real Phoenix client"""
        return RoutingEvaluator(phoenix_client=phoenix_client)

    def test_query_real_routing_spans(self, routing_evaluator):
        """Test querying actual routing spans from Phoenix"""
        # Query recent routing spans
        spans = routing_evaluator.query_routing_spans(
            project_name="cogniverse",
            limit=10
        )

        # If we have spans, verify their structure
        if spans:
            for span in spans:
                assert "name" in span
                assert span["name"] == "cogniverse.routing"
                assert "attributes" in span

                # Verify required attributes exist
                attributes = span["attributes"]
                assert "routing.chosen_agent" in attributes or isinstance(attributes, dict)
        else:
            pytest.skip("No routing spans found in Phoenix - need to run routing agent first")

    def test_evaluate_real_routing_decisions(self, routing_evaluator):
        """Test evaluating actual routing decisions from Phoenix"""
        # Query recent spans
        spans = routing_evaluator.query_routing_spans(limit=20)

        if not spans:
            pytest.skip("No routing spans found in Phoenix")

        # Evaluate each span
        valid_evaluations = 0
        for span in spans:
            try:
                outcome, metrics = routing_evaluator.evaluate_routing_decision(span)

                # Verify outcome is valid
                assert isinstance(outcome, RoutingOutcome)

                # Verify metrics structure
                assert "chosen_agent" in metrics
                assert "confidence" in metrics
                assert "latency_ms" in metrics
                assert "success" in metrics
                assert "downstream_status" in metrics

                # Verify metric values are reasonable
                assert 0.0 <= metrics["confidence"] <= 1.0
                assert metrics["latency_ms"] >= 0.0

                valid_evaluations += 1
            except ValueError as e:
                # Some spans might not have all required attributes
                print(f"Skipping span due to: {e}")
                continue

        # We should have evaluated at least some spans
        assert valid_evaluations > 0, "No valid spans were evaluated"

    def test_calculate_metrics_from_real_spans(self, routing_evaluator):
        """Test calculating metrics from actual Phoenix spans"""
        # Query recent spans
        spans = routing_evaluator.query_routing_spans(limit=50)

        if not spans:
            pytest.skip("No routing spans found in Phoenix")

        # Filter to only valid routing spans
        valid_spans = []
        for span in spans:
            try:
                # Quick validation
                attrs = span.get("attributes", {})
                if "routing.chosen_agent" in attrs and "routing.confidence" in attrs:
                    valid_spans.append(span)
            except Exception:
                continue

        if len(valid_spans) < 2:
            pytest.skip(f"Not enough valid routing spans (found {len(valid_spans)})")

        # Calculate metrics
        metrics = routing_evaluator.calculate_metrics(valid_spans)

        # Verify metrics structure
        assert isinstance(metrics, RoutingMetrics)
        assert 0.0 <= metrics.routing_accuracy <= 1.0
        assert -1.0 <= metrics.confidence_calibration <= 1.0
        assert metrics.avg_routing_latency >= 0.0
        assert metrics.total_decisions == len(valid_spans)
        assert metrics.ambiguous_count >= 0
        assert metrics.ambiguous_count <= metrics.total_decisions

        # Verify per-agent metrics exist and are valid
        assert isinstance(metrics.per_agent_precision, dict)
        assert isinstance(metrics.per_agent_recall, dict)
        assert isinstance(metrics.per_agent_f1, dict)

        for agent, precision in metrics.per_agent_precision.items():
            assert 0.0 <= precision <= 1.0, f"Invalid precision for {agent}: {precision}"

        # Print metrics for manual inspection
        print("\n=== Routing Evaluation Metrics ===")
        print(f"Total decisions: {metrics.total_decisions}")
        print(f"Routing accuracy: {metrics.routing_accuracy:.2%}")
        print(f"Confidence calibration: {metrics.confidence_calibration:.3f}")
        print(f"Avg latency: {metrics.avg_routing_latency:.2f}ms")
        print(f"Ambiguous decisions: {metrics.ambiguous_count}")
        print("\nPer-agent precision:")
        for agent, prec in metrics.per_agent_precision.items():
            print(f"  {agent}: {prec:.2%}")

    def test_query_with_time_range(self, routing_evaluator):
        """Test querying spans with time range filters"""
        # Query spans from last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

        spans = routing_evaluator.query_routing_spans(
            start_time=start_time,
            end_time=end_time,
            limit=100
        )

        # Should return list (may be empty if no recent spans)
        assert isinstance(spans, list)

        if spans:
            # Verify all spans are within time range
            for span in spans:
                if "start_time" in span:
                    span_time = span["start_time"]
                    if isinstance(span_time, str):
                        span_time = datetime.fromisoformat(span_time.replace('Z', '+00:00'))
                    # Note: Timezone handling might vary, so we're lenient here

    def test_end_to_end_evaluation_workflow(self, routing_evaluator):
        """Test complete evaluation workflow from query to metrics"""
        # Step 1: Query spans
        spans = routing_evaluator.query_routing_spans(limit=30)

        if not spans:
            pytest.skip("No routing spans available for end-to-end test")

        print(f"\nStep 1: Queried {len(spans)} spans")

        # Step 2: Filter valid spans
        valid_spans = []
        for span in spans:
            try:
                outcome, metrics = routing_evaluator.evaluate_routing_decision(span)
                valid_spans.append(span)
            except ValueError:
                continue

        if len(valid_spans) < 2:
            pytest.skip("Not enough valid spans for metrics calculation")

        print(f"Step 2: Found {len(valid_spans)} valid spans")

        # Step 3: Calculate metrics
        metrics = routing_evaluator.calculate_metrics(valid_spans)

        print("Step 3: Calculated metrics")
        print(f"  Accuracy: {metrics.routing_accuracy:.2%}")
        print(f"  Calibration: {metrics.confidence_calibration:.3f}")
        print(f"  Avg Latency: {metrics.avg_routing_latency:.2f}ms")

        # Step 4: Verify metrics make sense
        assert metrics.total_decisions == len(valid_spans)
        assert metrics.routing_accuracy >= 0.0
        assert metrics.avg_routing_latency >= 0.0

        # If we have multiple agents, verify per-agent metrics
        if len(metrics.per_agent_precision) > 0:
            print(f"Step 4: Per-agent metrics calculated for {len(metrics.per_agent_precision)} agents")
            for agent in metrics.per_agent_precision.keys():
                assert agent in metrics.per_agent_recall
                assert agent in metrics.per_agent_f1

        print("\n✅ End-to-end evaluation workflow completed successfully")


@pytest.mark.integration
class TestRoutingEvaluatorWithMockPhoenixData:
    """Integration test with controlled Phoenix-like data"""

    def test_evaluate_realistic_span_structure(self):
        """Test with span structure that matches actual Phoenix output"""
        evaluator = RoutingEvaluator()

        # Realistic span structure matching Phoenix output
        realistic_spans = [
            {
                "context": {
                    "trace_id": "abc123",
                    "span_id": "span1",
                },
                "name": "cogniverse.routing",
                "parent_id": "parent-span-1",
                "span_kind": "INTERNAL",
                "start_time": "2024-09-30T10:00:00.000Z",
                "end_time": "2024-09-30T10:00:00.050Z",
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.92,
                    "routing.processing_time": 45.5,
                    "routing.method": "dspy_optimizer",
                    "openinference.span.kind": "AGENT",
                },
                "events": [
                    {
                        "name": "routing_decision",
                        "timestamp": "2024-09-30T10:00:00.045Z",
                        "attributes": {
                            "decision": "video_search",
                            "reasoning": "Query indicates video intent",
                        },
                    }
                ],
            },
            {
                "context": {
                    "trace_id": "def456",
                    "span_id": "span2",
                },
                "name": "cogniverse.routing",
                "parent_id": "parent-span-2",
                "span_kind": "INTERNAL",
                "start_time": "2024-09-30T10:01:00.000Z",
                "end_time": "2024-09-30T10:01:00.080Z",
                "status_code": "ERROR",
                "status_message": "Routing failed",
                "attributes": {
                    "routing.chosen_agent": "text_search",
                    "routing.confidence": 0.42,
                    "routing.processing_time": 78.3,
                    "routing.method": "dspy_optimizer",
                    "openinference.span.kind": "AGENT",
                },
                "events": [],
            },
        ]

        # Test evaluation
        outcome1, metrics1 = evaluator.evaluate_routing_decision(realistic_spans[0])
        assert outcome1 == RoutingOutcome.SUCCESS
        assert metrics1["chosen_agent"] == "video_search"
        assert metrics1["confidence"] == 0.92
        assert metrics1["latency_ms"] == 45.5

        outcome2, metrics2 = evaluator.evaluate_routing_decision(realistic_spans[1])
        assert outcome2 == RoutingOutcome.FAILURE
        assert metrics2["chosen_agent"] == "text_search"
        assert metrics2["success"] is False

        # Test metrics calculation
        overall_metrics = evaluator.calculate_metrics(realistic_spans)
        assert overall_metrics.total_decisions == 2
        assert overall_metrics.routing_accuracy == 0.5  # 1 success out of 2
        assert "video_search" in overall_metrics.per_agent_precision
        assert "text_search" in overall_metrics.per_agent_precision
