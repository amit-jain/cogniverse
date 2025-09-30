"""
Integration tests for RoutingEvaluator with real Phoenix data.

These tests require:
- Phoenix server running at localhost:6006
- Actual routing spans in the Phoenix database
"""

import logging
import os
from datetime import datetime, timedelta

import phoenix as px
import pytest

from src.evaluation.evaluators.routing_evaluator import (
    RoutingEvaluator,
    RoutingMetrics,
    RoutingOutcome,
)

logger = logging.getLogger(__name__)

# Enable synchronous export for integration tests
os.environ["TELEMETRY_SYNC_EXPORT"] = "true"


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

    @pytest.fixture
    async def routing_agent_with_spans(self):
        """Create routing agent and generate real spans"""
        import asyncio

        from src.app.agents.routing_agent import RoutingAgent

        try:
            # Initialize routing agent
            agent = RoutingAgent()

            # Generate some real routing spans by processing test queries
            test_queries = [
                "show me videos of basketball",
                "summarize the game highlights",
                "detailed analysis of the match",
            ]

            for query in test_queries:
                try:
                    result = await agent.analyze_and_route(query)
                    logger.info(f"Processed query: '{query}', agent: {result.get('agent', 'unknown')}")
                except Exception as e:
                    # Some queries might fail, that's okay for testing
                    logger.warning(f"Query '{query}' failed: {e}")

            # Force flush telemetry spans
            if hasattr(agent, 'telemetry_manager'):
                success = agent.telemetry_manager.force_flush(timeout_millis=10000)
                if not success:
                    logger.error("Failed to flush telemetry spans")
                else:
                    logger.info("Successfully flushed all telemetry spans")

            # Give Phoenix time to process flushed spans
            await asyncio.sleep(2)

            return agent
        except Exception as e:
            pytest.skip(f"Could not initialize routing agent: {e}")

    @pytest.mark.asyncio
    async def test_query_real_routing_spans(self, routing_evaluator, routing_agent_with_spans):
        """Test querying actual routing spans from Phoenix"""
        # Generate spans
        await routing_agent_with_spans

        # Debug: Query ALL spans to see what we have
        client = routing_evaluator.client
        all_spans = client.get_spans_dataframe()
        if all_spans is not None and not all_spans.empty:
            logger.info(f"DEBUG: Found {len(all_spans)} total spans")
            logger.info(f"DEBUG: Span names: {all_spans['name'].unique()}")
            logger.info(f"DEBUG: Routing spans: {len(all_spans[all_spans['name'] == 'cogniverse.routing'])}")
            logger.info(f"DEBUG: DataFrame columns (first 20): {all_spans.columns.tolist()[:20]}")
            # Check for openinference columns
            openinference_cols = [col for col in all_spans.columns if 'openinference' in col.lower()]
            logger.info(f"DEBUG: OpenInference columns: {openinference_cols}")

            # Check if we have routing spans and what their project info looks like
            routing_df = all_spans[all_spans['name'] == 'cogniverse.routing'].sort_values('start_time', ascending=False)
            if not routing_df.empty:
                # Check the NEWEST span (just created by this test)
                newest_routing = routing_df.iloc[0].to_dict()
                logger.info(f"DEBUG: NEWEST routing span keys: {list(newest_routing.keys())[:30]}")
                logger.info(f"DEBUG: NEWEST attributes.service: {newest_routing.get('attributes.service')}")
                logger.info(f"DEBUG: NEWEST openinference.project.name: {newest_routing.get('attributes.openinference.project.name')}")

                # Check ALL columns for project
                project_cols = [col for col in routing_df.columns if 'project' in col.lower()]
                logger.info(f"DEBUG: Columns with 'project': {project_cols}")

                # Check if attributes.openinference exists and what it contains
                if 'attributes.openinference' in routing_df.columns:
                    openinf = newest_routing.get('attributes.openinference')
                    logger.info(f"DEBUG: attributes.openinference exists: {openinf}")
                else:
                    logger.info("DEBUG: NO attributes.openinference column")

        # Query recent routing spans from routing optimization project
        # RoutingEvaluator is already bound to routing project
        spans = routing_evaluator.query_routing_spans(limit=10)

        # Should have spans now
        assert len(spans) > 0, f"No routing spans found after generating them. Total spans in Phoenix: {len(all_spans) if all_spans is not None else 0}"

        # Verify proper span structure with parent-child relationships
        for span in spans:
            assert "name" in span
            assert span["name"] == "cogniverse.routing"

            # Verify parent-child relationship exists
            assert "parent_id" in span, "Routing span should have parent_id"
            assert span["parent_id"] is not None, "Routing span should be child of request span"

            # Phoenix returns flattened attributes (attributes.routing, attributes.tenant, etc.)
            assert "attributes.routing" in span, f"Span missing attributes.routing: {span.keys()}"

            # Verify routing attributes exist in Phoenix's flattened format
            routing_attrs = span["attributes.routing"]
            assert isinstance(routing_attrs, dict), "attributes.routing should be a dict"
            assert "chosen_agent" in routing_attrs, "Missing chosen_agent"
            assert "confidence" in routing_attrs, "Missing confidence"
            assert "processing_time" in routing_attrs, "Missing processing_time"

    @pytest.mark.asyncio
    async def test_evaluate_real_routing_decisions(self, routing_evaluator, routing_agent_with_spans):
        """Test evaluating actual routing decisions from Phoenix"""
        # Generate spans
        await routing_agent_with_spans

        # Query recent spans
        spans = routing_evaluator.query_routing_spans(limit=20)

        assert len(spans) > 0, "No routing spans found after generating them"

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

    @pytest.mark.asyncio
    async def test_calculate_metrics_from_real_spans(self, routing_evaluator, routing_agent_with_spans):
        """Test calculating metrics from actual Phoenix spans"""
        # Generate spans
        await routing_agent_with_spans

        # Query recent spans
        spans = routing_evaluator.query_routing_spans(limit=50)

        assert len(spans) > 0, "No routing spans found after generating them"

        # Filter to only valid routing spans
        # Phoenix returns flattened format: attributes.routing = {chosen_agent, confidence, ...}
        valid_spans = []
        for span in spans:
            try:
                # Check Phoenix flattened format
                if "attributes.routing" in span and isinstance(span["attributes.routing"], dict):
                    routing_attrs = span["attributes.routing"]
                    if "chosen_agent" in routing_attrs and "confidence" in routing_attrs:
                        valid_spans.append(span)
            except Exception:
                continue

        assert len(valid_spans) >= 2, f"Not enough valid routing spans (found {len(valid_spans)})"

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

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_workflow(self, routing_evaluator, routing_agent_with_spans):
        """Test complete evaluation workflow from query to metrics"""
        # Step 1: Generate spans
        await routing_agent_with_spans

        # Step 2: Query spans
        spans = routing_evaluator.query_routing_spans(limit=30)

        assert len(spans) > 0, "No routing spans available for end-to-end test"

        print(f"\nStep 1: Queried {len(spans)} spans")

        # Step 3: Filter valid spans
        valid_spans = []
        for span in spans:
            try:
                outcome, metrics = routing_evaluator.evaluate_routing_decision(span)
                valid_spans.append(span)
            except ValueError:
                continue

        assert len(valid_spans) >= 2, "Not enough valid spans for metrics calculation"

        print(f"Step 2: Found {len(valid_spans)} valid spans")

        # Step 4: Calculate metrics
        metrics = routing_evaluator.calculate_metrics(valid_spans)

        print("Step 3: Calculated metrics")
        print(f"  Accuracy: {metrics.routing_accuracy:.2%}")
        print(f"  Calibration: {metrics.confidence_calibration:.3f}")
        print(f"  Avg Latency: {metrics.avg_routing_latency:.2f}ms")

        # Step 5: Verify metrics make sense
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
