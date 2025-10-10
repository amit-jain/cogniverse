"""
Integration tests for PhoenixSpanEvaluator

Tests the complete flow of:
1. Generating real routing spans via RoutingAgent
2. Querying spans from Phoenix
3. Extracting RoutingExperience objects
4. Feeding experiences to AdvancedRoutingOptimizer
"""

import asyncio
import logging
import os

import phoenix as px
import pytest

from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.phoenix_span_evaluator import PhoenixSpanEvaluator

# Set synchronous export for integration tests
os.environ["TELEMETRY_SYNC_EXPORT"] = "true"

logger = logging.getLogger(__name__)


@pytest.fixture
async def routing_agent_with_spans():
    """Create routing agent and generate real routing spans"""
    from cogniverse_agents.routing_agent import RoutingAgent

    agent = RoutingAgent()

    # Generate real routing spans by processing test queries
    test_queries = [
        ("show me videos of basketball", "video_search_agent"),
        ("summarize the game highlights", "summarizer_agent"),
        ("detailed analysis of the match", "detailed_report_agent"),
        ("find soccer highlights", "video_search_agent"),
    ]

    logger.info("🔄 Generating routing spans...")
    for query, expected_agent in test_queries:
        result = await agent.route_query(query, user_id="test-tenant")
        # result is a RoutingDecision object
        agent_name = result.recommended_agent if result else 'unknown'
        logger.info(
            f"✅ Processed query: '{query}', "
            f"agent: {agent_name}, "
            f"expected: {expected_agent}"
        )

    # Force flush telemetry spans
    success = agent.telemetry_manager.force_flush(timeout_millis=10000)
    if not success:
        logger.error("❌ Failed to flush telemetry spans")
    else:
        logger.info("✅ Telemetry spans flushed")

    # Give Phoenix time to process spans
    await asyncio.sleep(2)

    return agent


@pytest.fixture(scope="function")
def optimizer():
    """Create AdvancedRoutingOptimizer for testing - fresh for each test"""
    return AdvancedRoutingOptimizer()


@pytest.fixture(scope="function")
def span_evaluator(optimizer):
    """Create PhoenixSpanEvaluator for testing - fresh for each test"""
    return PhoenixSpanEvaluator(optimizer=optimizer, tenant_id="test-tenant")


class TestPhoenixSpanEvaluatorIntegration:
    """Integration tests for PhoenixSpanEvaluator"""

    @pytest.mark.asyncio
    async def test_query_real_routing_spans(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test querying real routing spans from Phoenix"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Query spans from Phoenix
        logger.info(f"📊 Querying spans from project: {span_evaluator.project_name}")

        phoenix_client = px.Client()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=span_evaluator.project_name
        )

        logger.info(f"📊 Total spans in project: {len(spans_df)}")

        # Filter for routing spans
        routing_spans = spans_df[spans_df["name"] == "cogniverse.routing"]
        logger.info(f"📊 Routing spans found: {len(routing_spans)}")

        assert (
            len(routing_spans) >= 4
        ), f"Expected at least 4 routing spans, found {len(routing_spans)}"

    @pytest.mark.asyncio
    async def test_extract_routing_experiences(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test extracting routing experiences from real spans"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate routing spans
        logger.info("🔍 Evaluating routing spans...")
        results = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        logger.info(f"📊 Evaluation results: {results}")

        assert (
            results["spans_processed"] >= 4
        ), f"Expected at least 4 spans processed, got {results['spans_processed']}"
        assert (
            results["experiences_created"] >= 4
        ), f"Expected at least 4 experiences created, got {results['experiences_created']}"

    @pytest.mark.asyncio
    async def test_feed_experiences_to_optimizer(self):
        """Test feeding extracted experiences to AdvancedRoutingOptimizer"""
        import tempfile

        from cogniverse_agents.routing_agent import RoutingAgent

        # Create fresh routing agent and generate unique spans
        agent = RoutingAgent()

        # Use unique queries to avoid span ID collisions with other tests
        unique_queries = [
            ("show me tennis matches", "video_search_agent"),
            ("summarize the tennis tournament", "summarizer_agent"),
            ("detailed report on tennis finals", "detailed_report_agent"),
            ("find tennis highlights", "video_search_agent"),
        ]

        logger.info("🔄 Generating unique routing spans for optimizer test...")
        for query, _ in unique_queries:
            result = await agent.route_query(query, user_id="test-tenant")
            agent_name = result.recommended_agent if result else 'unknown'
            logger.info(
                f"✅ Processed query: '{query}', agent: {agent_name}"
            )

        # Force flush telemetry spans
        success = agent.telemetry_manager.force_flush(timeout_millis=10000)
        if not success:
            logger.error("❌ Failed to flush telemetry spans")
        else:
            logger.info("✅ Telemetry spans flushed")
        await asyncio.sleep(2)

        # Create optimizer with temporary storage to avoid loading existing data
        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer = AdvancedRoutingOptimizer(storage_dir=temp_dir)

            # Verify optimizer starts empty (no loaded data)
            initial_count = len(optimizer.experience_replay)
            logger.info(f"📊 Initial experience count: {initial_count}")
            assert (
                initial_count == 0
            ), f"Expected empty optimizer, got {initial_count} experiences"

            # Create span evaluator with our optimizer
            evaluator = PhoenixSpanEvaluator(optimizer=optimizer, tenant_id="test-tenant")

            # Evaluate routing spans
            logger.info(f"📊 Evaluating spans from project: {evaluator.project_name}")
            results = await evaluator.evaluate_routing_spans(lookback_hours=1)
            logger.info(f"📊 Evaluation results: {results}")

            # Check that experiences were added to optimizer
            final_count = len(evaluator.optimizer.experience_replay)
            logger.info(f"📊 Final experience count: {final_count}")

            experiences_added = final_count - initial_count
            logger.info(f"📊 Experiences added: {experiences_added}")

            assert (
                experiences_added >= 4
            ), f"Expected at least 4 experiences added to optimizer, got {experiences_added}"

    @pytest.mark.asyncio
    async def test_routing_experience_structure(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that extracted RoutingExperience objects have correct structure"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate routing spans
        await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        # Get experiences from optimizer
        experiences = span_evaluator.optimizer.experience_replay

        assert (
            len(experiences) >= 4
        ), f"Expected at least 4 experiences, got {len(experiences)}"

        # Validate structure of first experience
        exp = experiences[0]
        assert exp.query, "Experience should have query"
        assert exp.chosen_agent, "Experience should have chosen_agent"
        assert 0.0 <= exp.routing_confidence <= 1.0, "Confidence should be in [0,1]"
        assert 0.0 <= exp.search_quality <= 1.0, "Search quality should be in [0,1]"
        assert isinstance(exp.agent_success, bool), "agent_success should be boolean"

        logger.info(f"✅ Experience structure validated: {exp}")

    @pytest.mark.asyncio
    async def test_duplicate_span_filtering(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that duplicate spans are not processed twice"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate routing spans twice
        results1 = await span_evaluator.evaluate_routing_spans(lookback_hours=1)
        results2 = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        logger.info(
            f"📊 First evaluation: {results1['experiences_created']} experiences"
        )
        logger.info(
            f"📊 Second evaluation: {results2['experiences_created']} experiences"
        )

        # Second evaluation should create 0 new experiences (duplicates filtered)
        assert results2["experiences_created"] == 0, (
            f"Expected 0 new experiences on second run (duplicates), "
            f"got {results2['experiences_created']}"
        )

    @pytest.mark.asyncio
    async def test_span_extraction_with_batch_size(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test span extraction respects batch size limits"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate with small batch size
        results = await span_evaluator.evaluate_routing_spans(
            lookback_hours=1, batch_size=2
        )

        logger.info(f"📊 Batch size limited results: {results}")

        # Should process at most 2 spans
        assert (
            results["spans_processed"] <= 2
        ), f"Expected at most 2 spans with batch_size=2, got {results['spans_processed']}"
        assert results["experiences_created"] <= 2, (
            f"Expected at most 2 experiences with batch_size=2, "
            f"got {results['experiences_created']}"
        )

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_workflow(self, optimizer):
        """Test complete end-to-end workflow from span generation to experience creation"""
        from cogniverse_agents.routing_agent import RoutingAgent

        # 1. Create fresh routing agent
        agent = RoutingAgent()

        # 2. Process a single query
        query = "show me basketball dunks"
        logger.info(f"🔄 Processing query: '{query}'")
        result = await agent.route_query(query, user_id="test-tenant")
        logger.info(f"✅ Result: {result}")

        # 3. Flush telemetry
        agent.telemetry_manager.force_flush(timeout_millis=10000)
        await asyncio.sleep(2)

        # 4. Create span evaluator
        evaluator = PhoenixSpanEvaluator(optimizer=optimizer, tenant_id="test-tenant")

        # 5. Evaluate spans
        results = await evaluator.evaluate_routing_spans(lookback_hours=1)
        logger.info(f"📊 Evaluation results: {results}")

        # 6. Verify experience was created
        assert results["spans_processed"] >= 1, "Should process at least 1 span"
        assert (
            results["experiences_created"] >= 1
        ), "Should create at least 1 experience"

        # 7. Verify experience is in optimizer replay buffer
        assert (
            len(optimizer.experience_replay) >= 1
        ), "Optimizer should have experiences"

        # 8. Verify experience content - check that the query exists in any experience
        queries = [exp.query for exp in optimizer.experience_replay]
        assert (
            query in queries
        ), f"Expected query '{query}' in experiences, got: {queries}"

        # Get the matching experience
        exp = next(exp for exp in optimizer.experience_replay if exp.query == query)
        logger.info(
            f"✅ End-to-end workflow complete: {exp.chosen_agent} (confidence: {exp.routing_confidence})"
        )

    @pytest.mark.asyncio
    async def test_realistic_span_structure(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that spans have the expected structure from Phase 1"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Query Phoenix directly to inspect span structure
        phoenix_client = px.Client()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=span_evaluator.project_name
        )

        routing_spans = spans_df[spans_df["name"] == "cogniverse.routing"]

        assert (
            len(routing_spans) >= 4
        ), f"Expected at least 4 routing spans, found {len(routing_spans)}"

        # Inspect first routing span structure
        first_span = routing_spans.iloc[0]
        logger.info(f"📊 Span structure: {first_span.to_dict()}")

        # Check for expected attributes
        assert (
            first_span["name"] == "cogniverse.routing"
        ), "Span name should be cogniverse.routing"

        # Check for routing attributes (either flattened or nested)
        has_flattened = (
            "attributes.routing" in first_span and first_span["attributes.routing"]
        )
        has_nested = "attributes" in first_span and any(
            k.startswith("routing.") for k in first_span.get("attributes", {}).keys()
        )

        assert (
            has_flattened or has_nested
        ), "Span should have routing attributes in either flattened or nested format"

        if has_flattened:
            routing_attrs = first_span["attributes.routing"]
            logger.info(f"📊 Flattened routing attributes: {routing_attrs}")
            assert "chosen_agent" in routing_attrs, "Should have chosen_agent"
            assert "confidence" in routing_attrs, "Should have confidence"
        else:
            logger.info("📊 Using nested attribute format")

        logger.info("✅ Span structure matches Phase 1 expectations")
