"""
Integration tests for RoutingSpanEvaluator

Tests the complete flow of:
1. Generating real routing spans via RoutingAgent
2. Querying spans from Phoenix
3. Extracting RoutingExperience objects
4. Feeding experiences to AdvancedRoutingOptimizer

Uses real Phoenix via shared phoenix_container fixture from tests/conftest.py.
"""

import asyncio
import logging
import tempfile

import pytest

from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.routing_span_evaluator import RoutingSpanEvaluator
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)

_TEST_TENANT = "span_evaluator_test"


@pytest.fixture
def real_telemetry_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)


@pytest.fixture
async def routing_agent_with_spans(telemetry_manager_with_phoenix):
    """Create routing agent and generate real routing spans"""
    from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps

    agent = RoutingAgent(
        deps=RoutingDeps(telemetry_config=telemetry_manager_with_phoenix.config)
    )

    test_queries = [
        ("show me videos of basketball", "video_search_agent"),
        ("summarize the game highlights", "summarizer_agent"),
        ("detailed analysis of the match", "detailed_report_agent"),
        ("find soccer highlights", "video_search_agent"),
    ]

    for query, expected_agent in test_queries:
        result = await agent.route_query(query, tenant_id=_TEST_TENANT)
        agent_name = result.recommended_agent if result else "unknown"
        logger.info(f"Processed query: '{query}', agent: {agent_name}")

    success = agent.telemetry_manager.force_flush(timeout_millis=10000)
    if not success:
        logger.error("Failed to flush telemetry spans")

    # Poll for spans to arrive in Phoenix (gRPC export is async)
    from phoenix.client import Client
    client = Client(base_url="http://localhost:16006")
    project = telemetry_manager_with_phoenix.config.get_project_name(
        _TEST_TENANT, "routing"
    )
    for _ in range(15):
        await asyncio.sleep(2)
        try:
            df = client.spans.get_spans_dataframe(project_identifier=project)
            routing = df[df["name"] == "cogniverse.routing"] if df is not None else []
            if len(routing) >= 3:
                break
        except Exception:
            pass

    return agent


@pytest.fixture(scope="function")
def optimizer(real_telemetry_provider):
    """Create AdvancedRoutingOptimizer for testing - fresh for each test"""
    return AdvancedRoutingOptimizer(
        tenant_id=_TEST_TENANT,
        llm_config=LLMEndpointConfig(
            model="ollama/gemma3:4b", api_base="http://localhost:11434"
        ),
        telemetry_provider=real_telemetry_provider,
    )


@pytest.fixture(scope="function")
def span_evaluator(optimizer):
    """Create RoutingSpanEvaluator for testing - fresh for each test"""
    return RoutingSpanEvaluator(optimizer=optimizer, tenant_id=_TEST_TENANT)


class TestRoutingSpanEvaluatorIntegration:
    """Integration tests for RoutingSpanEvaluator"""

    @pytest.mark.asyncio
    async def test_query_real_routing_spans(
        self, routing_agent_with_spans, span_evaluator, telemetry_manager_with_phoenix
    ):
        """Test querying real routing spans from Phoenix"""
        _ = routing_agent_with_spans

        provider = telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)
        spans_df = await provider.traces.get_spans(project=span_evaluator.project_name)

        routing_spans = spans_df[spans_df["name"] == "cogniverse.routing"]

        # 4 queries are routed; allow for 1 gRPC export failure due to timing
        assert len(routing_spans) >= 3, (
            f"Expected at least 3 routing spans, found {len(routing_spans)}"
        )

    @pytest.mark.asyncio
    async def test_extract_routing_experiences(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test extracting routing experiences from real spans"""
        _ = routing_agent_with_spans

        results = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results["spans_processed"] >= 4, (
            f"Expected at least 4 spans processed, got {results['spans_processed']}"
        )
        assert results["experiences_created"] >= 4, (
            f"Expected at least 4 experiences created, got {results['experiences_created']}"
        )

    @pytest.mark.asyncio
    async def test_feed_experiences_to_optimizer(
        self, telemetry_manager_with_phoenix, real_telemetry_provider
    ):
        """Test feeding extracted experiences to AdvancedRoutingOptimizer"""
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps

        agent = RoutingAgent(
            deps=RoutingDeps(telemetry_config=telemetry_manager_with_phoenix.config)
        )

        unique_queries = [
            ("show me tennis matches", "video_search_agent"),
            ("summarize the tennis tournament", "summarizer_agent"),
            ("detailed report on tennis finals", "detailed_report_agent"),
            ("find tennis highlights", "video_search_agent"),
        ]

        for query, _ in unique_queries:
            result = await agent.route_query(query, tenant_id=_TEST_TENANT)
            agent_name = result.recommended_agent if result else "unknown"
            logger.info(f"Processed query: '{query}', agent: {agent_name}")

        success = agent.telemetry_manager.force_flush(timeout_millis=10000)
        if not success:
            logger.error("Failed to flush telemetry spans")
        await asyncio.sleep(2)

        with tempfile.TemporaryDirectory() as _:
            optimizer = AdvancedRoutingOptimizer(
                tenant_id=_TEST_TENANT,
                llm_config=LLMEndpointConfig(
                    model="ollama/gemma3:4b", api_base="http://localhost:11434"
                ),
                telemetry_provider=real_telemetry_provider,
            )

            initial_count = len(optimizer.experience_replay)
            assert initial_count == 0, (
                f"Expected empty optimizer, got {initial_count} experiences"
            )

            evaluator = RoutingSpanEvaluator(
                optimizer=optimizer, tenant_id=_TEST_TENANT
            )

            await evaluator.evaluate_routing_spans(lookback_hours=1)

            final_count = len(evaluator.optimizer.experience_replay)

            experiences_added = final_count - initial_count

            assert experiences_added >= 4, (
                f"Expected at least 4 experiences added to optimizer, got {experiences_added}"
            )

    @pytest.mark.asyncio
    async def test_routing_experience_structure(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that extracted RoutingExperience objects have correct structure"""
        _ = routing_agent_with_spans

        await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        experiences = span_evaluator.optimizer.experience_replay

        assert len(experiences) >= 4, (
            f"Expected at least 4 experiences, got {len(experiences)}"
        )

        exp = experiences[0]
        assert exp.query, "Experience should have query"
        assert exp.chosen_agent, "Experience should have chosen_agent"
        assert 0.0 <= exp.routing_confidence <= 1.0, "Confidence should be in [0,1]"
        assert 0.0 <= exp.search_quality <= 1.0, "Search quality should be in [0,1]"
        assert isinstance(exp.agent_success, bool), "agent_success should be boolean"

    @pytest.mark.asyncio
    async def test_duplicate_span_filtering(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that duplicate spans are not processed twice"""
        _ = routing_agent_with_spans

        await span_evaluator.evaluate_routing_spans(lookback_hours=1)
        results2 = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results2["experiences_created"] == 0, (
            f"Expected 0 new experiences on second run (duplicates), "
            f"got {results2['experiences_created']}"
        )

    @pytest.mark.asyncio
    async def test_span_extraction_with_batch_size(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test span extraction respects batch size limits"""
        _ = routing_agent_with_spans

        results = await span_evaluator.evaluate_routing_spans(
            lookback_hours=1, batch_size=2
        )

        assert results["spans_processed"] <= 2, (
            f"Expected at most 2 spans with batch_size=2, got {results['spans_processed']}"
        )
        assert results["experiences_created"] <= 2, (
            f"Expected at most 2 experiences with batch_size=2, "
            f"got {results['experiences_created']}"
        )

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_workflow(
        self, optimizer, telemetry_manager_with_phoenix
    ):
        """Test complete end-to-end workflow from span generation to experience creation"""
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps

        agent = RoutingAgent(
            deps=RoutingDeps(telemetry_config=telemetry_manager_with_phoenix.config)
        )

        query = "show me basketball dunks"
        await agent.route_query(query, tenant_id=_TEST_TENANT)

        agent.telemetry_manager.force_flush(timeout_millis=10000)
        await asyncio.sleep(2)

        evaluator = RoutingSpanEvaluator(optimizer=optimizer, tenant_id=_TEST_TENANT)

        results = await evaluator.evaluate_routing_spans(lookback_hours=1)

        assert results["spans_processed"] >= 1, "Should process at least 1 span"
        assert results["experiences_created"] >= 1, (
            "Should create at least 1 experience"
        )

        assert len(optimizer.experience_replay) >= 1, (
            "Optimizer should have experiences"
        )

        queries = [exp.query for exp in optimizer.experience_replay]
        assert query in queries, (
            f"Expected query '{query}' in experiences, got: {queries}"
        )

        exp = next(exp for exp in optimizer.experience_replay if exp.query == query)
        logger.info(
            f"End-to-end workflow complete: {exp.chosen_agent} (confidence: {exp.routing_confidence})"
        )

    @pytest.mark.asyncio
    async def test_realistic_span_structure(
        self, routing_agent_with_spans, span_evaluator, telemetry_manager_with_phoenix
    ):
        """Test that spans have the expected structure"""
        _ = routing_agent_with_spans

        provider = telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)
        spans_df = await provider.traces.get_spans(project=span_evaluator.project_name)

        routing_spans = spans_df[spans_df["name"] == "cogniverse.routing"]

        assert len(routing_spans) >= 4, (
            f"Expected at least 4 routing spans, found {len(routing_spans)}"
        )

        first_span = routing_spans.iloc[0]

        assert first_span["name"] == "cogniverse.routing", (
            "Span name should be cogniverse.routing"
        )

        has_flattened = (
            "attributes.routing" in first_span and first_span["attributes.routing"]
        )
        has_nested = "attributes" in first_span and any(
            k.startswith("routing.") for k in first_span.get("attributes", {}).keys()
        )

        assert has_flattened or has_nested, (
            "Span should have routing attributes in either flattened or nested format"
        )
