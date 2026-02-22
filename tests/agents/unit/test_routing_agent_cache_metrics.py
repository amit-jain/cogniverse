"""
Unit tests for cache and metrics integration in RoutingAgent:
- Cache checking and storing
- Metrics tracking
- Production components
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_agents.search.multi_modal_reranker import QueryModality
from cogniverse_foundation.telemetry.config import TelemetryConfig


class TestRoutingAgentCacheMetrics:
    """Test cache and metrics integration in RoutingAgent"""

    @pytest.fixture(scope="function")
    def routing_agent(self):
        """Create routing agent with test configuration"""
        # Reset TelemetryManager singleton to avoid conflicts with previous tests
        from cogniverse_foundation.telemetry.manager import TelemetryManager

        TelemetryManager._instance = None

        # Create telemetry config
        telemetry_config = TelemetryConfig(
            enabled=False,  # Disable telemetry for unit tests (cache/metrics don't need it)
            provider_config={},
        )

        # Create deps with test configuration
        deps = RoutingDeps(
            telemetry_config=telemetry_config,
            enable_caching=True,
            enable_metrics_tracking=True,
            enable_parallel_execution=True,
            enable_contextual_analysis=True,
            enable_memory=False,  # Disabled for unit tests
            enable_advanced_optimization=False,  # Disabled for unit tests
            enable_relationship_extraction=False,  # Disabled for faster tests
            enable_query_enhancement=False,  # Disabled for faster tests
        )

        # Mock all external dependencies to avoid network calls and delays
        # Use patch.object for dspy.settings.configure to avoid attribute cleanup issues
        def _mock_configure_dspy(self_agent, deps_arg):
            """Mock _configure_dspy that sets _dspy_lm to a MagicMock."""
            self_agent._dspy_lm = MagicMock()

        with (
            patch.object(RoutingAgent, "_configure_dspy", _mock_configure_dspy),
            patch("cogniverse_core.agents.a2a_agent.FastAPI"),
            patch("cogniverse_core.agents.a2a_agent.A2AClient"),
            patch(
                "cogniverse_agents.routing.cross_modal_optimizer.ModalitySpanCollector"
            ),
        ):
            agent = RoutingAgent(deps=deps, port=8001)

            # Yield agent for test use
            yield agent

            # Cleanup after each test to prevent state pollution
            if hasattr(agent, "cache_manager") and agent.cache_manager:
                # Clear all caches
                agent.cache_manager.invalidate_all()
                agent.cache_manager.reset_stats()

            if hasattr(agent, "metrics_tracker") and agent.metrics_tracker:
                # Reset all metrics
                agent.metrics_tracker.reset_all_stats()

            # Reset TelemetryManager singleton after test
            from cogniverse_foundation.telemetry.manager import TelemetryManager

            TelemetryManager._instance = None

    def test_cache_metrics_components_initialized(self, routing_agent):
        """Test that cache and metrics components are initialized"""
        assert hasattr(routing_agent, "parallel_executor")
        assert hasattr(routing_agent, "cache_manager")
        assert hasattr(routing_agent, "lazy_executor")
        assert hasattr(routing_agent, "metrics_tracker")

        assert routing_agent.parallel_executor is not None
        assert routing_agent.cache_manager is not None
        assert routing_agent.lazy_executor is not None
        assert routing_agent.metrics_tracker is not None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, routing_agent):
        """Test that cache hit returns cached result without processing"""
        query = "machine learning videos"

        # Create a RoutingDecision to cache
        from datetime import datetime

        from cogniverse_agents.routing_agent import RoutingOutput

        cached_decision = RoutingOutput(
            query=query,
            recommended_agent="video_search_agent",
            confidence=0.9,
            reasoning="Cached routing decision",
            enhanced_query=query,
            entities=[],
            relationships=[],
            metadata={"cached": True},
            timestamp=datetime.now(),
        )

        # Pre-populate cache (video_search_agent maps to VIDEO modality)
        routing_agent.cache_manager.cache_result(
            query, QueryModality.VIDEO, cached_decision
        )

        # Mock the routing module to ensure it's not called on cache hit
        with patch.object(routing_agent.routing_module, "forward") as mock_forward:
            # Call route_query
            result = await routing_agent.route_query(query, tenant_id="test_tenant")

            # Verify cached result was returned
            assert result.query == cached_decision.query
            assert result.recommended_agent == cached_decision.recommended_agent
            assert result.confidence == cached_decision.confidence

            # Verify routing module was not called (cache hit)
            mock_forward.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_processes_query(self, routing_agent):
        """Test that cache miss processes query normally"""
        query = "deep learning tutorials"

        # Mock the routing module to return a prediction
        mock_prediction = MagicMock()
        mock_prediction.recommended_agent = "video_search_agent"
        mock_prediction.confidence = 0.9
        mock_prediction.reasoning = "Test routing decision"
        mock_prediction.primary_intent = "search"
        mock_prediction.complexity_score = 0.5

        with patch.object(
            routing_agent.routing_module, "forward", return_value=mock_prediction
        ):
            # Call route_query
            result = await routing_agent.route_query(query, tenant_id="test_tenant")

            # Verify routing was performed
            assert result.query == query
            assert result.recommended_agent == "video_search_agent"
            assert result.confidence > 0

            # Verify result was cached (video_search_agent maps to VIDEO modality)
            cached = routing_agent.cache_manager.get_cached_result(
                query, QueryModality.VIDEO
            )
            assert cached is not None
            assert cached.query == result.query
            assert cached.recommended_agent == result.recommended_agent

    @pytest.mark.asyncio
    async def test_metrics_tracked_on_success(self, routing_agent):
        """Test that metrics are tracked on successful execution"""
        query = "machine learning"

        # Mock the routing module
        mock_prediction = MagicMock()
        mock_prediction.recommended_agent = "video_search_agent"
        mock_prediction.confidence = 0.9
        mock_prediction.reasoning = "Test routing decision"
        mock_prediction.primary_intent = "search"
        mock_prediction.complexity_score = 0.5

        with patch.object(
            routing_agent.routing_module, "forward", return_value=mock_prediction
        ):
            # Call route_query
            await routing_agent.route_query(query, tenant_id="test_tenant")

            # Verify metrics were recorded
            stats = routing_agent.metrics_tracker.get_summary_stats()
            assert stats["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_metrics_tracked_on_failure(self, routing_agent):
        """Test that metrics are tracked on failed execution"""
        query = "test query"

        # Mock routing module to raise exception
        with patch.object(
            routing_agent.routing_module,
            "forward",
            side_effect=Exception("Routing failed"),
        ):
            # Call route_query (should not raise, returns fallback decision)
            result = await routing_agent.route_query(query, tenant_id="test_tenant")

            # Verify fallback decision was returned
            assert result is not None
            assert result.query == query

            # Routing stats are still tracked even on failure
            stats = routing_agent.get_routing_statistics()
            assert stats["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_cache_stores_result_after_processing(self, routing_agent):
        """Test that result is cached after processing"""
        query = "neural networks"

        # Ensure cache is empty (video_search_agent maps to VIDEO modality)
        cached = routing_agent.cache_manager.get_cached_result(
            query, QueryModality.VIDEO
        )
        assert cached is None

        # Mock the routing module
        mock_prediction = MagicMock()
        mock_prediction.recommended_agent = "video_search_agent"
        mock_prediction.confidence = 0.9
        mock_prediction.reasoning = "Test routing decision"
        mock_prediction.primary_intent = "search"
        mock_prediction.complexity_score = 0.5

        with patch.object(
            routing_agent.routing_module, "forward", return_value=mock_prediction
        ):
            # Process query
            result = await routing_agent.route_query(query, tenant_id="test_tenant")

            # Verify result was cached (video_search_agent maps to VIDEO modality)
            cached = routing_agent.cache_manager.get_cached_result(
                query, QueryModality.VIDEO
            )
            assert cached is not None
            assert cached.query == result.query
            assert cached.recommended_agent == result.recommended_agent
            assert cached.confidence == result.confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
