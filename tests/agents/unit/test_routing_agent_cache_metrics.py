"""
Unit tests for cache and metrics integration in RoutingAgent:
- Cache checking and storing
- Metrics tracking
- Production components
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.routing_agent import RoutingAgent, RoutingConfig
from cogniverse_agents.search.multi_modal_reranker import QueryModality


class TestRoutingAgentCacheMetrics:
    """Test cache and metrics integration in RoutingAgent"""

    @pytest.fixture(scope="class")
    def routing_agent(self):
        """Create routing agent with test configuration"""
        # Create minimal config for testing
        config = RoutingConfig(
            enable_caching=True,
            enable_metrics_tracking=True,
            enable_parallel_execution=True,
            enable_contextual_analysis=True,
            enable_memory=False,  # Disabled for unit tests
            enable_mlflow_tracking=False,  # Disabled for unit tests
            enable_advanced_optimization=False,  # Disabled for unit tests
            enable_relationship_extraction=False,  # Disabled for faster tests
            enable_query_enhancement=False,  # Disabled for faster tests
        )

        # Mock all external dependencies to avoid network calls and delays
        # Use patch.object for dspy.settings.configure to avoid attribute cleanup issues
        with patch.object(RoutingAgent, "_configure_dspy", return_value=None), \
             patch("cogniverse_agents.dspy_a2a_agent_base.FastAPI"), \
             patch("cogniverse_agents.dspy_a2a_agent_base.A2AClient"):
            agent = RoutingAgent(tenant_id="test_tenant", config=config, port=8001, enable_telemetry=False)
            yield agent

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

        from cogniverse_agents.routing_agent import RoutingDecision

        cached_decision = RoutingDecision(
            query=query,
            recommended_agent="video_search_agent",
            confidence=0.9,
            reasoning="Cached routing decision",
            enhanced_query=query,
            entities=[],
            relationships=[],
            metadata={"cached": True},
            timestamp=datetime.now()
        )

        # Pre-populate cache (routing uses TEXT modality by default)
        routing_agent.cache_manager.cache_result(
            query, QueryModality.TEXT, cached_decision
        )

        # Mock the routing module to ensure it's not called on cache hit
        with patch.object(routing_agent.routing_module, 'forward') as mock_forward:
            # Call route_query
            result = await routing_agent.route_query(query)

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

        with patch.object(routing_agent.routing_module, 'forward', return_value=mock_prediction):
            # Call route_query
            result = await routing_agent.route_query(query)

            # Verify routing was performed
            assert result.query == query
            assert result.recommended_agent == "video_search_agent"
            assert result.confidence > 0

            # Verify result was cached (routing uses TEXT modality by default)
            cached = routing_agent.cache_manager.get_cached_result(query, QueryModality.TEXT)
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

        with patch.object(routing_agent.routing_module, 'forward', return_value=mock_prediction):
            # Call route_query
            await routing_agent.route_query(query)

            # Verify metrics were recorded
            stats = routing_agent.metrics_tracker.get_summary_stats()
            assert stats["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_metrics_tracked_on_failure(self, routing_agent):
        """Test that metrics are tracked on failed execution"""
        query = "test query"

        # Mock routing module to raise exception
        with patch.object(routing_agent.routing_module, 'forward', side_effect=Exception("Routing failed")):
            # Call route_query (should not raise, returns fallback decision)
            result = await routing_agent.route_query(query)

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

        # Ensure cache is empty (routing uses TEXT modality by default)
        cached = routing_agent.cache_manager.get_cached_result(
            query, QueryModality.TEXT
        )
        assert cached is None

        # Mock the routing module
        mock_prediction = MagicMock()
        mock_prediction.recommended_agent = "video_search_agent"
        mock_prediction.confidence = 0.9
        mock_prediction.reasoning = "Test routing decision"
        mock_prediction.primary_intent = "search"
        mock_prediction.complexity_score = 0.5

        with patch.object(routing_agent.routing_module, 'forward', return_value=mock_prediction):
            # Process query
            result = await routing_agent.route_query(query)

            # Verify result was cached (routing uses TEXT modality by default)
            cached = routing_agent.cache_manager.get_cached_result(query, QueryModality.TEXT)
            assert cached is not None
            assert cached.query == result.query
            assert cached.recommended_agent == result.recommended_agent
            assert cached.confidence == result.confidence



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
