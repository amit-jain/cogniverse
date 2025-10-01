"""
Unit tests for Phase 12 integration in RoutingAgent:
- Cache checking and storing
- Metrics tracking
- Helper methods
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.agents.routing_agent import RoutingAgent
from src.app.search.multi_modal_reranker import QueryModality


class TestRoutingAgentCacheMetrics:
    """Test cache and metrics integration in RoutingAgent"""

    @pytest.fixture
    def routing_agent(self):
        """Create routing agent with mocked dependencies"""
        with patch("src.app.agents.routing_agent.get_config") as mock_config:
            mock_config.return_value = {
                "video_agent_url": "http://localhost:8002",
                "optimization_dir": "/tmp/optimization",
            }
            agent = RoutingAgent()
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

    def test_detect_primary_modality_video(self, routing_agent):
        """Test video modality detection"""
        query = "show me machine learning videos"
        modality = routing_agent._detect_primary_modality_from_query(query)
        assert modality == QueryModality.VIDEO

    def test_detect_primary_modality_image(self, routing_agent):
        """Test image modality detection"""
        query = "find diagram of neural network"
        modality = routing_agent._detect_primary_modality_from_query(query)
        assert modality == QueryModality.IMAGE

    def test_detect_primary_modality_audio(self, routing_agent):
        """Test audio modality detection"""
        query = "listen to podcast about AI"
        modality = routing_agent._detect_primary_modality_from_query(query)
        assert modality == QueryModality.AUDIO

    def test_detect_primary_modality_document(self, routing_agent):
        """Test document modality detection"""
        query = "read research papers on deep learning"
        modality = routing_agent._detect_primary_modality_from_query(query)
        assert modality == QueryModality.DOCUMENT

    def test_detect_primary_modality_default_text(self, routing_agent):
        """Test default text modality"""
        query = "what is machine learning"
        modality = routing_agent._detect_primary_modality_from_query(query)
        assert modality == QueryModality.TEXT

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, routing_agent):
        """Test that cache hit returns cached result without processing"""
        query = "machine learning videos"
        cached_result = {
            "query": query,
            "routing_decision": {"agent": "video_search_agent"},
            "workflow_type": "single",
        }

        # Pre-populate cache
        routing_agent.cache_manager.cache_result(
            query, QueryModality.VIDEO, cached_result
        )

        # Mock router to ensure it's not called
        routing_agent.router.route = AsyncMock()

        # Call analyze_and_route
        result = await routing_agent.analyze_and_route(query)

        # Verify cached result was returned
        assert result == cached_result

        # Verify router was not called (cache hit)
        routing_agent.router.route.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_processes_query(self, routing_agent):
        """Test that cache miss processes query normally"""
        query = "deep learning tutorials"

        # Mock query expander
        routing_agent.query_expander.expand_query = AsyncMock(return_value={
            "modality_intent": ["video"],
            "temporal": {}
        })

        # Mock cross modal optimizer
        routing_agent.cross_modal_optimizer.get_fusion_recommendations = MagicMock(return_value=None)

        # Mock router
        mock_decision = MagicMock()
        mock_decision.requires_orchestration = False
        mock_decision.search_modality = QueryModality.VIDEO
        mock_decision.confidence_score = 0.9
        mock_decision.routing_method = "fast_path"
        mock_decision.detected_modalities = ["video"]
        mock_decision.to_dict = MagicMock(return_value={"agent": "video_search_agent"})
        routing_agent.router.route = AsyncMock(return_value=mock_decision)

        # Mock contextual analyzer
        routing_agent.contextual_analyzer.update_context = MagicMock()
        routing_agent.contextual_analyzer.get_contextual_hints = MagicMock(return_value=[])

        # Call analyze_and_route
        result = await routing_agent.analyze_and_route(query)

        # Verify router was called
        routing_agent.router.route.assert_called_once()

        # Verify result was cached - check all modalities to see where it was cached
        for modality in QueryModality:
            cached = routing_agent.cache_manager.get_cached_result(query, modality)
            if cached:
                # Found it - verify it matches
                assert cached is not None
                return

        # If we get here, nothing was cached
        assert False, f"Result was not cached in any modality. Result keys: {result.keys() if result else 'None'}"

    @pytest.mark.asyncio
    async def test_metrics_tracked_on_success(self, routing_agent):
        """Test that metrics are tracked on successful execution"""
        query = "machine learning"

        # Mock router
        mock_decision = MagicMock()
        mock_decision.requires_orchestration = False
        mock_decision.search_modality = QueryModality.VIDEO  # Use actual enum
        mock_decision.confidence_score = 0.9
        mock_decision.routing_method = "fast_path"
        mock_decision.detected_modalities = ["video"]
        mock_decision.to_dict = MagicMock(return_value={"agent": "video_search_agent"})

        routing_agent.router.route = AsyncMock(return_value=mock_decision)

        # Call analyze_and_route
        await routing_agent.analyze_and_route(query)

        # Verify metrics were recorded
        stats = routing_agent.metrics_tracker.get_summary_stats()
        assert stats["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_metrics_tracked_on_failure(self, routing_agent):
        """Test that metrics are tracked on failed execution"""
        query = "test query"

        # Mock router to raise exception
        routing_agent.router.route = AsyncMock(side_effect=Exception("Router failed"))

        # Call analyze_and_route (should raise)
        with pytest.raises(Exception):
            await routing_agent.analyze_and_route(query)

        # Verify error metrics were recorded
        stats = routing_agent.metrics_tracker.get_summary_stats()
        assert stats["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_cache_stores_result_after_processing(self, routing_agent):
        """Test that result is cached after processing"""
        query = "neural networks"

        # Ensure cache is empty
        cached = routing_agent.cache_manager.get_cached_result(
            query, QueryModality.VIDEO
        )
        assert cached is None

        # Mock query expander
        routing_agent.query_expander.expand_query = AsyncMock(return_value={
            "modality_intent": ["video"],
            "temporal": {}
        })

        # Mock cross modal optimizer
        routing_agent.cross_modal_optimizer.get_fusion_recommendations = MagicMock(return_value=None)

        # Mock router
        mock_decision = MagicMock()
        mock_decision.requires_orchestration = False
        mock_decision.search_modality = QueryModality.VIDEO
        mock_decision.confidence_score = 0.9
        mock_decision.routing_method = "fast_path"
        mock_decision.detected_modalities = ["video"]
        mock_decision.to_dict = MagicMock(return_value={"agent": "video_search_agent"})
        routing_agent.router.route = AsyncMock(return_value=mock_decision)

        # Mock contextual analyzer
        routing_agent.contextual_analyzer.update_context = MagicMock()
        routing_agent.contextual_analyzer.get_contextual_hints = MagicMock(return_value=[])

        # Process query
        result = await routing_agent.analyze_and_route(query)

        # Verify result was cached - check all modalities
        for modality in QueryModality:
            cached = routing_agent.cache_manager.get_cached_result(query, modality)
            if cached:
                # Found it - verify it matches
                assert cached == result
                return

        # If we get here, nothing was cached
        assert False, f"Result was not cached in any modality. Result keys: {result.keys() if result else 'None'}"

    def test_modality_detection_case_insensitive(self, routing_agent):
        """Test modality detection is case insensitive"""
        queries = [
            "SHOW ME VIDEOS",
            "Show Me Videos",
            "show me videos",
        ]

        for query in queries:
            modality = routing_agent._detect_primary_modality_from_query(query)
            assert modality == QueryModality.VIDEO

    def test_modality_detection_with_multiple_keywords(self, routing_agent):
        """Test modality detection prioritizes first match"""
        # Video keyword appears first
        query = "watch this video and read the document"
        modality = routing_agent._detect_primary_modality_from_query(query)
        assert modality == QueryModality.VIDEO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
