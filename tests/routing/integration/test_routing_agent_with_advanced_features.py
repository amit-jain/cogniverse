"""
End-to-End Integration Test for RoutingAgent with Advanced Multi-Modal Features

Tests the complete pipeline with QueryExpander, MultiModalReranker, and ContextualAnalyzer
integrated into the RoutingAgent.

Validates:
1. Query expansion enriches routing context
2. Contextual analyzer tracks conversation history
3. Reranker method is available and functional
4. All components work together in real routing flow
"""

from unittest.mock import MagicMock, patch

import pytest

from src.app.agents.routing_agent import RoutingAgent
from src.app.routing.base import GenerationType, RoutingDecision, SearchModality


@pytest.mark.integration
class TestRoutingAgentWithAdvancedFeatures:
    """Test RoutingAgent with Phase 10 components integrated"""

    @pytest.fixture
    def mock_config(self):
        """Mock system configuration"""
        return {
            "tenant_id": "test_tenant",
            "video_agent_url": "http://localhost:8001",
            "optimization_dir": "/tmp/test_optimization",
        }

    @pytest.fixture
    async def routing_agent(self, mock_config):
        """Create RoutingAgent instance with mocked dependencies"""
        with patch("src.app.agents.routing_agent.get_config", return_value=mock_config):
            with patch("src.app.agents.routing_agent.TelemetryManager"):
                with patch("src.app.agents.routing_agent.PhoenixSpanEvaluator"):
                    with patch("src.app.agents.routing_agent.MultiAgentOrchestrator"):
                        with patch(
                            "src.app.agents.routing_agent.PhoenixOrchestrationEvaluator"
                        ):
                            with patch("src.app.agents.routing_agent.UnifiedOptimizer"):
                                with patch(
                                    "src.app.agents.routing_agent.OrchestrationFeedbackLoop"
                                ):
                                    agent = RoutingAgent()
                                    yield agent

    @pytest.mark.asyncio
    async def test_query_expander_integrated(self, routing_agent):
        """Test that QueryExpander is initialized and accessible"""
        assert routing_agent.query_expander is not None
        assert hasattr(routing_agent.query_expander, "expand_query")

        # Test expansion
        query = "show me videos about machine learning from 2023"
        expansion = await routing_agent.query_expander.expand_query(query)

        assert "modality_intent" in expansion
        assert "temporal" in expansion
        assert expansion["temporal"]["requires_temporal_search"] is True

    @pytest.mark.asyncio
    async def test_contextual_analyzer_integrated(self, routing_agent):
        """Test that ContextualAnalyzer is initialized and tracks context"""
        assert routing_agent.contextual_analyzer is not None
        assert hasattr(routing_agent.contextual_analyzer, "update_context")
        assert hasattr(routing_agent.contextual_analyzer, "get_contextual_hints")

        # Initially no queries
        assert routing_agent.contextual_analyzer.total_queries == 0

        # Update context
        routing_agent.contextual_analyzer.update_context(
            query="machine learning videos",
            detected_modalities=["video"],
            result_count=5,
        )

        assert routing_agent.contextual_analyzer.total_queries == 1
        assert "video" in routing_agent.contextual_analyzer.modality_preferences

    @pytest.mark.asyncio
    async def test_multi_modal_reranker_integrated(self, routing_agent):
        """Test that MultiModalReranker is initialized and accessible"""
        assert routing_agent.multi_modal_reranker is not None
        assert hasattr(routing_agent.multi_modal_reranker, "rerank_results")

    @pytest.mark.asyncio
    async def test_rerank_search_results_method(self, routing_agent):
        """Test the rerank_search_results method"""
        # Create mock search results
        mock_results = [
            {
                "id": "video_1",
                "title": "ML Tutorial",
                "description": "Machine learning basics",
                "modality": "video",
                "relevance_score": 0.9,
                "metadata": {},
            },
            {
                "id": "doc_1",
                "title": "ML Paper",
                "description": "Machine learning research",
                "modality": "document",
                "relevance_score": 0.85,
                "metadata": {},
            },
        ]

        # Apply reranking
        reranked = await routing_agent.rerank_search_results(
            query="machine learning tutorial",
            results=mock_results,
            modality_intent=["video"],
            temporal_context={},
        )

        # Verify reranking applied
        assert len(reranked) == 2
        assert all("reranking_metadata" in r for r in reranked)
        assert all("relevance_score" in r for r in reranked)
        # Video should rank higher due to modality match
        assert reranked[0]["modality"] == "video"

    @pytest.mark.asyncio
    async def test_analyze_and_route_with_query_expansion(self, routing_agent):
        """Test that analyze_and_route uses query expansion"""
        query = "show me tutorials from 2023"

        # Mock the router to avoid actual routing logic
        mock_routing_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.9,
            routing_method="fast_path",
            reasoning="Visual content requested",
            detected_modalities=["video"],
        )

        with patch.object(
            routing_agent.router, "route", return_value=mock_routing_decision
        ):
            # Mock telemetry manager span context
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=None)

            with patch.object(
                routing_agent.telemetry_manager, "span", return_value=mock_span
            ):
                result = await routing_agent.analyze_and_route(query)

                # Verify query expansion is included
                assert "query_expansion" in result
                assert "modality_intent" in result["query_expansion"]
                assert "temporal" in result["query_expansion"]

                # Verify contextual hints are included
                assert "contextual_hints" in result

    @pytest.mark.asyncio
    async def test_context_tracking_across_queries(self, routing_agent):
        """Test that contextual analyzer tracks multiple queries"""
        queries = [
            ("machine learning videos", ["video"]),
            ("neural network diagrams", ["image"]),
            ("watch deep learning video tutorials", ["video"]),
        ]

        # Mock routing
        mock_routing_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.9,
            routing_method="fast_path",
            reasoning="Test",
            detected_modalities=["video"],
        )

        with patch.object(
            routing_agent.router, "route", return_value=mock_routing_decision
        ):
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=None)

            with patch.object(
                routing_agent.telemetry_manager, "span", return_value=mock_span
            ):
                for query, _ in queries:
                    await routing_agent.analyze_and_route(query)

                # Verify context accumulated
                assert routing_agent.contextual_analyzer.total_queries == 3

                # Verify modality preferences learned
                # Note: min_preference_count is 3, but video appears only 2 times
                # So we check modality_preferences directly
                assert "video" in routing_agent.contextual_analyzer.modality_preferences
                assert (
                    routing_agent.contextual_analyzer.modality_preferences["video"] >= 2
                )
                assert "image" in routing_agent.contextual_analyzer.modality_preferences
                assert (
                    routing_agent.contextual_analyzer.modality_preferences["image"] == 1
                )

    @pytest.mark.asyncio
    async def test_temporal_query_expansion_integration(self, routing_agent):
        """Test temporal expansion flows through to routing"""
        query = "videos from last week about python"

        # Expand query directly
        expansion = await routing_agent.query_expander.expand_query(query)

        # Verify temporal extraction
        assert expansion["temporal"]["requires_temporal_search"] is True
        assert expansion["temporal"]["temporal_type"] == "relative"
        assert "last week" in expansion["temporal"]["temporal_keywords"]

        # Test that this flows through to reranking
        from datetime import datetime

        mock_results = [
            {
                "id": "recent_video",
                "title": "Python Tutorial",
                "description": "Recent Python content",
                "modality": "video",
                "relevance_score": 0.85,
                "metadata": {},
                "timestamp": datetime(2025, 9, 25),  # Recent
            },
            {
                "id": "old_video",
                "title": "Python Basics",
                "description": "Python fundamentals",
                "modality": "video",
                "relevance_score": 0.90,
                "metadata": {},
                "timestamp": datetime(2020, 1, 1),  # Old
            },
        ]

        # Rerank with temporal context
        reranked = await routing_agent.rerank_search_results(
            query=query,
            results=mock_results,
            modality_intent=expansion["modality_intent"],
            temporal_context=expansion["temporal"],
        )

        # Recent content should benefit from temporal alignment
        # (though we can't guarantee it's first without mocking dates)
        assert len(reranked) == 2
        assert all("reranking_metadata" in r for r in reranked)

    @pytest.mark.asyncio
    async def test_cross_modal_fusion_hint_generation(self, routing_agent):
        """Test that contextual hints help with cross-modal queries"""
        # Build history: user prefers videos
        for i in range(3):
            routing_agent.contextual_analyzer.update_context(
                query=f"video query {i}", detected_modalities=["video"], result_count=5
            )

        # Now user asks about images
        routing_agent.contextual_analyzer.update_context(
            query="image query", detected_modalities=["image"], result_count=3
        )

        # Get hints for new query
        hints = routing_agent.contextual_analyzer.get_contextual_hints(
            "show me content about AI"
        )

        # Should identify video preference
        assert "preferred_modalities" in hints
        assert len(hints["preferred_modalities"]) > 0
        assert hints["preferred_modalities"][0]["modality"] == "video"

        # Should detect modality shift
        assert "conversation_context" in hints
        assert hints["conversation_context"]["modality_shifts"] > 0


@pytest.mark.integration
class TestFullPipelineWithRealComponents:
    """Test full pipeline with minimal mocking"""

    @pytest.mark.asyncio
    async def test_query_expansion_and_reranking_together(self):
        """Test QueryExpander and Reranker work together without RoutingAgent"""
        from datetime import datetime

        from src.app.routing.query_expansion import QueryExpander
        from src.app.search.multi_modal_reranker import (
            MultiModalReranker,
            QueryModality,
            SearchResult,
        )

        # Initialize components
        expander = QueryExpander()
        reranker = MultiModalReranker()

        # Expand query
        query = "show me machine learning tutorials from 2023"
        expansion = await expander.expand_query(query)

        # Create mock results
        results = [
            SearchResult(
                id="video_2023",
                title="ML Tutorial 2023",
                content="Machine learning tutorial from 2023",
                modality="video",
                score=0.85,
                metadata={},
                timestamp=datetime(2023, 6, 1),
            ),
            SearchResult(
                id="doc_2020",
                title="ML Paper 2020",
                content="Machine learning research paper",
                modality="document",
                score=0.90,
                metadata={},
                timestamp=datetime(2020, 1, 1),
            ),
        ]

        # Convert modality intent to QueryModality
        modalities = []
        for intent in expansion["modality_intent"]:
            if intent == "visual":
                modalities.extend([QueryModality.VIDEO, QueryModality.IMAGE])
            elif intent in ["video", "image", "audio", "document"]:
                modalities.append(QueryModality(intent.upper()))

        # Rerank with temporal context
        context = {"temporal": expansion["temporal"]}
        reranked = await reranker.rerank_results(results, query, modalities, context)

        # Verify reranking occurred
        assert len(reranked) == 2
        assert all("reranking_score" in r.metadata for r in reranked)

        # 2023 video should rank higher (video modality + temporal match)
        assert reranked[0].id == "video_2023"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
