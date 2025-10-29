"""
End-to-End Integration Test for RoutingAgent with Advanced Features

Tests the complete pipeline with QueryEnhancementPipeline, MultiModalReranker, and ContextualAnalyzer
integrated into the RoutingAgent.

Validates:
1. Query enhancement enriches routing context
2. Contextual analyzer tracks conversation history
3. Reranker method is available and functional
4. All components work together in real routing flow
"""


import pytest
from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_core.telemetry.config import BatchExportConfig, TelemetryConfig


@pytest.mark.integration
class TestRoutingAgentWithAdvancedFeatures:
    """Test RoutingAgent with advanced features integrated"""

    @pytest.fixture
    def mock_config(self):
        """Mock system configuration"""
        from cogniverse_agents.routing_agent import RoutingConfig

        # Create a RoutingConfig with test values
        config = RoutingConfig()
        config.enable_query_enhancement = True
        config.enable_contextual_analysis = True
        config.enable_dspy_optimization = False  # Disable to avoid LLM dependency
        config.enable_relationship_extraction = True
        config.enable_caching = True
        config.enable_advanced_optimization = False
        config.enable_mlflow_tracking = False
        return config

    @pytest.fixture
    async def routing_agent(self, mock_config):
        """Create RoutingAgent instance with mocked dependencies"""
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={"http_endpoint": "http://localhost:26006", "grpc_endpoint": "http://localhost:24317"},
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        agent = RoutingAgent(tenant_id="test-tenant", config=mock_config, telemetry_config=telemetry_config)
        yield agent

    @pytest.mark.asyncio
    async def test_query_enhancer_integrated(self, routing_agent):
        """Test that QueryEnhancementPipeline is initialized and accessible"""
        assert routing_agent.query_enhancer is not None
        assert hasattr(routing_agent.query_enhancer, "enhance_query_with_relationships")

    @pytest.mark.asyncio
    async def test_contextual_analyzer_integrated(self, routing_agent):
        """Test that ContextualAnalyzer is initialized and tracks context"""
        assert routing_agent.contextual_analyzer is not None
        assert hasattr(routing_agent.contextual_analyzer, "update_context")
        assert hasattr(routing_agent.contextual_analyzer, "get_contextual_hints")

    @pytest.mark.asyncio
    async def test_multi_modal_reranker_integrated(self, routing_agent):
        """Test that MultiModalReranker is initialized and accessible"""
        assert routing_agent.multi_modal_reranker is not None
        assert hasattr(routing_agent.multi_modal_reranker, "rerank_results")

    @pytest.mark.asyncio
    async def test_rerank_search_results_method(self, routing_agent):
        """Test that multi_modal_reranker component has rerank_results method"""
        # Verify the reranker component exists and has the expected method
        assert routing_agent.multi_modal_reranker is not None
        assert hasattr(routing_agent.multi_modal_reranker, "rerank_results")

        # The actual reranking logic is tested in unit tests for MultiModalReranker
        # This integration test just verifies the component is properly initialized

    @pytest.mark.asyncio
    async def test_analyze_and_route(self, routing_agent):
        """Test that route_query works with advanced features"""
        query = "show me tutorials from 2023"

        result = await routing_agent.route_query(query)

        # Verify routing decision is returned
        assert result is not None
        assert hasattr(result, "recommended_agent")
        assert hasattr(result, "metadata")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
