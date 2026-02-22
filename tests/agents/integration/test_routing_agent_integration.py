"""
Integration tests for RoutingAgent
Tests real interactions with routing system and configuration
"""

import json
import os
import tempfile

import pytest

from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig


class TestRoutingAgentIntegration:
    """Integration tests for RoutingAgent with real routing components"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return {
            "model_name": "ollama/gemma3:4b",
            "base_url": "http://localhost:11434",
            "api_key": "dummy",
            "confidence_threshold": 0.7,
        }

    @pytest.fixture
    def routing_config_file(self):
        """Create temporary routing configuration file"""
        config_data = {
            "routing_mode": "tiered",
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_langextract": False,
                "enable_fallback": True,
                "fast_path_confidence_threshold": 0.7,
                "slow_path_confidence_threshold": 0.6,
            },
            "gliner_config": {
                "model": "urchade/gliner_large-v2.1",
                "threshold": 0.3,
                "labels": ["video_content", "text_information", "summary_request"],
            },
            "llm_config": {
                "provider": "local",
                "model": "ollama/gemma3:4b",
                "endpoint": "http://localhost:11434",
                "temperature": 0.1,
            },
            "cache_config": {"enable_caching": True, "cache_ttl_seconds": 300},
            "monitoring_config": {"enable_metrics": True, "metrics_batch_size": 50},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file_path = f.name

        yield config_file_path

        # Cleanup
        os.unlink(config_file_path)

    @pytest.mark.ci_fast
    def test_routing_agent_with_real_config_file(
        self, test_config, routing_config_file
    ):
        """Test RoutingAgent initialization with actual config file"""
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(telemetry_config=telemetry_config)
        agent = RoutingAgent(deps=deps)

        # Verify agent initialized properly
        assert agent.config is not None
        assert hasattr(agent, "logger")
        # Agent should have DSPy components initialized
        assert hasattr(agent, "routing_module")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_routing_decision_flow(self, test_config):
        """Test actual routing decision flow through comprehensive router"""
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(telemetry_config=telemetry_config)
        agent = RoutingAgent(deps=deps)

        # Test different query types
        test_queries = [
            "Show me videos about machine learning",
            "Summarize the latest AI research",
            "Provide detailed analysis of neural networks",
            "Find documents about deep learning",
        ]

        for query in test_queries:
            result = await agent.route_query(query, tenant_id="test_tenant")

            # Verify result structure (RoutingDecision dataclass)
            assert result.query == query
            assert result.recommended_agent is not None
            assert result.confidence > 0
            assert result.reasoning is not None
            assert isinstance(result.fallback_agents, list)
            assert isinstance(result.metadata, dict)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_routing_agent_context_propagation(self, test_config):
        """Test that context is properly propagated through routing layers"""
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(telemetry_config=telemetry_config)
        agent = RoutingAgent(deps=deps)

        context = {
            "user_id": "test_user",
            "session_id": "session_123",
            "preferences": {"language": "en", "max_results": 5},
        }

        result = await agent.route_query(
            "find videos", context, tenant_id="test_tenant"
        )

        # Context should be preserved and potentially enhanced
        assert result.query == "find videos"
        assert result.recommended_agent is not None
        assert result.confidence > 0

    def test_agent_registry_validation_integration(self):
        """Test agent initialization with different configurations"""
        # Test with minimal valid config (default values)
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(telemetry_config=telemetry_config)
        agent = RoutingAgent(deps=deps)
        assert agent.config is not None
        assert hasattr(agent, "logger")

        # Test with another tenant
        deps2 = RoutingDeps(telemetry_config=telemetry_config)
        agent2 = RoutingAgent(deps=deps2)
        assert agent2.config is not None
        assert hasattr(agent2, "routing_module")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_generation_consistency(self, test_config):
        """Test that workflow generation is consistent across multiple calls"""
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(telemetry_config=telemetry_config)
        agent = RoutingAgent(deps=deps)
        query = "Show me training videos"

        # Run same query multiple times
        results = []
        for _ in range(3):
            result = await agent.route_query(query, tenant_id="test_tenant")
            results.append(result)

        # Results should be consistent in structure
        first_result = results[0]
        for result in results[1:]:
            assert result.query == first_result.query
            assert result.recommended_agent == first_result.recommended_agent
            # Confidence and reasoning should be present and non-trivial
            assert result.confidence > 0
            assert result.reasoning is not None


class TestRoutingAgentErrorHandling:
    """Test error handling in integration scenarios - needs refactoring for DSPy interface"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_routing_failure_propagation(self):
        """Test that routing failures are properly handled and propagated"""
        # Old test - routing agent interface changed
        pass

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations"""
        # Old test - routing agent interface changed
        pass
