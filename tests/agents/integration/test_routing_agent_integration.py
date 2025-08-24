"""
Integration tests for RoutingAgent
Tests real interactions with routing system and configuration
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.app.agents.routing_agent import RoutingAgent, app, routing_agent
from src.tools.a2a_utils import A2AMessage, DataPart, TextPart


class TestRoutingAgentIntegration:
    """Integration tests for RoutingAgent with real routing components"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return {
            "video_agent_url": "http://localhost:8002",
            "text_agent_url": "http://localhost:8003",
            "routing_agent_port": 8001,
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
                "model": "gemma2:2b",
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

    @patch("src.app.agents.routing_agent.get_config")
    def test_routing_agent_with_real_config_file(
        self, mock_get_config, test_config, routing_config_file
    ):
        """Test RoutingAgent initialization with actual config file"""
        mock_get_config.return_value = test_config

        agent = RoutingAgent(config_path=routing_config_file)

        # Verify agent initialized properly
        assert agent.system_config == test_config
        assert agent.routing_config is not None
        assert agent.router is not None
        assert (
            len(agent.router.strategies) > 0
        )  # Should have at least fallback strategy

    @patch("src.app.agents.routing_agent.get_config")
    @pytest.mark.asyncio
    async def test_real_routing_decision_flow(self, mock_get_config, test_config):
        """Test actual routing decision flow through comprehensive router"""
        mock_get_config.return_value = test_config

        agent = RoutingAgent()

        # Test different query types
        test_queries = [
            "Show me videos about machine learning",
            "Summarize the latest AI research",
            "Provide detailed analysis of neural networks",
            "Find documents about deep learning",
        ]

        for query in test_queries:
            result = await agent.analyze_and_route(query)

            # Verify result structure
            assert "query" in result
            assert "routing_decision" in result
            assert "workflow_type" in result
            assert "agents_to_call" in result
            assert "execution_plan" in result
            assert "confidence" in result

            # Verify execution plan has valid structure
            assert len(result["execution_plan"]) > 0
            for step in result["execution_plan"]:
                assert "step" in step
                assert "agent" in step
                assert "action" in step
                assert "parameters" in step

    @patch("src.app.agents.routing_agent.get_config")
    @pytest.mark.asyncio
    async def test_routing_agent_context_propagation(
        self, mock_get_config, test_config
    ):
        """Test that context is properly propagated through routing layers"""
        mock_get_config.return_value = test_config

        agent = RoutingAgent()

        context = {
            "user_id": "test_user",
            "session_id": "session_123",
            "preferences": {"language": "en", "max_results": 5},
        }

        result = await agent.analyze_and_route("find videos", context)

        # Context should be preserved and potentially enhanced
        assert result["query"] == "find videos"
        assert isinstance(result["routing_decision"], dict)
        assert result["confidence"] > 0

    @patch("src.app.agents.routing_agent.get_config")
    def test_agent_registry_validation_integration(self, mock_get_config):
        """Test agent registry validation with different configurations"""
        # Test with minimal valid config
        mock_get_config.return_value = {"video_agent_url": "http://localhost:8002"}
        agent = RoutingAgent()
        assert agent.agent_registry["video_search"] == "http://localhost:8002"
        assert agent.agent_registry["text_search"] is None

        # Test with full config
        full_config = {
            "video_agent_url": "http://localhost:8002",
            "text_agent_url": "http://localhost:8003",
        }
        mock_get_config.return_value = full_config
        agent = RoutingAgent()
        assert agent.agent_registry["video_search"] == "http://localhost:8002"
        assert agent.agent_registry["text_search"] == "http://localhost:8003"

    @patch("src.app.agents.routing_agent.get_config")
    @pytest.mark.asyncio
    async def test_workflow_generation_consistency(self, mock_get_config, test_config):
        """Test that workflow generation is consistent across multiple calls"""
        mock_get_config.return_value = test_config

        agent = RoutingAgent()
        query = "Show me training videos"

        # Run same query multiple times
        results = []
        for _ in range(3):
            result = await agent.analyze_and_route(query)
            results.append(result)

        # Results should be consistent in structure
        first_result = results[0]
        for result in results[1:]:
            assert result["query"] == first_result["query"]
            assert result["workflow_type"] == first_result["workflow_type"]
            assert result["agents_to_call"] == first_result["agents_to_call"]
            assert len(result["execution_plan"]) == len(first_result["execution_plan"])


class TestRoutingAgentFastAPIIntegration:
    """Integration tests for FastAPI endpoints"""

    @pytest.fixture
    def test_client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)

    @pytest.fixture
    def mock_initialized_agent(self):
        """Mock initialized routing agent for testing"""
        with patch.object(routing_agent, "query_analyzer") as mock_analyzer:
            mock_analyzer.analyze.return_value = {
                "routing_decision": "video_search",
                "workflow_type": "raw_results",
                "execution_plan": [
                    {
                        "step": 1,
                        "agent": "video_search",
                        "action": "search",
                        "parameters": {},
                    }
                ],
                "agents_to_call": ["video_search"],
            }
            yield mock_analyzer

    def test_health_check_with_uninitialized_agent(self, test_client):
        """Test health check when agent not initialized"""
        # Use module-level patching
        with patch("src.app.agents.routing_agent.routing_agent", None):
            response = test_client.get("/health")
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]

    def test_health_check_with_initialized_agent(self, test_client):
        """Test health check when agent is initialized"""
        # Mock the global routing_agent to be initialized
        mock_agent = Mock()
        mock_agent.agent_registry = {
            "video_search": "http://localhost:8002",
            "text_search": None,
        }
        mock_agent.routing_config = Mock()
        mock_agent.routing_config.routing_mode = "tiered"
        mock_agent.router = Mock()
        mock_agent.router.strategies = {"fast_path": Mock(), "fallback": Mock()}

        with patch("src.app.agents.routing_agent.routing_agent", mock_agent):
            response = test_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["agent"] == "routing_agent"
            assert "available_downstream_agents" in data
            assert "routing_config" in data

    def test_analyze_query_endpoint(self, test_client):
        """Test query analysis endpoint"""
        # Mock the routing agent with an async analyze_and_route method
        mock_agent = Mock()
        mock_result = {
            "query": "Show me videos about python programming",
            "routing_decision": {"search_modality": "video"},
            "workflow_type": "raw_results",
            "agents_to_call": ["video_search"],
            "execution_plan": [{"step": 1, "agent": "video_search"}],
            "confidence": 0.8,
        }
        mock_agent.analyze_and_route = AsyncMock(return_value=mock_result)

        request_data = {
            "query": "Show me videos about python programming",
            "context": {"user_id": "test_user"},
        }

        with patch("src.app.agents.routing_agent.routing_agent", mock_agent):
            response = test_client.post("/analyze", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert data["query"] == request_data["query"]
            assert "routing_decision" in data
            assert "workflow_type" in data
            assert "agents_to_call" in data
            assert "execution_plan" in data
            assert "confidence" in data

    def test_analyze_query_missing_query(self, test_client):
        """Test analysis endpoint with missing query"""
        mock_agent = Mock()

        with patch("src.app.agents.routing_agent.routing_agent", mock_agent):
            response = test_client.post("/analyze", json={})

            assert response.status_code == 400
            assert "Query is required" in response.json()["detail"]

    def test_process_a2a_task(self, test_client):
        """Test A2A task processing endpoint"""
        # Mock agent with analyze_and_route method
        mock_agent = Mock()
        mock_result = {
            "query": "Find machine learning videos",
            "routing_decision": {"search_modality": "video"},
            "workflow_type": "raw_results",
            "agents_to_call": ["video_search"],
            "execution_plan": [{"step": 1, "agent": "video_search"}],
            "confidence": 0.85,
        }
        mock_agent.analyze_and_route = AsyncMock(return_value=mock_result)

        # Create A2A message
        message = A2AMessage(
            role="user",
            parts=[
                DataPart(data={"query": "Find machine learning videos", "top_k": 5})
            ],
        )

        task_data = {"id": "test_task_123", "messages": [message.to_dict()]}

        with patch("src.app.agents.routing_agent.routing_agent", mock_agent):
            response = test_client.post("/process", json=task_data)

            assert response.status_code == 200
            data = response.json()

            # Verify A2A response structure
            assert data["task_id"] == "test_task_123"
            assert "routing_decision" in data
            assert "agents_to_call" in data
            assert "workflow_type" in data
            assert "execution_plan" in data
            assert data["status"] == "completed"

    def test_process_a2a_task_with_text_part(self, test_client, mock_initialized_agent):
        """Test A2A task processing with TextPart"""
        message = A2AMessage(parts=[TextPart(text="Summarize recent AI developments")])

        task_data = {"id": "test_task_456", "messages": [message.to_dict()]}

        response = test_client.post("/process", json=task_data)

        assert response.status_code == 200
        data = response.json()

        assert data["task_id"] == "test_task_456"
        assert data["workflow_type"] == "summary"  # Should detect summary request

    def test_process_a2a_task_no_messages(self, test_client, mock_initialized_agent):
        """Test A2A task processing with no messages"""
        task_data = {"id": "test_task_empty", "messages": []}

        response = test_client.post("/process", json=task_data)

        assert response.status_code == 400
        assert "No messages in task" in response.json()["detail"]

    def test_process_a2a_task_no_query(self, test_client, mock_initialized_agent):
        """Test A2A task processing with message but no query"""
        message = A2AMessage(parts=[DataPart(data={"other_field": "value"})])
        task_data = {"id": "test_task_no_query", "messages": [message.to_dict()]}

        response = test_client.post("/process", json=task_data)

        assert response.status_code == 400
        assert "No query found in message" in response.json()["detail"]

    def test_get_routing_stats(self, test_client, mock_initialized_agent):
        """Test routing statistics endpoint"""
        response = test_client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        # Should return performance report structure
        assert "total_queries" in data
        assert "cache_size" in data
        assert "tier_performance" in data

    def test_uninitialized_agent_endpoints(self, test_client):
        """Test that endpoints fail properly when agent not initialized"""
        global routing_agent
        original_agent = routing_agent
        routing_agent = None

        try:
            # Test analyze endpoint
            response = test_client.post("/analyze", json={"query": "test"})
            assert response.status_code == 503

            # Test process endpoint
            message = A2AMessage(parts=[TextPart(text="test")])
            task_data = {"id": "test", "messages": [message.to_dict()]}
            response = test_client.post("/process", json=task_data)
            assert response.status_code == 503

            # Test stats endpoint
            response = test_client.get("/stats")
            assert response.status_code == 503

        finally:
            routing_agent = original_agent


class TestRoutingAgentErrorHandling:
    """Test error handling in integration scenarios"""

    @patch("src.app.agents.routing_agent.get_config")
    @pytest.mark.asyncio
    async def test_routing_failure_propagation(self, mock_get_config):
        """Test that routing failures are properly handled and propagated"""
        mock_get_config.return_value = {"video_agent_url": "http://localhost:8002"}

        agent = RoutingAgent()

        # Mock router to raise exception
        async def failing_route(query, context=None):
            raise RuntimeError("Routing service unavailable")

        agent.router.route = failing_route

        with pytest.raises(RuntimeError, match="Routing service unavailable"):
            await agent.analyze_and_route("test query")

    @patch("src.app.agents.routing_agent.get_config")
    def test_invalid_configuration_handling(self, mock_get_config):
        """Test handling of invalid configurations"""
        # Test with completely empty config
        mock_get_config.return_value = {}

        with pytest.raises(ValueError, match="video_agent_url not configured"):
            RoutingAgent()

        # Test with invalid URL format (this should still work as we don't validate URL format)
        mock_get_config.return_value = {"video_agent_url": "invalid-url"}
        agent = RoutingAgent()  # Should not raise exception
        assert agent.agent_registry["video_search"] == "invalid-url"
