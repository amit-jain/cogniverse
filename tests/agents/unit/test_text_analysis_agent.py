"""
Unit tests for TextAnalysisAgent
"""

from unittest.mock import MagicMock, patch

import pytest
from cogniverse_agents.text_analysis_agent import (
    TextAnalysisAgent,
    _agent_instances,
    app,
    get_agent,
    set_config_manager,
)
from fastapi.testclient import TestClient


class TestTextAnalysisAgent:
    """Test TextAnalysisAgent initialization and core functionality"""

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy")
    @patch("cogniverse_core.config.utils.get_config")
    def test_initialization(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
        config_manager_memory,
    ):
        """Test agent initializes correctly with tenant_id"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "ollama_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        # Initialize agent with config_manager
        agent = TextAnalysisAgent(tenant_id="test_tenant", config_manager=config_manager_memory)

        assert agent.tenant_id == "test_tenant"
        assert agent.config is not None
        assert agent.config.agent_name == "text_analysis_agent"
        assert "text_analysis" in agent.config.capabilities

        # Verify DSPy was initialized
        mock_initialize_dspy.assert_called_once()
        mock_register_signature.assert_called_once()

    def test_initialization_without_tenant_id_raises_error(self, config_manager_memory):
        """Test that initializing without tenant_id raises ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            TextAnalysisAgent(tenant_id="", config_manager=config_manager_memory)

        with pytest.raises(ValueError, match="tenant_id is required"):
            TextAnalysisAgent(tenant_id=None, config_manager=config_manager_memory)

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy")
    @patch("cogniverse_core.config.utils.get_config")
    def test_analyze_text(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
        config_manager_memory,
    ):
        """Test text analysis functionality"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "ollama_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        agent = TextAnalysisAgent(tenant_id="test_tenant", config_manager=config_manager_memory)

        # Mock the get_or_create_module method to return a mock module
        mock_module = MagicMock()
        mock_result = MagicMock()
        mock_result.result = "Test analysis result"
        mock_result.confidence = "0.95"
        mock_module.return_value = mock_result

        with patch.object(agent, "get_or_create_module", return_value=mock_module):
            # Test analyze_text
            result = agent.analyze_text("Test text", "summary")

            assert result["result"] == "Test analysis result"
            assert result["confidence"] == 0.95
            assert result["analysis_type"] == "summary"
            mock_module.assert_called_once_with(text="Test text", analysis_type="summary")

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy")
    @patch("cogniverse_core.config.utils.get_config")
    def test_get_agent_factory(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
        config_manager_memory,
    ):
        """Test get_agent factory function creates and caches instances"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "ollama_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        # Clear cache
        _agent_instances.clear()

        # Set config_manager for factory function
        set_config_manager(config_manager_memory)

        # First call creates new instance
        agent1 = get_agent("test_tenant_1")
        assert isinstance(agent1, TextAnalysisAgent)
        assert agent1.tenant_id == "test_tenant_1"

        # Second call returns cached instance
        agent2 = get_agent("test_tenant_1")
        assert agent2 is agent1

        # Different tenant creates new instance
        agent3 = get_agent("test_tenant_2")
        assert agent3 is not agent1
        assert agent3.tenant_id == "test_tenant_2"

    def test_get_agent_without_tenant_id_raises_error(self):
        """Test that get_agent without tenant_id raises ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            get_agent("")

        with pytest.raises(ValueError, match="tenant_id is required"):
            get_agent(None)


class TestTextAnalysisEndpoints:
    """Test FastAPI endpoints"""

    def setup_method(self):
        """Clear agent cache before each test"""
        _agent_instances.clear()

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy")
    @patch("cogniverse_core.config.utils.get_config")
    def test_analyze_endpoint(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
    ):
        """Test /analyze endpoint accepts tenant_id and returns analysis"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "ollama_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        client = TestClient(app)

        # Mock the agent's analyze_text method
        with patch.object(TextAnalysisAgent, "analyze_text") as mock_analyze:
            mock_analyze.return_value = {
                "result": "Test analysis result",
                "confidence": 0.95,
                "module_type": "predict",
                "analysis_type": "summary",
            }

            # Test endpoint
            response = client.post(
                "/analyze",
                params={
                    "text": "Test text to analyze",
                    "tenant_id": "test_tenant",
                    "analysis_type": "summary",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["analysis"]["result"] == "Test analysis result"
            assert data["analysis"]["confidence"] == 0.95

            # Verify analyze_text was called
            mock_analyze.assert_called_once_with("Test text to analyze", "summary")

    @patch("cogniverse_core.config.utils.get_config")
    def test_analyze_endpoint_without_tenant_id_fails(self, mock_get_config):
        """Test that /analyze endpoint requires tenant_id parameter"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "ollama_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        client = TestClient(app)

        # Test without tenant_id - should fail with 422 (missing required parameter)
        response = client.post(
            "/analyze",
            params={
                "text": "Test text to analyze",
                "analysis_type": "summary",
            },
        )

        assert response.status_code == 422  # FastAPI validation error
        error_detail = response.json()["detail"]
        assert any(
            error["loc"] == ["query", "tenant_id"] for error in error_detail
        ), "Should have validation error for missing tenant_id"

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy")
    @patch("cogniverse_core.config.utils.get_config")
    def test_analyze_endpoint_error_handling(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
    ):
        """Test that endpoint properly handles analysis errors"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "ollama_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        # Configure TestClient to not re-raise server exceptions
        client = TestClient(app, raise_server_exceptions=False)

        # Mock analyze_text to raise exception
        with patch.object(TextAnalysisAgent, "analyze_text") as mock_analyze:
            mock_analyze.side_effect = RuntimeError("Analysis failed")

            response = client.post(
                "/analyze",
                params={
                    "text": "Test text",
                    "tenant_id": "test_tenant",
                    "analysis_type": "summary",
                },
            )

            # Should return 500 error
            assert response.status_code == 500
