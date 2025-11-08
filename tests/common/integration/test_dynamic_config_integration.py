"""
Integration tests for Dynamic DSPy Configuration.
Tests complete end-to-end flow with TextAnalysisAgent.
"""

from unittest.mock import MagicMock, patch

import pytest
from cogniverse_agents.text_analysis_agent import TextAnalysisAgent, app
from cogniverse_core.config.agent_config import DSPyModuleType
from fastapi.testclient import TestClient


class TestDynamicConfigIntegration:
    """Test dynamic DSPy configuration integration"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def fresh_agent(self, tmp_path):
        """Create fresh TextAnalysisAgent instance with clean config"""
        from cogniverse_core.config.utils import create_default_config_manager
        from fastapi import FastAPI

        # Create temporary ConfigManager instance
        temp_db = tmp_path / "test_config.db"
        config_manager = create_default_config_manager(db_path=temp_db)

        fresh_app = FastAPI()
        with patch("dspy.LM"):
            agent = TextAnalysisAgent(
                tenant_id="test_tenant", config_manager=config_manager
            )

            # Reset to default PREDICT module for consistent tests
            from cogniverse_core.config.agent_config import DSPyModuleType, ModuleConfig

            default_module_config = ModuleConfig(
                module_type=DSPyModuleType.PREDICT,
                signature="TextAnalysisSignature",
                max_retries=3,
                temperature=0.7,
            )
            agent.update_module_config(default_module_config)

            agent.setup_config_endpoints(
                fresh_app, config_manager=config_manager, tenant_id=agent.tenant_id
            )
        return agent, fresh_app

    def test_agent_initialization_with_dynamic_dspy(self, fresh_agent):
        """Test agent initializes with DynamicDSPyMixin"""
        agent, _ = fresh_agent
        assert hasattr(agent, "agent_config")
        assert hasattr(agent, "_signatures")
        assert hasattr(agent, "_dynamic_modules")
        assert hasattr(agent, "_optimizer")

        # Verify signature registered
        assert "text_analysis" in agent._signatures

        # Verify configuration (reset to PREDICT in fixture)
        assert agent.agent_config.module_config.module_type == DSPyModuleType.PREDICT

    def test_agent_initialization_with_config_api(self, fresh_agent):
        """Test agent has ConfigAPIMixin setup"""
        agent, _ = fresh_agent
        assert hasattr(agent, "setup_config_endpoints")
        assert hasattr(agent, "agent_config")

    def test_get_config_endpoint_returns_agent_config(self, fresh_agent):
        """Test GET /config returns agent configuration"""
        agent, fresh_app = fresh_agent
        client = TestClient(fresh_app)

        response = client.get("/config")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["config"]["agent_name"] == "text_analysis_agent"
        assert data["config"]["agent_version"] == "1.0.0"

    def test_get_module_config_returns_module_info(self, fresh_agent):
        """Test GET /config/module returns current module configuration"""
        agent, fresh_app = fresh_agent
        client = TestClient(fresh_app)

        response = client.get("/config/module")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["module_info"]["module_type"] == "predict"
        assert "text_analysis" in data["module_info"]["registered_signatures"]

    def test_update_module_config_at_runtime(self, fresh_agent):
        """Test POST /config/module updates module configuration at runtime"""
        agent, fresh_app = fresh_agent
        client = TestClient(fresh_app)

        # Initial module type
        assert agent.agent_config.module_config.module_type == DSPyModuleType.PREDICT

        # Update to ChainOfThought
        response = client.post(
            "/config/module",
            json={
                "module_type": "chain_of_thought",
                "signature": "TextAnalysisSignature",
                "max_retries": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "chain_of_thought" in data["message"]
        assert data["module_info"]["module_type"] == "chain_of_thought"

        # Verify agent config was updated
        assert (
            agent.agent_config.module_config.module_type == DSPyModuleType.CHAIN_OF_THOUGHT
        )
        assert agent.agent_config.module_config.max_retries == 5

    def test_update_module_config_clears_module_cache(self, fresh_agent):
        """Test updating module config clears cached modules"""
        agent, fresh_app = fresh_agent
        client = TestClient(fresh_app)

        # Create a module (will be cached)
        agent.create_module("text_analysis")
        assert len(agent._dynamic_modules) == 1

        # Update config
        response = client.post(
            "/config/module",
            json={
                "module_type": "react",
                "signature": "TextAnalysisSignature",
            },
        )

        assert response.status_code == 200
        # Verify cache was cleared
        assert len(agent._dynamic_modules) == 0

    def test_list_available_modules(self, fresh_agent):
        """Test GET /config/modules/available lists all module types"""
        agent, fresh_app = fresh_agent
        client = TestClient(fresh_app)

        response = client.get("/config/modules/available")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        modules = data["available_modules"]
        assert "predict" in modules
        assert "chain_of_thought" in modules
        assert "react" in modules
        assert "multi_chain_comparison" in modules
        assert "program_of_thought" in modules

    def test_list_available_optimizers(self, fresh_agent):
        """Test GET /config/optimizers/available lists all optimizer types"""
        agent, fresh_app = fresh_agent
        client = TestClient(fresh_app)

        response = client.get("/config/optimizers/available")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        optimizers = data["available_optimizers"]
        assert "bootstrap_few_shot" in optimizers
        assert "labeled_few_shot" in optimizers
        assert "copro" in optimizers
        assert "mipro_v2" in optimizers

    def test_a2a_endpoints_present(self, fresh_agent):
        """Test A2A endpoints are available"""
        agent, fresh_app = fresh_agent
        agent.setup_a2a_endpoints(fresh_app)
        client = TestClient(fresh_app)

        # Test /.well-known/agent-card.json
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "text_analysis_agent"
        assert data["version"] == "1.0.0"
        assert "capabilities" in data
        assert "text_analysis" in data["capabilities"]

    def test_health_check_endpoint(self, fresh_agent):
        """Test health check endpoint via HealthCheckMixin"""
        agent, fresh_app = fresh_agent
        agent.setup_health_endpoint(fresh_app)
        client = TestClient(fresh_app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_analyze_text_uses_dynamic_module(self, fresh_agent):
        """Test text analysis uses dynamically configured module"""
        agent, _ = fresh_agent

        # Mock DSPy module execution
        mock_result = MagicMock()
        mock_result.result = "This is a summary"
        mock_result.confidence = "0.9"

        with patch.object(agent, "get_or_create_module") as mock_get_module:
            mock_module = MagicMock()
            mock_module.return_value = mock_result
            mock_get_module.return_value = mock_module

            result = agent.analyze_text("Some text to analyze", "summary")

        assert result["result"] == "This is a summary"
        assert result["confidence"] == 0.9
        assert result["module_type"] == "predict"

    def test_analyze_text_endpoint(self, fresh_agent):
        """Test /analyze endpoint"""
        agent, fresh_app = fresh_agent

        # Add /analyze endpoint to fresh_app
        @fresh_app.post("/analyze")
        async def analyze_endpoint(text: str, tenant_id: str, analysis_type: str = "summary"):
            result = agent.analyze_text(text, analysis_type)
            return {"status": "success", "analysis": result}

        client = TestClient(fresh_app)

        with patch.object(agent, "analyze_text") as mock_analyze:
            mock_analyze.return_value = {
                "result": "Analysis result",
                "confidence": 0.85,
                "module_type": "predict",
                "analysis_type": "sentiment",
            }

            response = client.post(
                "/analyze?text=Test%20text&tenant_id=test_tenant&analysis_type=sentiment"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["analysis"]["result"] == "Analysis result"
        assert data["analysis"]["confidence"] == 0.85

    def test_end_to_end_configuration_update_workflow(self, fresh_agent):
        """
        Test complete end-to-end workflow:
        1. Check initial config (Predict)
        2. Update to ChainOfThought
        3. Verify cache cleared
        4. Update to ReAct
        5. Verify final config
        """
        agent, fresh_app = fresh_agent
        client = TestClient(fresh_app)

        # Step 1: Check initial config
        response = client.get("/config/module")
        assert response.status_code == 200
        assert response.json()["module_info"]["module_type"] == "predict"

        # Step 2: Update to ChainOfThought
        response = client.post(
            "/config/module",
            json={
                "module_type": "chain_of_thought",
                "signature": "TextAnalysisSignature",
            },
        )
        assert response.status_code == 200
        assert response.json()["module_info"]["module_type"] == "chain_of_thought"

        # Step 3: Verify agent updated
        assert (
            agent.agent_config.module_config.module_type == DSPyModuleType.CHAIN_OF_THOUGHT
        )

        # Step 4: Update to ReAct
        response = client.post(
            "/config/module",
            json={
                "module_type": "react",
                "signature": "TextAnalysisSignature",
            },
        )
        assert response.status_code == 200

        # Step 5: Verify final config
        response = client.get("/config/module")
        assert response.status_code == 200
        assert response.json()["module_info"]["module_type"] == "react"
        assert agent.agent_config.module_config.module_type == DSPyModuleType.REACT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
