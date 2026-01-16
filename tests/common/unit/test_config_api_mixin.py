"""
Unit tests for ConfigAPIMixin.
"""

from unittest.mock import patch

import dspy
import pytest
from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)
from cogniverse_foundation.config.api_mixin import ConfigAPIMixin
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestSignature(dspy.Signature):
    """Test signature for module creation"""

    input_text = dspy.InputField()
    output_text = dspy.OutputField()


class TestAgent(DynamicDSPyMixin, ConfigAPIMixin):
    """Test agent class using both mixins"""

    def __init__(self, config: AgentConfig, app: FastAPI, config_manager, tenant_id: str = "default"):
        self.initialize_dynamic_dspy(config)
        self.setup_config_endpoints(app, config_manager, tenant_id)


class TestConfigAPIMixin:
    """Test ConfigAPIMixin functionality"""

    @pytest.fixture
    def agent_config(self):
        """Create test AgentConfig"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="TestSignature"
        )

        return AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[],
            module_config=module_config,
            llm_model="gpt-4",
            llm_base_url="http://localhost:11434",
        )

    @pytest.fixture
    def agent_config_with_optimizer(self):
        """Create test AgentConfig with optimizer"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.CHAIN_OF_THOUGHT, signature="TestSignature"
        )
        optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT,
            max_bootstrapped_demos=4,
        )

        return AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[],
            module_config=module_config,
            optimizer_config=optimizer_config,
            llm_model="gpt-4",
        )

    @pytest.fixture
    def app(self):
        """Create FastAPI app"""
        return FastAPI()

    @pytest.fixture
    def config_manager(self, config_manager_memory):
        """Create ConfigManager for testing (uses in-memory store)"""
        return config_manager_memory

    @pytest.fixture
    def client(self, agent_config, app, config_manager):
        """Create test client with agent"""
        with patch("dspy.LM"):
            agent = TestAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", TestSignature)
        return TestClient(app)

    def test_get_config_endpoint(self, client):
        """Test GET /config endpoint returns agent configuration"""
        response = client.get("/config")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "config" in data
        assert data["config"]["agent_name"] == "test_agent"
        assert data["config"]["agent_version"] == "1.0.0"

    def test_get_module_config_endpoint(self, client):
        """Test GET /config/module endpoint returns module info"""
        response = client.get("/config/module")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "module_info" in data
        assert data["module_info"]["module_type"] == "predict"
        assert data["module_info"]["llm_model"] == "gpt-4"

    def test_post_module_config_valid(self, client):
        """Test POST /config/module with valid data updates configuration"""
        request_data = {
            "module_type": "chain_of_thought",
            "signature": "TestSignature",
            "max_retries": 5,
            "temperature": 0.9,
        }

        response = client.post("/config/module", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "chain_of_thought" in data["message"]
        assert data["module_info"]["module_type"] == "chain_of_thought"

    def test_post_module_config_invalid_module_type(self, client):
        """Test POST /config/module with invalid module type returns 400"""
        request_data = {
            "module_type": "invalid_module",
            "signature": "TestSignature",
        }

        response = client.post("/config/module", json=request_data)

        assert response.status_code == 400
        assert "Invalid module type" in response.json()["detail"]

    def test_get_optimizer_config_endpoint(self, agent_config_with_optimizer, app, config_manager):
        """Test GET /config/optimizer endpoint returns optimizer info"""
        with patch("dspy.LM"):
            agent = TestAgent(agent_config_with_optimizer, app, config_manager)
            agent.register_signature("test_sig", TestSignature)

        client = TestClient(app)
        response = client.get("/config/optimizer")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "optimizer_info" in data
        assert data["optimizer_info"]["optimizer_configured"] is True
        assert data["optimizer_info"]["optimizer_type"] == "bootstrap_few_shot"

    def test_get_optimizer_config_no_optimizer(self, client):
        """Test GET /config/optimizer without optimizer configured"""
        response = client.get("/config/optimizer")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["optimizer_info"]["optimizer_configured"] is False

    def test_post_optimizer_config_valid(self, client):
        """Test POST /config/optimizer with valid data updates configuration"""
        request_data = {
            "optimizer_type": "copro",
            "max_bootstrapped_demos": 8,
            "max_labeled_demos": 32,
        }

        response = client.post("/config/optimizer", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "copro" in data["message"]
        assert data["optimizer_info"]["optimizer_type"] == "copro"

    def test_post_optimizer_config_invalid_optimizer_type(self, client):
        """Test POST /config/optimizer with invalid optimizer type returns 400"""
        request_data = {
            "optimizer_type": "invalid_optimizer",
        }

        response = client.post("/config/optimizer", json=request_data)

        assert response.status_code == 400
        assert "Invalid optimizer type" in response.json()["detail"]

    def test_post_llm_config_update_model(self, agent_config, app, config_manager):
        """Test POST /config/llm updates LLM model"""
        with patch("dspy.LM"):
            agent = TestAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", TestSignature)

        client = TestClient(app)

        request_data = {
            "llm_model": "gpt-4-turbo",
            "llm_temperature": 0.5,
        }

        # Mock _configure_dspy_lm to avoid thread issues
        with patch.object(agent, "_configure_dspy_lm"):
            response = client.post("/config/llm", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["llm_config"]["model"] == "gpt-4-turbo"
        assert data["llm_config"]["temperature"] == 0.5

    def test_post_llm_config_update_base_url(self, agent_config, app, config_manager):
        """Test POST /config/llm updates base URL"""
        with patch("dspy.LM"):
            agent = TestAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", TestSignature)

        client = TestClient(app)

        request_data = {
            "llm_base_url": "http://new-url:8000",
        }

        # Mock _configure_dspy_lm to avoid thread issues
        with patch.object(agent, "_configure_dspy_lm"):
            response = client.post("/config/llm", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["llm_config"]["base_url"] == "http://new-url:8000"

    def test_get_available_modules(self, client):
        """Test GET /config/modules/available lists all module types"""
        response = client.get("/config/modules/available")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "available_modules" in data

        # Verify all module types are present
        modules = data["available_modules"]
        assert "predict" in modules
        assert "chain_of_thought" in modules
        assert "react" in modules
        assert "multi_chain_comparison" in modules
        assert "program_of_thought" in modules

    def test_get_available_optimizers(self, client):
        """Test GET /config/optimizers/available lists all optimizer types"""
        response = client.get("/config/optimizers/available")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "available_optimizers" in data

        # Verify all optimizer types are present
        optimizers = data["available_optimizers"]
        assert "bootstrap_few_shot" in optimizers
        assert "labeled_few_shot" in optimizers
        assert "bootstrap_few_shot_with_random_search" in optimizers
        assert "copro" in optimizers
        assert "mipro_v2" in optimizers

    def test_post_module_config_clears_cache(self, agent_config, app, config_manager):
        """Test updating module config clears cached modules"""
        with patch("dspy.LM"):
            agent = TestAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", TestSignature)

            # Create initial module
            agent.create_module("test_sig")
            assert len(agent._dynamic_modules) == 1

        client = TestClient(app)

        # Update module config
        request_data = {
            "module_type": "chain_of_thought",
            "signature": "TestSignature",
        }

        response = client.post("/config/module", json=request_data)

        assert response.status_code == 200
        # Verify cache was cleared
        assert len(agent._dynamic_modules) == 0

    def test_post_optimizer_config_clears_optimizer(self, agent_config, app, config_manager):
        """Test updating optimizer config clears cached optimizer"""
        with patch("dspy.LM"):
            agent = TestAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", TestSignature)

            # Create initial optimizer
            optimizer_config = OptimizerConfig(
                optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT
            )
            agent.agent_config.optimizer_config = optimizer_config
            agent.create_optimizer()
            assert agent._optimizer is not None

        client = TestClient(app)

        # Update optimizer config
        request_data = {
            "optimizer_type": "copro",
        }

        response = client.post("/config/optimizer", json=request_data)

        assert response.status_code == 200
        # Verify optimizer was cleared
        assert agent._optimizer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
