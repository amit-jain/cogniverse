"""Unit tests for ConfigAPIMixin."""

import threading
from unittest.mock import patch

import dspy
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)
from cogniverse_foundation.config.api_mixin import ConfigAPIMixin
from cogniverse_foundation.config.manager import ConfigManager
from tests.utils.memory_store import InMemoryConfigStore


class ExampleSignature(dspy.Signature):
    """Test signature for module creation"""

    input_text = dspy.InputField()
    output_text = dspy.OutputField()


class ConfigurableAgent(DynamicDSPyMixin, ConfigAPIMixin):
    """Test agent class using both mixins"""

    def __init__(
        self,
        config: AgentConfig,
        app: FastAPI,
        config_manager,
        tenant_id: str = "test:unit",
    ):
        self.initialize_dynamic_dspy(config)
        self.setup_config_endpoints(app, config_manager, tenant_id)


class TestConfigAPIMixin:
    """Test ConfigAPIMixin functionality"""

    @pytest.fixture
    def agent_config(self):
        """Create test AgentConfig"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="ExampleSignature"
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
            module_type=DSPyModuleType.CHAIN_OF_THOUGHT, signature="ExampleSignature"
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
            agent = ConfigurableAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", ExampleSignature)
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
            "signature": "ExampleSignature",
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
            "signature": "ExampleSignature",
        }

        response = client.post("/config/module", json=request_data)

        assert response.status_code == 400
        assert "Invalid module type" in response.json()["detail"]

    def test_get_optimizer_config_endpoint(
        self, agent_config_with_optimizer, app, config_manager
    ):
        """Test GET /config/optimizer endpoint returns optimizer info"""
        with patch("dspy.LM"):
            agent = ConfigurableAgent(agent_config_with_optimizer, app, config_manager)
            agent.register_signature("test_sig", ExampleSignature)

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
            agent = ConfigurableAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", ExampleSignature)

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
            agent = ConfigurableAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", ExampleSignature)

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

        modules = data["available_modules"]
        assert modules.keys() == {"predict", "chain_of_thought", "react"}

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
            agent = ConfigurableAgent(agent_config, app, config_manager)
            agent.register_signature("test_sig", ExampleSignature)

            # Create initial module
            agent.create_module("test_sig")
            assert len(agent._dynamic_modules) == 1

        client = TestClient(app)

        # Update module config
        request_data = {
            "module_type": "chain_of_thought",
            "signature": "ExampleSignature",
        }

        response = client.post("/config/module", json=request_data)

        assert response.status_code == 200
        # Verify cache was cleared
        assert len(agent._dynamic_modules) == 0


class _FailingSetStore(InMemoryConfigStore):
    def set_config(self, *args, **kwargs):
        raise ConnectionError("configuration persistence unavailable")


class _TrackingSetStore(InMemoryConfigStore):
    def __init__(self):
        super().__init__()
        self.set_thread_ids = []

    def set_config(self, *args, **kwargs):
        self.set_thread_ids.append(threading.get_ident())
        return super().set_config(*args, **kwargs)


def _api_agent_config():
    return AgentConfig(
        agent_name="test_agent",
        agent_version="1.0.0",
        agent_description="Test agent",
        agent_url="http://localhost:8000",
        capabilities=["test"],
        skills=[],
        module_config=ModuleConfig(
            module_type=DSPyModuleType.PREDICT,
            signature="ExampleSignature",
        ),
        llm_model="gpt-4",
    )


def test_module_persistence_failure_restores_in_memory_config():
    app = FastAPI()
    manager = ConfigManager(store=_FailingSetStore())
    with patch("dspy.LM"):
        agent = ConfigurableAgent(_api_agent_config(), app, manager, tenant_id="acme")
        agent.register_signature("test_sig", ExampleSignature)
        agent.create_module("test_sig")
    cached_modules = dict(agent._dynamic_modules)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/config/module",
        json={
            "module_type": "chain_of_thought",
            "signature": "ExampleSignature",
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "configuration persistence unavailable"}
    assert agent.agent_config.module_config.module_type == DSPyModuleType.PREDICT
    assert agent._dynamic_modules == cached_modules


@pytest.mark.asyncio
async def test_module_persistence_runs_off_event_loop():
    app = FastAPI()
    store = _TrackingSetStore()
    manager = ConfigManager(store=store)
    with patch("dspy.LM"):
        agent = ConfigurableAgent(_api_agent_config(), app, manager, tenant_id="acme")
        agent.register_signature("test_sig", ExampleSignature)
    event_loop_thread = threading.get_ident()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/config/module",
            json={
                "module_type": "chain_of_thought",
                "signature": "ExampleSignature",
            },
        )

    assert response.status_code == 200
    assert len(store.set_thread_ids) == 1
    assert store.set_thread_ids[0] != event_loop_thread


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
