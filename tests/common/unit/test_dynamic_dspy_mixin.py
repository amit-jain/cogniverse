"""
Unit tests for DynamicDSPyMixin.
"""

from unittest.mock import patch

import dspy
import pytest
from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
from cogniverse_core.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)


class TestSignature(dspy.Signature):
    """Test signature for module creation"""

    input_text = dspy.InputField()
    output_text = dspy.OutputField()


class TestAgent(DynamicDSPyMixin):
    """Test agent class using DynamicDSPyMixin"""

    def __init__(self, config: AgentConfig):
        self.initialize_dynamic_dspy(config)


class TestDynamicDSPyMixin:
    """Test DynamicDSPyMixin functionality"""

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

    def test_initialize_dynamic_dspy(self, agent_config):
        """Test initialization with DynamicDSPyMixin"""
        agent = TestAgent(agent_config)

        assert agent.agent_config == agent_config
        assert hasattr(agent, "_signatures")
        assert hasattr(agent, "_dynamic_modules")
        assert hasattr(agent, "_optimizer")
        assert agent._signatures == {}
        assert agent._dynamic_modules == {}
        assert agent._optimizer is None

    def test_configure_dspy_lm(self, agent_config):
        """Test DSPy LM configuration"""
        with patch("dspy.LM") as mock_lm:
            TestAgent(agent_config)

            # Verify LM was created with correct parameters
            mock_lm.assert_called_once()
            call_kwargs = mock_lm.call_args[1]
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_register_signature(self, agent_config):
        """Test signature registration"""
        agent = TestAgent(agent_config)

        agent.register_signature("test_sig", TestSignature)

        assert "test_sig" in agent._signatures
        assert agent._signatures["test_sig"] == TestSignature

    def test_create_module_predict(self, agent_config):
        """Test creating Predict module"""
        agent = TestAgent(agent_config)
        agent.register_signature("test_sig", TestSignature)

        module = agent.create_module("test_sig")

        assert isinstance(module, dspy.Predict)
        assert "test_sig" in agent._dynamic_modules
        assert agent._dynamic_modules["test_sig"] == module

    def test_create_module_chain_of_thought(self, agent_config):
        """Test creating ChainOfThought module"""
        agent_config.module_config.module_type = DSPyModuleType.CHAIN_OF_THOUGHT
        agent = TestAgent(agent_config)
        agent.register_signature("test_sig", TestSignature)

        module = agent.create_module("test_sig")

        assert isinstance(module, dspy.ChainOfThought)
        assert "test_sig" in agent._dynamic_modules

    def test_create_module_with_custom_config(self, agent_config):
        """Test creating module with custom ModuleConfig override"""
        agent = TestAgent(agent_config)
        agent.register_signature("test_sig", TestSignature)

        custom_config = ModuleConfig(
            module_type=DSPyModuleType.CHAIN_OF_THOUGHT,
            signature="TestSignature",
            max_retries=5,
        )

        module = agent.create_module("test_sig", module_config=custom_config)

        assert isinstance(module, dspy.ChainOfThought)

    def test_create_module_unregistered_signature_raises_error(self, agent_config):
        """Test creating module with unregistered signature raises error"""
        agent = TestAgent(agent_config)

        with pytest.raises(ValueError, match="Signature .* not registered"):
            agent.create_module("unregistered_sig")

    def test_get_or_create_module_creates_new(self, agent_config):
        """Test get_or_create_module creates new module if not cached"""
        agent = TestAgent(agent_config)
        agent.register_signature("test_sig", TestSignature)

        module = agent.get_or_create_module("test_sig")

        assert isinstance(module, dspy.Predict)
        assert "test_sig" in agent._dynamic_modules

    def test_get_or_create_module_returns_cached(self, agent_config):
        """Test get_or_create_module returns cached module"""
        agent = TestAgent(agent_config)
        agent.register_signature("test_sig", TestSignature)

        # Create first module
        module1 = agent.get_or_create_module("test_sig")

        # Get cached module
        module2 = agent.get_or_create_module("test_sig")

        # Should be same instance
        assert module1 is module2

    def test_create_optimizer(self, agent_config_with_optimizer):
        """Test creating optimizer"""
        agent = TestAgent(agent_config_with_optimizer)

        optimizer = agent.create_optimizer()

        assert optimizer is not None
        assert isinstance(optimizer, dspy.BootstrapFewShot)
        assert agent._optimizer == optimizer

    def test_create_optimizer_no_config_raises_error(self, agent_config):
        """Test creating optimizer without config raises error"""
        agent = TestAgent(agent_config)

        with pytest.raises(ValueError, match="No optimizer configuration"):
            agent.create_optimizer()

    def test_create_optimizer_with_custom_config(self, agent_config):
        """Test creating optimizer with custom OptimizerConfig override"""
        agent = TestAgent(agent_config)

        custom_optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.COPRO, max_bootstrapped_demos=8
        )

        optimizer = agent.create_optimizer(optimizer_config=custom_optimizer_config)

        assert optimizer is not None
        assert isinstance(optimizer, dspy.COPRO)

    def test_update_module_config(self, agent_config):
        """Test updating module configuration"""
        agent = TestAgent(agent_config)
        agent.register_signature("test_sig", TestSignature)

        # Create initial module
        agent.create_module("test_sig")
        assert DSPyModuleType.PREDICT == agent.agent_config.module_config.module_type

        # Update config
        new_config = ModuleConfig(
            module_type=DSPyModuleType.CHAIN_OF_THOUGHT, signature="TestSignature"
        )
        agent.update_module_config(new_config)

        # Verify config updated
        assert agent.agent_config.module_config == new_config

        # Verify cache cleared
        assert len(agent._dynamic_modules) == 0

    def test_update_optimizer_config(self, agent_config_with_optimizer):
        """Test updating optimizer configuration"""
        agent = TestAgent(agent_config_with_optimizer)

        # Create initial optimizer
        agent.create_optimizer()
        assert (
            OptimizerType.BOOTSTRAP_FEW_SHOT
            == agent.agent_config.optimizer_config.optimizer_type
        )

        # Update config
        new_config = OptimizerConfig(
            optimizer_type=OptimizerType.MIPRO_V2, max_bootstrapped_demos=10
        )
        agent.update_optimizer_config(new_config)

        # Verify config updated
        assert agent.agent_config.optimizer_config == new_config

        # Verify optimizer cleared
        assert agent._optimizer is None

    def test_get_module_info(self, agent_config):
        """Test getting module information"""
        agent = TestAgent(agent_config)
        agent.register_signature("sig1", TestSignature)
        agent.register_signature("sig2", TestSignature)
        agent.create_module("sig1")

        info = agent.get_module_info()

        assert info["module_type"] == "predict"
        assert "sig1" in info["registered_signatures"]
        assert "sig2" in info["registered_signatures"]
        assert "sig1" in info["cached_modules"]
        assert "sig2" not in info["cached_modules"]
        assert info["llm_model"] == "gpt-4"

    def test_get_optimizer_info_no_config(self, agent_config):
        """Test getting optimizer info without config"""
        agent = TestAgent(agent_config)

        info = agent.get_optimizer_info()

        assert info["optimizer_configured"] is False

    def test_get_optimizer_info_with_config(self, agent_config_with_optimizer):
        """Test getting optimizer info with config"""
        agent = TestAgent(agent_config_with_optimizer)

        info = agent.get_optimizer_info()

        assert info["optimizer_configured"] is True
        assert info["optimizer_type"] == "bootstrap_few_shot"
        assert info["max_bootstrapped_demos"] == 4
        assert info["max_labeled_demos"] == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
