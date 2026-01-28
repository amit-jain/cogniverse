"""
Unit tests for AgentConfig and related schemas.
"""

import pytest

from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)


class TestDSPyModuleType:
    """Test DSPyModuleType enum"""

    def test_all_module_types_defined(self):
        """Test all expected module types are defined"""
        expected_types = {
            "predict",
            "chain_of_thought",
            "react",
            "multi_chain_comparison",
            "program_of_thought",
        }
        actual_types = {t.value for t in DSPyModuleType}
        assert actual_types == expected_types

    def test_module_type_from_string(self):
        """Test creating module type from string"""
        module_type = DSPyModuleType("predict")
        assert module_type == DSPyModuleType.PREDICT

    def test_invalid_module_type_raises_error(self):
        """Test invalid module type raises ValueError"""
        with pytest.raises(ValueError):
            DSPyModuleType("invalid_type")


class TestOptimizerType:
    """Test OptimizerType enum"""

    def test_all_optimizer_types_defined(self):
        """Test all expected optimizer types are defined"""
        expected_types = {
            "bootstrap_few_shot",
            "labeled_few_shot",
            "bootstrap_few_shot_with_random_search",
            "copro",
            "mipro_v2",
        }
        actual_types = {t.value for t in OptimizerType}
        assert actual_types == expected_types

    def test_optimizer_type_from_string(self):
        """Test creating optimizer type from string"""
        optimizer_type = OptimizerType("copro")
        assert optimizer_type == OptimizerType.COPRO

    def test_invalid_optimizer_type_raises_error(self):
        """Test invalid optimizer type raises ValueError"""
        with pytest.raises(ValueError):
            OptimizerType("invalid_optimizer")


class TestModuleConfig:
    """Test ModuleConfig dataclass"""

    def test_module_config_creation(self):
        """Test creating ModuleConfig with required fields"""
        config = ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="TestSignature"
        )

        assert config.module_type == DSPyModuleType.PREDICT
        assert config.signature == "TestSignature"
        assert config.max_retries == 3
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.custom_params == {}

    def test_module_config_with_custom_params(self):
        """Test ModuleConfig with custom parameters"""
        custom_params = {"param1": "value1", "param2": 42}
        config = ModuleConfig(
            module_type=DSPyModuleType.CHAIN_OF_THOUGHT,
            signature="TestSignature",
            max_retries=5,
            temperature=0.9,
            max_tokens=1000,
            custom_params=custom_params,
        )

        assert config.max_retries == 5
        assert config.temperature == 0.9
        assert config.max_tokens == 1000
        assert config.custom_params == custom_params


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass"""

    def test_optimizer_config_creation(self):
        """Test creating OptimizerConfig with required fields"""
        config = OptimizerConfig(optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT)

        assert config.optimizer_type == OptimizerType.BOOTSTRAP_FEW_SHOT
        assert config.max_bootstrapped_demos == 4
        assert config.max_labeled_demos == 16
        assert config.num_trials == 10
        assert config.metric is None
        assert config.teacher_settings == {}
        assert config.custom_params == {}

    def test_optimizer_config_with_custom_settings(self):
        """Test OptimizerConfig with custom settings"""
        teacher_settings = {"temperature": 0.5}
        custom_params = {"seed": 42}

        config = OptimizerConfig(
            optimizer_type=OptimizerType.MIPRO_V2,
            max_bootstrapped_demos=8,
            max_labeled_demos=32,
            num_trials=20,
            metric="accuracy",
            teacher_settings=teacher_settings,
            custom_params=custom_params,
        )

        assert config.max_bootstrapped_demos == 8
        assert config.max_labeled_demos == 32
        assert config.num_trials == 20
        assert config.metric == "accuracy"
        assert config.teacher_settings == teacher_settings
        assert config.custom_params == custom_params


class TestAgentConfig:
    """Test AgentConfig dataclass"""

    def test_agent_config_minimal(self):
        """Test creating AgentConfig with minimal required fields"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="TestSignature"
        )

        config = AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[],
            module_config=module_config,
        )

        assert config.agent_name == "test_agent"
        assert config.agent_version == "1.0.0"
        assert config.agent_description == "Test agent"
        assert config.agent_url == "http://localhost:8000"
        assert config.capabilities == ["test"]
        assert config.skills == []
        assert config.module_config == module_config
        assert config.optimizer_config is None
        assert config.llm_model == "gpt-4"
        assert config.thinking_enabled is True

    def test_agent_config_with_optimizer(self):
        """Test AgentConfig with optimizer configuration"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.CHAIN_OF_THOUGHT, signature="TestSignature"
        )
        optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT
        )

        config = AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[],
            module_config=module_config,
            optimizer_config=optimizer_config,
        )

        assert config.optimizer_config == optimizer_config

    def test_agent_config_to_dict(self):
        """Test AgentConfig serialization to dict"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="TestSignature"
        )

        config = AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[{"name": "skill1"}],
            module_config=module_config,
            llm_api_key="secret_key",
        )

        config_dict = config.to_dict()

        assert config_dict["agent_name"] == "test_agent"
        assert config_dict["agent_version"] == "1.0.0"
        assert config_dict["module_config"]["module_type"] == "predict"
        assert config_dict["module_config"]["signature"] == "TestSignature"
        assert config_dict["llm_api_key"] == "***"  # Should be masked
        assert config_dict["optimizer_config"] is None

    def test_agent_config_from_dict(self):
        """Test AgentConfig deserialization from dict"""
        config_dict = {
            "agent_name": "test_agent",
            "agent_version": "1.0.0",
            "agent_description": "Test agent",
            "agent_url": "http://localhost:8000",
            "capabilities": ["test"],
            "skills": [],
            "module_config": {
                "module_type": "predict",
                "signature": "TestSignature",
                "max_retries": 3,
                "temperature": 0.7,
                "max_tokens": None,
                "custom_params": {},
            },
        }

        config = AgentConfig.from_dict(config_dict)

        assert config.agent_name == "test_agent"
        assert config.agent_version == "1.0.0"
        assert config.module_config.module_type == DSPyModuleType.PREDICT
        assert config.module_config.signature == "TestSignature"

    def test_agent_config_roundtrip(self):
        """Test AgentConfig serialization/deserialization roundtrip"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.REACT,
            signature="TestSignature",
            max_retries=5,
        )
        optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.COPRO, max_bootstrapped_demos=8
        )

        original = AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[],
            module_config=module_config,
            optimizer_config=optimizer_config,
            llm_model="gpt-4-turbo",
            llm_temperature=0.5,
        )

        # Serialize and deserialize
        config_dict = original.to_dict()
        restored = AgentConfig.from_dict(config_dict)

        # Verify all fields match
        assert restored.agent_name == original.agent_name
        assert restored.agent_version == original.agent_version
        assert restored.module_config.module_type == original.module_config.module_type
        assert restored.module_config.max_retries == original.module_config.max_retries
        assert (
            restored.optimizer_config.optimizer_type
            == original.optimizer_config.optimizer_type
        )
        assert (
            restored.optimizer_config.max_bootstrapped_demos
            == original.optimizer_config.max_bootstrapped_demos
        )
        assert restored.llm_model == original.llm_model
        assert restored.llm_temperature == original.llm_temperature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
