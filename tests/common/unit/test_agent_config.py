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

    def test_invalid_module_type_raises_error(self):
        """Test invalid module type raises ValueError"""
        with pytest.raises(ValueError):
            DSPyModuleType("invalid_type")


class TestOptimizerType:
    """Test OptimizerType enum"""

    def test_invalid_optimizer_type_raises_error(self):
        """Test invalid optimizer type raises ValueError"""
        with pytest.raises(ValueError):
            OptimizerType("invalid_optimizer")


class TestAgentConfig:
    """Test AgentConfig dataclass"""

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
