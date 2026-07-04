"""
Unit tests for DynamicDSPyMixin.
"""

from unittest.mock import MagicMock, patch

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
from cogniverse_foundation.config.unified_config import (
    LLMConfig,
    LLMEndpointConfig,
    SemanticRouterConfig,
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
        assert agent._signatures == {}
        assert agent._dynamic_modules == {}

    def test_configure_dspy_lm(self, agent_config):
        """The mixin attaches a litellm provider prefix before dspy.LM.

        ``config.llm_model`` is the in-cluster serving model, which the chart
        populates BARE. litellm needs an explicit provider, so the mixin
        prefixes it — ``gpt-4`` → ``openai/gpt-4``.
        """
        with patch("dspy.LM") as mock_lm:
            TestAgent(agent_config)

            mock_lm.assert_called_once()
            call_args = mock_lm.call_args
            assert call_args[0][0] == "openai/gpt-4"
            assert call_args[1]["api_base"] == "http://localhost:11434"

    def test_bare_ollama_model_gains_provider_prefix(self, agent_config):
        """A bare ollama tag like ``gemma3:4b`` must reach dspy.LM as
        ``openai/gemma3:4b`` — litellm rejects the bare id with "LLM
        Provider NOT provided", which is the live e2e agent-500 bug."""
        agent_config.llm_model = "gemma3:4b"
        agent_config.llm_api_key = None
        with patch("dspy.LM") as mock_lm:
            TestAgent(agent_config)

            assert mock_lm.call_args[0][0] == "openai/gemma3:4b"
            # A null key would make litellm's openai client refuse to dispatch
            # even against the no-auth in-cluster endpoint.
            assert mock_lm.call_args[1]["api_key"] == "placeholder-no-auth-needed"

    def test_stale_persisted_config_overridden_by_llm_config_primary(
        self, agent_config
    ):
        """A persisted AgentConfig with an empty llm_base_url (saved on an
        earlier deploy) must be overridden by config.json's live
        llm_config.primary, so litellm targets the in-cluster endpoint instead
        of silently falling back to the public OpenAI host."""
        agent_config.llm_base_url = None  # stale persisted endpoint
        agent_config.llm_model = "stale:model"
        primary = LLMEndpointConfig(
            model="openai/google/gemma-4-e4b-it",
            api_base="http://cogniverse-vllm-llm-student:8000/v1",
            api_key="placeholder-no-auth-needed",
        )
        sysconf = MagicMock()
        sysconf.get_llm_config.return_value = LLMConfig(
            primary=primary, teacher=primary
        )
        agent = TestAgent.__new__(TestAgent)
        agent.system_config = sysconf
        with patch("dspy.LM") as mock_lm:
            agent.initialize_dynamic_dspy(agent_config)

            assert mock_lm.call_args[0][0] == "openai/google/gemma-4-e4b-it"
            assert (
                mock_lm.call_args[1]["api_base"]
                == "http://cogniverse-vllm-llm-student:8000/v1"
            )

    def test_semantic_router_applied_when_enabled(self, agent_config):
        """When SystemConfig.semantic_router is enabled, the mixin builds the
        LM against the semantic router api_base with the resolved tier/task headers."""
        agent_config.agent_name = "query_enhancement_agent"
        primary = LLMEndpointConfig(
            model="openai/student", api_base="http://vllm:8101/v1"
        )
        sysconf = MagicMock()
        sysconf.get_llm_config.return_value = LLMConfig(
            primary=primary, teacher=primary
        )
        sysconf.get_semantic_router.return_value = SemanticRouterConfig(
            enabled=True,
            semantic_router_url="http://envoy:8801/v1",
            tenant_tiers={"acme:prod": "pro"},
            default_tier="free",
        )
        agent = TestAgent.__new__(TestAgent)
        agent.system_config = sysconf
        agent.tenant_id = "acme:prod"

        agent.initialize_dynamic_dspy(agent_config)

        assert agent._dspy_lm.kwargs["api_base"] == "http://envoy:8801/v1"
        assert agent._dspy_lm.kwargs["extra_headers"] == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_semantic_router_noop_when_disabled(self, agent_config):
        """Disabled router routing leaves the LM on the direct backend with no
        routing headers — the default path is unchanged."""
        primary = LLMEndpointConfig(
            model="openai/student", api_base="http://vllm:8101/v1"
        )
        sysconf = MagicMock()
        sysconf.get_llm_config.return_value = LLMConfig(
            primary=primary, teacher=primary
        )
        sysconf.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
        agent = TestAgent.__new__(TestAgent)
        agent.system_config = sysconf
        agent.tenant_id = "acme:prod"

        agent.initialize_dynamic_dspy(agent_config)

        assert agent._dspy_lm.kwargs["api_base"] == "http://vllm:8101/v1"
        assert "extra_headers" not in agent._dspy_lm.kwargs

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


class TestPrimaryKnobsCarryIntoAgentLM:
    """The per-agent LM must inherit the deployment's determinism and
    transport knobs from llm_config.primary, not silently rebuild with
    library defaults (no seed, 1 retry, 120s timeout, no headers)."""

    def test_seed_headers_timeout_retries_flow_from_primary(self, monkeypatch):
        from unittest.mock import MagicMock

        from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
        from cogniverse_foundation.config.agent_config import AgentConfig
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        captured = {}

        def _capture(endpoint):
            captured["endpoint"] = endpoint
            return MagicMock()

        monkeypatch.setattr(
            "cogniverse_core.common.dynamic_dspy_mixin.create_dspy_lm", _capture
        )

        class Host(DynamicDSPyMixin):
            pass

        host = Host()
        host.system_config = MagicMock()
        host.system_config.get_llm_config.return_value.primary = LLMEndpointConfig(
            model="openai/google/gemma-4-e4b-it",
            api_base="http://vllm:8000/v1",
            api_key="k",
            seed=42,
            extra_headers={"x-gateway-auth": "tok"},
            extra_body={"top_k": 5},
            request_timeout=300.0,
            num_retries=3,
        )
        host._route_through_semantic_router = lambda ep: ep

        config = AgentConfig(
            agent_name="text_analysis_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[],
            module_config=ModuleConfig(
                module_type=DSPyModuleType.PREDICT, signature="TestSignature"
            ),
            llm_model="stale",
            llm_base_url="http://stale:1/v1",
        )
        host._configure_dspy_lm(config)

        ep = captured["endpoint"]
        assert ep.model == "openai/google/gemma-4-e4b-it"
        assert ep.api_base == "http://vllm:8000/v1"
        assert ep.seed == 42
        assert ep.extra_headers == {"x-gateway-auth": "tok"}
        assert ep.extra_body == {"top_k": 5}
        assert ep.request_timeout == 300.0
        assert ep.num_retries == 3
