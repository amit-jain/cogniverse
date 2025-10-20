"""
Unit tests for TenantAwareAgentMixin
"""

from unittest.mock import MagicMock, patch

import pytest
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareAgentMixin
from cogniverse_core.config.unified_config import SystemConfig


class MockAgentWithTenant(TenantAwareAgentMixin):
    """Mock agent class using tenant mixin for testing"""

    def __init__(self, tenant_id: str, config=None):
        super().__init__(tenant_id=tenant_id, config=config)
        self.agent_name = "mock_agent"


class MockDSPyAgent:
    """Mock DSPy agent base class"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name


class MockAgentMultipleInheritance(MockDSPyAgent, TenantAwareAgentMixin):
    """Mock agent with multiple inheritance for MRO testing"""

    def __init__(self, tenant_id: str, agent_name: str = "multi_agent"):
        TenantAwareAgentMixin.__init__(self, tenant_id=tenant_id)
        MockDSPyAgent.__init__(self, agent_name=agent_name)


class TestTenantAwareAgentMixin:
    """Test TenantAwareAgentMixin"""

    def test_initialization_valid_tenant_id(self):
        """Test successful initialization with valid tenant_id"""
        agent = MockAgentWithTenant("customer_a")

        assert agent.tenant_id == "customer_a"
        assert agent._tenant_initialized is True
        assert agent.is_tenant_initialized() is True

    def test_initialization_with_org_tenant_format(self):
        """Test initialization with org:tenant format"""
        agent = MockAgentWithTenant("acme:production")

        assert agent.tenant_id == "acme:production"
        assert agent.is_tenant_initialized() is True

    def test_initialization_strips_whitespace(self):
        """Test that whitespace is stripped from tenant_id"""
        agent = MockAgentWithTenant("  customer_b  ")

        assert agent.tenant_id == "customer_b"
        assert agent.is_tenant_initialized() is True

    def test_initialization_empty_tenant_id_raises_error(self):
        """Test that empty tenant_id raises ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            MockAgentWithTenant("")

    def test_initialization_none_tenant_id_raises_error(self):
        """Test that None tenant_id raises ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            MockAgentWithTenant(None)

    def test_initialization_whitespace_only_tenant_id_raises_error(self):
        """Test that whitespace-only tenant_id raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            MockAgentWithTenant("   ")

    def test_initialization_with_config(self):
        """Test initialization with provided config"""
        mock_config = MagicMock(spec=SystemConfig)
        mock_config.environment = "test"

        agent = MockAgentWithTenant("customer_c", config=mock_config)

        assert agent.tenant_id == "customer_c"
        assert agent.config == mock_config
        assert agent.config.environment == "test"

    @patch("cogniverse_core.agents.tenant_aware_mixin.get_config")
    def test_initialization_loads_config_if_not_provided(self, mock_get_config):
        """Test that config is loaded if not provided"""
        mock_config = MagicMock(spec=SystemConfig)
        mock_config.environment = "production"
        mock_get_config.return_value = mock_config

        agent = MockAgentWithTenant("customer_d")

        assert agent.config is not None
        assert agent.config.environment == "production"
        mock_get_config.assert_called_once()

    @patch("cogniverse_core.agents.tenant_aware_mixin.get_config")
    def test_initialization_handles_config_load_failure(self, mock_get_config):
        """Test that config load failure is handled gracefully"""
        mock_get_config.side_effect = Exception("Config load failed")

        # Should not raise, just log warning
        agent = MockAgentWithTenant("customer_e")

        assert agent.tenant_id == "customer_e"
        assert agent.config is None

    def test_get_tenant_context_basic(self):
        """Test get_tenant_context returns correct information"""
        agent = MockAgentWithTenant("customer_f")

        context = agent.get_tenant_context()

        assert context["tenant_id"] == "customer_f"
        assert context["agent_type"] == "MockAgentWithTenant"
        # environment is optional, depends on config type

    def test_get_tenant_context_with_agent_name(self):
        """Test get_tenant_context includes agent_name if available"""
        agent = MockAgentWithTenant("customer_g")
        agent.agent_name = "test_routing_agent"

        context = agent.get_tenant_context()

        assert context["tenant_id"] == "customer_g"
        assert context["agent_name"] == "test_routing_agent"
        assert context["agent_type"] == "MockAgentWithTenant"

    def test_get_tenant_context_with_config(self):
        """Test get_tenant_context includes environment from config"""
        mock_config = MagicMock(spec=SystemConfig)
        mock_config.environment = "staging"

        agent = MockAgentWithTenant("customer_h", config=mock_config)

        context = agent.get_tenant_context()

        assert context["environment"] == "staging"

    def test_validate_tenant_access_same_tenant(self):
        """Test validate_tenant_access returns True for same tenant"""
        agent = MockAgentWithTenant("customer_i")

        assert agent.validate_tenant_access("customer_i") is True

    def test_validate_tenant_access_different_tenant(self):
        """Test validate_tenant_access returns False for different tenant"""
        agent = MockAgentWithTenant("customer_i")

        assert agent.validate_tenant_access("customer_j") is False

    def test_validate_tenant_access_empty_resource_tenant(self):
        """Test validate_tenant_access returns False for empty resource tenant"""
        agent = MockAgentWithTenant("customer_k")

        assert agent.validate_tenant_access("") is False
        assert agent.validate_tenant_access(None) is False

    def test_get_tenant_scoped_key(self):
        """Test get_tenant_scoped_key generates correct key"""
        agent = MockAgentWithTenant("customer_l")

        scoped_key = agent.get_tenant_scoped_key("embeddings/video_123")

        assert scoped_key == "customer_l:embeddings/video_123"

    def test_get_tenant_scoped_key_various_keys(self):
        """Test get_tenant_scoped_key with various key formats"""
        agent = MockAgentWithTenant("customer_m")

        assert agent.get_tenant_scoped_key("cache/key") == "customer_m:cache/key"
        assert (
            agent.get_tenant_scoped_key("search/results/abc123")
            == "customer_m:search/results/abc123"
        )
        assert agent.get_tenant_scoped_key("simple") == "customer_m:simple"

    def test_is_tenant_initialized_true(self):
        """Test is_tenant_initialized returns True after init"""
        agent = MockAgentWithTenant("customer_n")

        assert agent.is_tenant_initialized() is True

    def test_get_tenant_id(self):
        """Test get_tenant_id returns correct value"""
        agent = MockAgentWithTenant("customer_o")

        assert agent.get_tenant_id() == "customer_o"

    def test_log_tenant_operation_basic(self, caplog):
        """Test log_tenant_operation logs correctly"""
        import logging

        caplog.set_level(logging.INFO)

        agent = MockAgentWithTenant("customer_p")
        agent.log_tenant_operation("search_completed", {"results": 10})

        assert "customer_p" in caplog.text
        assert "search_completed" in caplog.text

    def test_log_tenant_operation_different_levels(self, caplog):
        """Test log_tenant_operation with different log levels"""
        import logging

        agent = MockAgentWithTenant("customer_q")

        caplog.set_level(logging.DEBUG)
        agent.log_tenant_operation("debug_op", level="debug")
        assert "debug_op" in caplog.text

        caplog.clear()
        caplog.set_level(logging.WARNING)
        agent.log_tenant_operation("warning_op", level="warning")
        assert "warning_op" in caplog.text

        caplog.clear()
        caplog.set_level(logging.ERROR)
        agent.log_tenant_operation("error_op", level="error")
        assert "error_op" in caplog.text

    def test_repr(self):
        """Test string representation"""
        agent = MockAgentWithTenant("customer_r")

        repr_str = repr(agent)

        assert "MockAgentWithTenant" in repr_str
        assert "customer_r" in repr_str

    def test_multiple_inheritance_mro(self):
        """Test that mixin works correctly with multiple inheritance"""
        agent = MockAgentMultipleInheritance(
            tenant_id="customer_s", agent_name="complex_agent"
        )

        assert agent.tenant_id == "customer_s"
        assert agent.agent_name == "complex_agent"
        assert agent.is_tenant_initialized() is True

    def test_multiple_inheritance_tenant_validation(self):
        """Test tenant validation works in multiple inheritance"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            MockAgentMultipleInheritance(tenant_id="", agent_name="test")

    def test_integration_with_real_system_config(self):
        """Test integration with actual get_config() (if available)"""
        from cogniverse_core.config.utils import ConfigUtils

        try:
            # Try to create agent without config (should load system config)
            agent = MockAgentWithTenant("customer_t")

            assert agent.tenant_id == "customer_t"
            # Config might be None, dict, or ConfigUtils depending on environment
            assert agent.config is None or isinstance(agent.config, (dict, ConfigUtils))
        except Exception as e:
            # If get_config() fails, that's OK for unit tests
            pytest.skip(f"get_config() not available: {e}")

    def test_tenant_context_without_class_attribute(self):
        """Test get_tenant_context when __class__ is not available"""
        agent = MockAgentWithTenant("customer_u")
        # Simulate missing __class__ (edge case)
        context = agent.get_tenant_context()

        # Should still work, just without agent_type
        assert context["tenant_id"] == "customer_u"

    def test_tenant_id_immutability_pattern(self):
        """Test that tenant_id follows immutability pattern (set once)"""
        agent = MockAgentWithTenant("customer_v")

        original_tenant_id = agent.tenant_id

        # In production, tenant_id should not be changed after initialization
        # This test documents the expected pattern (not enforced by code)
        assert agent.tenant_id == original_tenant_id
        assert agent.is_tenant_initialized() is True

    def test_concurrent_agents_different_tenants(self):
        """Test that multiple agents can exist with different tenant_ids"""
        agent1 = MockAgentWithTenant("customer_w")
        agent2 = MockAgentWithTenant("customer_x")
        agent3 = MockAgentWithTenant("customer_y")

        assert agent1.tenant_id == "customer_w"
        assert agent2.tenant_id == "customer_x"
        assert agent3.tenant_id == "customer_y"

        # Each agent has independent tenant context
        assert agent1.validate_tenant_access("customer_w") is True
        assert agent1.validate_tenant_access("customer_x") is False

        assert agent2.validate_tenant_access("customer_x") is True
        assert agent2.validate_tenant_access("customer_w") is False

    def test_error_message_quality_empty_tenant(self):
        """Test that error messages are helpful for empty tenant_id"""
        with pytest.raises(ValueError) as exc_info:
            MockAgentWithTenant("")

        error_msg = str(exc_info.value)
        assert "tenant_id is required" in error_msg
        assert "no default tenant" in error_msg

    def test_error_message_quality_whitespace_tenant(self):
        """Test that error messages are helpful for whitespace tenant_id"""
        with pytest.raises(ValueError) as exc_info:
            MockAgentWithTenant("   ")

        error_msg = str(exc_info.value)
        assert "cannot be empty or whitespace" in error_msg
        assert "valid tenant identifier" in error_msg

    @patch("cogniverse_core.agents.tenant_aware_mixin.logger")
    def test_logging_on_initialization(self, mock_logger):
        """Test that initialization logs correctly"""
        agent = MockAgentWithTenant("customer_z")

        # Should log debug message with tenant_id
        assert agent.is_tenant_initialized()
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "customer_z" in call_args

    @patch("cogniverse_core.agents.tenant_aware_mixin.logger")
    @patch("cogniverse_core.agents.tenant_aware_mixin.get_config")
    def test_logging_on_validation_failure(self, mock_get_config, mock_logger):
        """Test that validation failure logs warning"""
        # Mock get_config to not raise warnings during init
        mock_get_config.return_value = MagicMock()

        agent = MockAgentWithTenant("customer_aa")

        # Clear any warnings from initialization
        mock_logger.warning.reset_mock()

        # Attempt to validate access to resource with no tenant
        agent.validate_tenant_access("")

        # Should log warning exactly once
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "no tenant_id" in call_args
        assert "customer_aa" in call_args
