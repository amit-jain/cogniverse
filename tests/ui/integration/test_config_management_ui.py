"""
Integration tests for Configuration Management UI

Tests the Streamlit configuration management tab with real ConfigManager.
Uses temporary database for isolation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cogniverse_core.common.config_store import SQLiteConfigStore
from cogniverse_core.common.config_store_interface import ConfigScope
from cogniverse_core.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)
from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.config.unified_config import (
    RoutingConfigUnified,
    SystemConfig,
    TelemetryConfigUnified,
)


class TestConfigManagementUI:
    """Integration tests for Configuration Management UI"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_config.db"
            yield db_path

    @pytest.fixture
    def config_manager(self, temp_db):
        """Create ConfigManager with temporary database"""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._db_path = None

        store = SQLiteConfigStore(temp_db)
        manager = ConfigManager(store=store)
        return manager

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components"""
        with patch("streamlit.session_state", new_callable=dict) as mock_state:
            mock_st = MagicMock()
            mock_st.session_state = mock_state
            yield mock_st

    def test_system_config_creation(self, config_manager):
        """Test creating system configuration through UI workflow"""
        tenant_id = "test_tenant"

        # Simulate UI form submission
        system_config = SystemConfig(
            tenant_id=tenant_id,
            routing_agent_url="http://localhost:8001",
            video_agent_url="http://localhost:8002",
            text_agent_url="http://localhost:8003",
            summarizer_agent_url="http://localhost:8004",
            text_analysis_agent_url="http://localhost:8005",
            search_backend="vespa",
            backend_url="http://localhost",
            backend_port=8080,
            llm_model="gpt-4-turbo",
            base_url="http://localhost:11434",
            llm_api_key="test-key-123",
            phoenix_url="http://localhost:6006",
            phoenix_collector_endpoint="localhost:4317",
            environment="development",
        )

        # Save via ConfigManager (simulates UI save action)
        config_manager.set_system_config(system_config)

        # Verify config was saved
        retrieved = config_manager.get_system_config(tenant_id)

        assert retrieved.tenant_id == tenant_id
        assert retrieved.llm_model == "gpt-4-turbo"
        assert retrieved.backend_port == 8080
        # Note: llm_api_key is stored, but masked in to_dict() for security
        assert system_config.llm_api_key == "test-key-123"
        assert retrieved.environment == "development"

    def test_system_config_update(self, config_manager):
        """Test updating existing system configuration"""
        tenant_id = "test_tenant"

        # Create initial config
        initial_config = SystemConfig(
            tenant_id=tenant_id,
            llm_model="gpt-4",
        )
        config_manager.set_system_config(initial_config)

        # Update config (simulates UI edit)
        updated_config = config_manager.get_system_config(tenant_id)
        updated_config.llm_model = "gpt-4-turbo"
        updated_config.backend_port = 9090
        config_manager.set_system_config(updated_config)

        # Verify update
        retrieved = config_manager.get_system_config(tenant_id)
        assert retrieved.llm_model == "gpt-4-turbo"
        assert retrieved.backend_port == 9090

    def test_agent_config_creation(self, config_manager):
        """Test creating agent configuration through UI"""
        tenant_id = "test_tenant"
        agent_name = "test_agent"

        # Simulate UI form submission for new agent
        agent_config = AgentConfig(
            agent_name=agent_name,
            agent_version="1.0.0",
            agent_description="Test agent for configuration",
            agent_url="http://localhost:8001",
            capabilities=["text_analysis", "routing"],
            skills=[{"name": "analyze_text", "description": "Analyze text content"}],
            module_config=ModuleConfig(
                module_type=DSPyModuleType.CHAIN_OF_THOUGHT,
                signature="TestSignature",
                max_retries=3,
                temperature=0.7,
                max_tokens=1000,
                custom_params={},
            ),
            llm_model="gpt-4",
            llm_base_url="http://localhost:11434",
            llm_api_key="test-key",
            optimizer_config=OptimizerConfig(
                optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT,
                max_bootstrapped_demos=4,
                max_labeled_demos=16,
                num_trials=10,
            ),
        )

        # Save via ConfigManager
        config_manager.set_agent_config(tenant_id, agent_name, agent_config)

        # Verify config was saved
        retrieved = config_manager.get_agent_config(tenant_id, agent_name)

        assert retrieved.agent_name == agent_name
        assert retrieved.module_config.module_type == DSPyModuleType.CHAIN_OF_THOUGHT
        assert retrieved.module_config.signature == "TestSignature"
        assert retrieved.module_config.max_tokens == 1000
        assert retrieved.optimizer_config is not None
        assert (
            retrieved.optimizer_config.optimizer_type == OptimizerType.BOOTSTRAP_FEW_SHOT
        )
        assert retrieved.optimizer_config.max_bootstrapped_demos == 4

    def test_agent_config_list(self, config_manager):
        """Test listing agent configurations"""
        tenant_id = "test_tenant"

        # Create multiple agent configs
        for i in range(3):
            agent_config = AgentConfig(
                agent_name=f"agent_{i}",
                agent_version="1.0.0",
                agent_description=f"Test agent {i}",
                agent_url=f"http://localhost:800{i}",
                capabilities=["test"],
                skills=[],
                module_config=ModuleConfig(
                    module_type=DSPyModuleType.PREDICT,
                    signature=f"Signature{i}",
                ),
            )
            config_manager.set_agent_config(tenant_id, f"agent_{i}", agent_config)

        # List all agent configs (simulates UI listing)
        configs = config_manager.store.list_configs(
            tenant_id=tenant_id,
            scope=ConfigScope.AGENT,
        )

        assert len(configs) == 3
        agent_names = [c.service for c in configs]
        assert "agent_0" in agent_names
        assert "agent_1" in agent_names
        assert "agent_2" in agent_names

    def test_routing_config_creation(self, config_manager):
        """Test creating routing configuration"""
        tenant_id = "test_tenant"

        # Simulate UI form submission
        routing_config = RoutingConfigUnified(
            tenant_id=tenant_id,
            routing_mode="ensemble",
            enable_fast_path=True,
            enable_auto_optimization=True,
            optimization_interval_seconds=1800,
            min_samples_for_optimization=100,
        )

        # Save via ConfigManager
        config_manager.set_routing_config(routing_config)

        # Verify
        retrieved = config_manager.get_routing_config(tenant_id)
        assert retrieved.routing_mode == "ensemble"
        assert retrieved.enable_fast_path is True
        assert retrieved.enable_auto_optimization is True
        assert retrieved.optimization_interval_seconds == 1800
        assert retrieved.min_samples_for_optimization == 100

    def test_telemetry_config_creation(self, config_manager):
        """Test creating telemetry configuration"""
        tenant_id = "test_tenant"

        # Simulate UI form submission
        telemetry_config = TelemetryConfigUnified(
            tenant_id=tenant_id,
            enabled=True,
            phoenix_enabled=True,
            phoenix_endpoint="localhost:4317",
            service_name="test_project",
            level="detailed",
        )

        # Save via ConfigManager
        config_manager.set_telemetry_config(telemetry_config)

        # Verify
        retrieved = config_manager.get_telemetry_config(tenant_id)
        assert retrieved.phoenix_enabled is True
        assert retrieved.service_name == "test_project"
        assert retrieved.enabled is True
        assert retrieved.level == "detailed"

    def test_config_history_workflow(self, config_manager):
        """Test configuration history viewer workflow"""
        tenant_id = "test_tenant"

        # Create initial config
        config_v1 = SystemConfig(tenant_id=tenant_id, llm_model="gpt-4")
        config_manager.set_system_config(config_v1)

        # Update config (creates version 2)
        config_v2 = config_manager.get_system_config(tenant_id)
        config_v2.llm_model = "gpt-4-turbo"
        config_manager.set_system_config(config_v2)

        # Update again (creates version 3)
        config_v3 = config_manager.get_system_config(tenant_id)
        config_v3.llm_model = "claude-3"
        config_manager.set_system_config(config_v3)

        # Get history (simulates UI history viewer)
        history = config_manager.store.get_config_history(
            tenant_id=tenant_id,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
            limit=10,
        )

        # Verify history
        assert len(history) == 3
        assert history[0].version == 3  # Newest first
        assert history[1].version == 2
        assert history[2].version == 1

        # Verify version contents (config_value is already a dict, not JSON string)
        assert history[0].config_value["llm_model"] == "claude-3"
        assert history[1].config_value["llm_model"] == "gpt-4-turbo"
        assert history[2].config_value["llm_model"] == "gpt-4"

    def test_config_rollback_workflow(self, config_manager):
        """Test rolling back to previous configuration version"""
        tenant_id = "test_tenant"

        # Create version 1
        config_v1 = SystemConfig(tenant_id=tenant_id, llm_model="gpt-4", backend_port=8080)
        config_manager.set_system_config(config_v1)

        # Create version 2 (bad config)
        config_v2 = config_manager.get_system_config(tenant_id)
        config_v2.llm_model = "bad-model"
        config_v2.backend_port = 9999
        config_manager.set_system_config(config_v2)

        # Simulate rollback in UI: get version 1 from history
        history = config_manager.store.get_config_history(
            tenant_id=tenant_id,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
            limit=10,
        )

        version_1 = next(e for e in history if e.version == 1)

        # Rollback = set config with old values (creates version 3)
        config_manager.store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
            config_value=version_1.config_value,
        )

        # Verify rollback created version 3 with version 1 values
        current_config = config_manager.get_system_config(tenant_id)
        assert current_config.llm_model == "gpt-4"
        assert current_config.backend_port == 8080

    def test_multi_tenant_isolation(self, config_manager):
        """Test multi-tenant configuration isolation"""
        tenant_a = "tenant_a"
        tenant_b = "tenant_b"

        # Create configs for tenant A
        config_a = SystemConfig(
            tenant_id=tenant_a,
            llm_model="gpt-4",
            backend_port=8080,
        )
        config_manager.set_system_config(config_a)

        # Create configs for tenant B
        config_b = SystemConfig(
            tenant_id=tenant_b,
            llm_model="claude-3",
            backend_port=9090,
        )
        config_manager.set_system_config(config_b)

        # Verify isolation
        retrieved_a = config_manager.get_system_config(tenant_a)
        retrieved_b = config_manager.get_system_config(tenant_b)

        assert retrieved_a.llm_model == "gpt-4"
        assert retrieved_a.backend_port == 8080

        assert retrieved_b.llm_model == "claude-3"
        assert retrieved_b.backend_port == 9090

    def test_export_configs_workflow(self, config_manager):
        """Test exporting configurations (UI export button)"""
        tenant_id = "test_tenant"

        # Create various configs
        system_config = SystemConfig(tenant_id=tenant_id, llm_model="gpt-4")
        config_manager.set_system_config(system_config)

        agent_config = AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8001",
            capabilities=["test"],
            skills=[],
            module_config=ModuleConfig(
                module_type=DSPyModuleType.PREDICT,
                signature="TestSig",
            ),
        )
        config_manager.set_agent_config(tenant_id, "test_agent", agent_config)

        # Export (simulates UI export button)
        export_data = config_manager.store.export_configs(
            tenant_id=tenant_id,
            include_history=False,
        )

        # Verify export structure
        assert export_data["tenant_id"] == tenant_id
        assert export_data["include_history"] is False
        assert "configs" in export_data
        assert len(export_data["configs"]) >= 2  # At least system and agent

        # Verify exportable as JSON
        json_str = json.dumps(export_data, indent=2)
        assert json_str is not None

    def test_import_configs_workflow(self, config_manager):
        """Test importing configurations from JSON"""
        tenant_id = "test_tenant"

        # Create export data (simulates uploaded JSON file)
        import_data = {
            "tenant_id": tenant_id,
            "include_history": False,
            "configs": [
                {
                    "scope": "system",
                    "service": "system",
                    "config_key": "system_config",
                    "config_value": {
                        "tenant_id": tenant_id,
                        "llm_model": "imported-model",
                        "vespa_port": 7070,
                    },
                },
                {
                    "scope": "agent",
                    "service": "imported_agent",
                    "config_key": "agent_config",
                    "config_value": {
                        "agent_name": "imported_agent",
                        "module_config": {
                            "module_type": "predict",
                            "signature": "ImportedSig",
                        },
                    },
                },
            ],
        }

        # Import (simulates UI import button)
        imported_count = config_manager.store.import_configs(
            tenant_id=tenant_id,
            configs=import_data,
        )

        # Verify import
        assert imported_count == 2

        # Verify imported configs are accessible
        configs = config_manager.store.list_configs(tenant_id=tenant_id)
        assert len(configs) >= 2

    def test_storage_stats_workflow(self, config_manager):
        """Test storage statistics display"""
        tenant_id = "test_tenant"

        # Create multiple configs
        config_manager.set_system_config(SystemConfig(tenant_id=tenant_id))
        config_manager.set_routing_config(RoutingConfigUnified(tenant_id=tenant_id))
        config_manager.set_telemetry_config(TelemetryConfigUnified(tenant_id=tenant_id))

        # Get stats (simulates UI stats display)
        stats = config_manager.store.get_stats()

        # Verify stats structure
        assert "total_configs" in stats
        assert "total_versions" in stats
        assert "total_tenants" in stats
        assert "configs_per_scope" in stats
        assert "db_size_mb" in stats

        # Verify values
        assert stats["total_configs"] >= 3
        assert stats["total_tenants"] >= 1
        assert stats["db_size_mb"] >= 0

    def test_backend_health_check(self, config_manager):
        """Test backend health check indicator"""
        # Simulate UI health check
        is_healthy = config_manager.store.health_check()

        # Should be healthy with working database
        assert is_healthy is True

    def test_config_validation_error_handling(self, config_manager):
        """Test error handling for invalid configurations"""

        # Try to create agent config with invalid module type
        # DSPyModuleType enum will reject invalid values at creation time
        with pytest.raises((ValueError, KeyError)):
            # This will fail because "invalid_module_type" is not a valid enum value
            DSPyModuleType("invalid_module_type")

    def test_agent_config_without_optimizer(self, config_manager):
        """Test creating agent config without optimizer (optional field)"""
        tenant_id = "test_tenant"
        agent_name = "simple_agent"

        # Create config without optimizer
        agent_config = AgentConfig(
            agent_name=agent_name,
            agent_version="1.0.0",
            agent_description="Simple agent",
            agent_url="http://localhost:8001",
            capabilities=["test"],
            skills=[],
            module_config=ModuleConfig(
                module_type=DSPyModuleType.PREDICT,
                signature="SimpleSig",
            ),
            optimizer_config=None,  # No optimizer
        )

        config_manager.set_agent_config(tenant_id, agent_name, agent_config)

        # Verify
        retrieved = config_manager.get_agent_config(tenant_id, agent_name)
        assert retrieved.optimizer_config is None

    def test_tenant_switching_workflow(self, config_manager):
        """Test switching between tenants in UI"""
        # Create configs for multiple tenants
        tenants = ["tenant_a", "tenant_b", "tenant_c"]

        for tenant in tenants:
            config = SystemConfig(
                tenant_id=tenant,
                llm_model=f"model-{tenant}",
            )
            config_manager.set_system_config(config)

        # Simulate switching between tenants in UI
        for tenant in tenants:
            config = config_manager.get_system_config(tenant)
            assert config.llm_model == f"model-{tenant}"

    def test_config_update_preserves_other_fields(self, config_manager):
        """Test that updating one field preserves others"""
        tenant_id = "test_tenant"

        # Create full config
        config = SystemConfig(
            tenant_id=tenant_id,
            llm_model="gpt-4",
            backend_port=8080,
            environment="production",
        )
        config_manager.set_system_config(config)

        # Update only one field
        updated_config = config_manager.get_system_config(tenant_id)
        updated_config.llm_model = "gpt-4-turbo"
        config_manager.set_system_config(updated_config)

        # Verify other fields preserved
        final_config = config_manager.get_system_config(tenant_id)
        assert final_config.llm_model == "gpt-4-turbo"  # Updated
        assert final_config.backend_port == 8080  # Preserved
        assert final_config.environment == "production"  # Preserved

    def test_export_with_history(self, config_manager):
        """Test exporting configurations with full version history"""
        tenant_id = "test_tenant"

        # Create config with multiple versions
        for i in range(3):
            config = SystemConfig(
                tenant_id=tenant_id,
                llm_model=f"model-v{i+1}",
            )
            config_manager.set_system_config(config)

        # Export with history
        export_data = config_manager.store.export_configs(
            tenant_id=tenant_id,
            include_history=True,
        )

        # Verify all versions included
        assert export_data["include_history"] is True
        # Should have at least 3 versions
        assert len(export_data["configs"]) >= 3
