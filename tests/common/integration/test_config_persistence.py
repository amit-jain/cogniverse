"""
Integration tests for configuration persistence with ConfigManager.
Tests complete flow: ConfigManager → SQLite → ConfigAPIMixin → Hot Reload
"""

import tempfile
from pathlib import Path

import pytest
from cogniverse_core.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
)
from cogniverse_core.config.config_manager import ConfigManager
from cogniverse_core.config.unified_config import (
    RoutingConfigUnified,
    SystemConfig,
    TelemetryConfigUnified,
)


class TestConfigPersistence:
    """Test configuration persistence across restarts"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_config.db"
            yield db_path

    @pytest.fixture
    def config_manager(self, temp_db):
        """Create ConfigManager with temp database"""
        # Reset singleton
        ConfigManager._instance = None
        manager = ConfigManager(db_path=temp_db)
        return manager

    def test_system_config_persistence(self, config_manager):
        """Test system configuration persists and loads"""
        # Create system config
        system_config = SystemConfig(
            tenant_id="test_tenant",
            routing_agent_url="http://localhost:9001",
            video_agent_url="http://localhost:9002",
            search_backend="vespa",
            llm_model="test-model",
        )

        # Persist
        config_manager.set_system_config(system_config)

        # Load back
        loaded_config = config_manager.get_system_config("test_tenant")

        assert loaded_config.tenant_id == "test_tenant"
        assert loaded_config.routing_agent_url == "http://localhost:9001"
        assert loaded_config.video_agent_url == "http://localhost:9002"
        assert loaded_config.llm_model == "test-model"

    def test_routing_config_persistence(self, config_manager):
        """Test routing configuration persists and loads"""
        routing_config = RoutingConfigUnified(
            tenant_id="test_tenant",
            routing_mode="ensemble",
            enable_fast_path=False,
            gliner_threshold=0.5,
            llm_routing_model="custom-model",
        )

        config_manager.set_routing_config(routing_config)

        loaded_config = config_manager.get_routing_config("test_tenant")

        assert loaded_config.tenant_id == "test_tenant"
        assert loaded_config.routing_mode == "ensemble"
        assert loaded_config.enable_fast_path is False
        assert loaded_config.gliner_threshold == 0.5
        assert loaded_config.llm_routing_model == "custom-model"

    def test_telemetry_config_persistence(self, config_manager):
        """Test telemetry configuration persists and loads"""
        telemetry_config = TelemetryConfigUnified(
            tenant_id="test_tenant",
            enabled=False,
            level="verbose",
            phoenix_enabled=False,
            service_name="test-service",
        )

        config_manager.set_telemetry_config(telemetry_config)

        loaded_config = config_manager.get_telemetry_config("test_tenant")

        assert loaded_config.tenant_id == "test_tenant"
        assert loaded_config.enabled is False
        assert loaded_config.level == "verbose"
        assert loaded_config.phoenix_enabled is False
        assert loaded_config.service_name == "test-service"

    def test_agent_config_persistence(self, config_manager):
        """Test agent configuration persists and loads"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.CHAIN_OF_THOUGHT,
            signature="TestSignature",
            max_retries=5,
            temperature=0.9,
        )

        agent_config = AgentConfig(
            agent_name="test_agent",
            agent_version="2.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:9999",
            capabilities=["test"],
            skills=[],
            module_config=module_config,
            llm_model="test-llm",
        )

        config_manager.set_agent_config(
            tenant_id="test_tenant", agent_name="test_agent", agent_config=agent_config
        )

        loaded_config = config_manager.get_agent_config("test_tenant", "test_agent")

        assert loaded_config.agent_name == "test_agent"
        assert loaded_config.agent_version == "2.0.0"
        assert loaded_config.module_config.module_type == DSPyModuleType.CHAIN_OF_THOUGHT
        assert loaded_config.module_config.max_retries == 5
        assert loaded_config.llm_model == "test-llm"

    def test_config_versioning(self, config_manager):
        """Test configuration version tracking"""
        # Create initial config
        system_config_v1 = SystemConfig(
            tenant_id="test_tenant", llm_model="model-v1"
        )
        config_manager.set_system_config(system_config_v1)

        # Update config
        system_config_v2 = SystemConfig(
            tenant_id="test_tenant", llm_model="model-v2"
        )
        config_manager.set_system_config(system_config_v2)

        # Update again
        system_config_v3 = SystemConfig(
            tenant_id="test_tenant", llm_model="model-v3"
        )
        config_manager.set_system_config(system_config_v3)

        # Check current version
        current = config_manager.get_system_config("test_tenant")
        assert current.llm_model == "model-v3"

        # Check storage has all versions
        store = config_manager.store
        entry = store.get_config(
            tenant_id="test_tenant",
            scope="system",  # Use string instead of removed ConfigScope enum
            service="system",
            config_key="system_config",
        )
        assert entry.version == 3

    def test_multi_tenant_isolation(self, config_manager):
        """Test tenant isolation in configuration storage"""
        # Create configs for tenant1
        config_tenant1 = SystemConfig(
            tenant_id="tenant1", llm_model="tenant1-model"
        )
        config_manager.set_system_config(config_tenant1)

        # Create configs for tenant2
        config_tenant2 = SystemConfig(
            tenant_id="tenant2", llm_model="tenant2-model"
        )
        config_manager.set_system_config(config_tenant2)

        # Verify isolation
        loaded_tenant1 = config_manager.get_system_config("tenant1")
        loaded_tenant2 = config_manager.get_system_config("tenant2")

        assert loaded_tenant1.llm_model == "tenant1-model"
        assert loaded_tenant2.llm_model == "tenant2-model"

    def test_config_survives_manager_restart(self, temp_db):
        """Test configuration survives ConfigManager restart"""
        # Create first manager instance
        manager1 = ConfigManager(db_path=temp_db)
        system_config = SystemConfig(
            tenant_id="test_tenant", llm_model="persistent-model"
        )
        manager1.set_system_config(system_config)

        # Simulate restart by creating new manager with same db_path
        # The singleton will return the same instance, but we verify
        # that data persists in the database
        ConfigManager._instance = None
        ConfigManager._db_path = None
        manager2 = ConfigManager(db_path=temp_db)

        # Load config with new instance
        loaded_config = manager2.get_system_config("test_tenant")

        assert loaded_config.llm_model == "persistent-model"

    def test_export_import_configs(self, config_manager, temp_db):
        """Test configuration export and import"""
        # Create multiple configs
        system_config = SystemConfig(tenant_id="test_tenant", llm_model="test-model")
        routing_config = RoutingConfigUnified(
            tenant_id="test_tenant", routing_mode="tiered"
        )

        config_manager.set_system_config(system_config)
        config_manager.set_routing_config(routing_config)

        # Export
        export_path = temp_db.parent / "export.json"
        config_manager.export_configs("test_tenant", export_path)

        assert export_path.exists()

        # Verify export content
        import json

        with open(export_path) as f:
            exported = json.load(f)

        assert exported["tenant_id"] == "test_tenant"
        assert "configs" in exported
        assert len(exported["configs"]) == 2

    def test_get_all_configs(self, config_manager):
        """Test retrieving all configurations for a tenant"""
        # Create multiple configs
        config_manager.set_system_config(
            SystemConfig(tenant_id="test_tenant", llm_model="model1")
        )
        config_manager.set_routing_config(
            RoutingConfigUnified(tenant_id="test_tenant", routing_mode="hybrid")
        )
        config_manager.set_telemetry_config(
            TelemetryConfigUnified(tenant_id="test_tenant", service_name="test")
        )

        # Get all configs
        all_configs = config_manager.get_all_configs("test_tenant")

        assert len(all_configs) == 3
        assert "system:system:system_config" in all_configs
        assert "routing:routing_agent:routing_config" in all_configs
        assert "telemetry:telemetry:telemetry_config" in all_configs

    def test_config_stats(self, config_manager):
        """Test configuration statistics"""
        # Create some configs
        config_manager.set_system_config(
            SystemConfig(tenant_id="tenant1", llm_model="model1")
        )
        config_manager.set_system_config(
            SystemConfig(tenant_id="tenant2", llm_model="model2")
        )
        config_manager.set_routing_config(
            RoutingConfigUnified(tenant_id="tenant1", routing_mode="tiered")
        )

        # Get stats
        stats = config_manager.get_stats()

        assert stats["total_configs"] == 3
        assert stats["total_tenants"] == 2
        assert stats["configs_per_scope"]["system"] == 2
        assert stats["configs_per_scope"]["routing"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
