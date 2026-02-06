"""
Integration tests for configuration persistence with ConfigManager.
Tests complete flow: ConfigManager → Vespa → ConfigAPIMixin → Hot Reload
"""

import logging
import tempfile
from pathlib import Path

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig, TelemetryLevel
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.async_polling import wait_for_vespa_indexing

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.requires_vespa
@pytest.mark.ci_fast
class TestConfigPersistence:
    """Test configuration persistence across restarts"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_config.db"
            yield db_path

    @pytest.fixture(scope="class")
    def config_manager(self, vespa_instance):
        """Create ConfigManager with VespaConfigStore pointing to test Docker container"""
        http_port = vespa_instance["http_port"]
        config_port = vespa_instance["config_port"]

        # Create ConfigManager with VespaConfigStore pointing to test Docker container
        store = VespaConfigStore(
            vespa_url="http://localhost",
            vespa_port=http_port,
        )
        config_manager = ConfigManager(store=store)

        # Deploy metadata schemas via backend initialization
        # This must happen BEFORE using config_manager which writes to config_metadata schema
        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        logger.info(f"Creating backend for system tenant on port {http_port}")
        BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id="system",
            config={
                "backend": {
                    "url": "http://localhost",
                    "port": http_port,
                    "config_port": config_port,
                }
            },
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        logger.info("Backend created successfully - metadata schemas deployed")

        # Wait for metadata schema activation
        wait_for_vespa_indexing(delay=3, description="metadata schema activation")

        return config_manager

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
        telemetry_config = TelemetryConfig(
            enabled=False,
            level=TelemetryLevel.VERBOSE,
            otlp_enabled=False,
            service_name="test-service",
        )

        config_manager.set_telemetry_config(telemetry_config, tenant_id="test_tenant")

        loaded_config = config_manager.get_telemetry_config("test_tenant")

        assert loaded_config.enabled is False
        assert loaded_config.level == TelemetryLevel.VERBOSE
        assert loaded_config.otlp_enabled is False
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
        assert (
            loaded_config.module_config.module_type == DSPyModuleType.CHAIN_OF_THOUGHT
        )
        assert loaded_config.module_config.max_retries == 5
        assert loaded_config.llm_model == "test-llm"

    def test_config_versioning(self, config_manager):
        """Test configuration version tracking"""
        # Create initial config
        system_config_v1 = SystemConfig(tenant_id="test_tenant", llm_model="model-v1")
        config_manager.set_system_config(system_config_v1)

        # Update config
        system_config_v2 = SystemConfig(tenant_id="test_tenant", llm_model="model-v2")
        config_manager.set_system_config(system_config_v2)

        # Update again
        system_config_v3 = SystemConfig(tenant_id="test_tenant", llm_model="model-v3")
        config_manager.set_system_config(system_config_v3)

        # Check current version
        current = config_manager.get_system_config("test_tenant")
        assert current.llm_model == "model-v3"

        # Check storage has all versions
        store = config_manager.store
        entry = store.get_config(
            tenant_id="test_tenant",
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
        )
        # Version should be at least 3 (may be higher if previous tests ran)
        assert entry.version >= 3

    def test_multi_tenant_isolation(self, config_manager):
        """Test tenant isolation in configuration storage"""
        # Create configs for tenant1
        config_tenant1 = SystemConfig(tenant_id="tenant1", llm_model="tenant1-model")
        config_manager.set_system_config(config_tenant1)

        # Create configs for tenant2
        config_tenant2 = SystemConfig(tenant_id="tenant2", llm_model="tenant2-model")
        config_manager.set_system_config(config_tenant2)

        # Verify isolation
        loaded_tenant1 = config_manager.get_system_config("tenant1")
        loaded_tenant2 = config_manager.get_system_config("tenant2")

        assert loaded_tenant1.llm_model == "tenant1-model"
        assert loaded_tenant2.llm_model == "tenant2-model"

    def test_config_survives_manager_restart(self, vespa_instance, config_manager):
        """Test configuration survives ConfigManager restart"""
        http_port = vespa_instance["http_port"]

        # Use the existing config_manager to set initial config
        system_config = SystemConfig(
            tenant_id="restart_test_tenant", llm_model="persistent-model"
        )
        config_manager.set_system_config(system_config)

        # Simulate restart by creating new manager instance with same store
        store2 = VespaConfigStore(
            vespa_url="http://localhost",
            vespa_port=http_port,
        )
        manager2 = ConfigManager(store=store2)

        # Load config with new instance
        loaded_config = manager2.get_system_config("restart_test_tenant")

        assert loaded_config.llm_model == "persistent-model"

    def test_export_import_configs(self, config_manager, temp_db):
        """Test configuration export and import"""
        # Use unique tenant_id to avoid state from other tests
        tenant_id = "export_test_tenant"

        # Create multiple configs
        system_config = SystemConfig(tenant_id=tenant_id, llm_model="test-model")
        routing_config = RoutingConfigUnified(
            tenant_id=tenant_id, routing_mode="tiered"
        )

        config_manager.set_system_config(system_config)
        config_manager.set_routing_config(routing_config)

        # Export
        export_path = temp_db.parent / "export.json"
        config_manager.export_configs(tenant_id, export_path)

        assert export_path.exists()

        # Verify export content
        import json

        with open(export_path) as f:
            exported = json.load(f)

        assert exported["tenant_id"] == tenant_id
        assert "configs" in exported
        assert len(exported["configs"]) == 2

    def test_get_all_configs(self, config_manager):
        """Test retrieving all configurations for a tenant"""
        # Use unique tenant_id to avoid state from other tests
        tenant_id = "get_all_test_tenant"

        # Create multiple configs
        config_manager.set_system_config(
            SystemConfig(tenant_id=tenant_id, llm_model="model1")
        )
        config_manager.set_routing_config(
            RoutingConfigUnified(tenant_id=tenant_id, routing_mode="hybrid")
        )
        config_manager.set_telemetry_config(
            TelemetryConfig(service_name="test"), tenant_id=tenant_id
        )

        # Get all configs
        all_configs = config_manager.get_all_configs(tenant_id)

        assert len(all_configs) == 3
        assert "system:system:system_config" in all_configs
        assert "routing:routing_agent:routing_config" in all_configs
        assert "telemetry:telemetry:telemetry_config" in all_configs

    def test_config_stats(self, config_manager):
        """Test configuration statistics"""
        # Use unique tenant IDs to avoid state from other tests
        tenant1 = "stats_tenant1"
        tenant2 = "stats_tenant2"

        # Create some configs
        config_manager.set_system_config(
            SystemConfig(tenant_id=tenant1, llm_model="model1")
        )
        config_manager.set_system_config(
            SystemConfig(tenant_id=tenant2, llm_model="model2")
        )
        config_manager.set_routing_config(
            RoutingConfigUnified(tenant_id=tenant1, routing_mode="tiered")
        )

        # Get stats - verify basic structure (counts may include other test data)
        stats = config_manager.get_stats()

        # Stats should include at least these new configs
        assert stats["total_configs"] >= 3
        assert stats["total_tenants"] >= 2
        assert stats["configs_per_scope"]["system"] >= 2
        assert stats["configs_per_scope"]["routing"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
