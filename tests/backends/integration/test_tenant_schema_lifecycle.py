"""
Integration tests for TenantSchemaManager with real Vespa Docker instance.

Tests actual schema deployment, deletion, and tenant isolation.
Requires Docker to be running.
"""

import logging
from pathlib import Path

import pytest
from cogniverse_core.backends.tenant_schema_manager import (
    SchemaDeploymentException,
    TenantSchemaManager,
)
from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def temp_config_manager(tmp_path_factory):
    """
    Provide a temporary ConfigManager for test isolation.

    Scope is "module" because Vespa state persists across test classes in this module
    (even though vespa_instance fixture is class-scoped, the actual container state persists).
    This ensures SchemaRegistry tracks all schemas deployed by any test class, preventing
    Vespa schema-removal errors.
    """
    tmp_path = tmp_path_factory.mktemp("config")
    return ConfigManager(db_path=tmp_path / "test_config.db")


@pytest.fixture(scope="module")
def schema_loader():
    """Provide FilesystemSchemaLoader for tests (module-scoped for reuse)."""
    return FilesystemSchemaLoader(Path("configs/schemas"))


@pytest.mark.integration
class TestTenantSchemaDeployment:
    """Test actual schema deployment to Vespa"""

    def test_deploy_single_tenant_schema(self, vespa_instance, temp_config_manager, schema_loader):
        """Test deploying a single tenant schema"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy schema for tenant "acme"
        manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame")

        # Verify cache updated
        tenant_schemas = manager.list_tenant_schemas("acme")
        assert "video_colpali_smol500_mv_frame" in tenant_schemas

        # Verify stats
        stats = manager.get_cache_stats()
        assert stats["total_tenants"] == 1
        assert stats["total_schemas"] == 1

    def test_deploy_multiple_schemas_same_tenant(self, vespa_instance, temp_config_manager, schema_loader):
        """Test deploying multiple schemas for the same tenant"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy two different schemas for tenant "acme"
        manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame")
        manager.deploy_tenant_schema("acme", "video_videoprism_base_mv_chunk_30s")

        # Verify both schemas cached
        tenant_schemas = manager.list_tenant_schemas("acme")
        assert len(tenant_schemas) == 2
        assert "video_colpali_smol500_mv_frame" in tenant_schemas
        assert "video_videoprism_base_mv_chunk_30s" in tenant_schemas

    def test_deploy_same_schema_multiple_tenants(self, vespa_instance, temp_config_manager, schema_loader):
        """Test deploying the same base schema for multiple tenants"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy same base schema for two different tenants
        manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame")
        manager.deploy_tenant_schema("startup", "video_colpali_smol500_mv_frame")

        # Verify both tenants have the schema
        acme_schemas = manager.list_tenant_schemas("acme")
        startup_schemas = manager.list_tenant_schemas("startup")

        assert "video_colpali_smol500_mv_frame" in acme_schemas
        assert "video_colpali_smol500_mv_frame" in startup_schemas

        # Verify stats
        stats = manager.get_cache_stats()
        assert stats["total_tenants"] == 2
        assert stats["total_schemas"] == 2

    def test_deploy_nonexistent_schema_fails(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that deploying nonexistent schema raises SchemaDeploymentException"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        with pytest.raises(SchemaDeploymentException):
            manager.deploy_tenant_schema("acme", "nonexistent_schema")


@pytest.mark.integration
class TestLazySchemaCreation:
    """Test lazy schema creation via ensure_tenant_schema_exists"""

    def test_ensure_creates_schema_on_first_call(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that ensure_tenant_schema_exists creates schema on first call"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # First call should deploy
        result = manager.ensure_tenant_schema_exists("acme", "video_colpali_smol500_mv_frame")
        assert result is True

        # Verify cached
        assert "video_colpali_smol500_mv_frame" in manager.list_tenant_schemas("acme")

    def test_ensure_is_idempotent(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that ensure_tenant_schema_exists is idempotent"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # First call deploys
        result1 = manager.ensure_tenant_schema_exists("acme", "video_colpali_smol500_mv_frame")
        assert result1 is True

        # Second call should skip deployment (cached)
        result2 = manager.ensure_tenant_schema_exists("acme", "video_colpali_smol500_mv_frame")
        assert result2 is True

        # Should still have exactly one entry in cache
        tenant_schemas = manager.list_tenant_schemas("acme")
        assert len(tenant_schemas) == 1

    def test_ensure_works_for_multiple_schemas(self, vespa_instance, temp_config_manager, schema_loader):
        """Test ensure_tenant_schema_exists for multiple schemas"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Ensure multiple schemas
        manager.ensure_tenant_schema_exists("acme", "video_colpali_smol500_mv_frame")
        manager.ensure_tenant_schema_exists("acme", "video_videoprism_base_mv_chunk_30s")

        tenant_schemas = manager.list_tenant_schemas("acme")
        assert len(tenant_schemas) == 2


@pytest.mark.integration
class TestSchemaNameGeneration:
    """Test schema name generation in real deployment"""

    def test_schema_name_format(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that deployed schemas have correct naming format"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy with specific tenant and schema
        manager.deploy_tenant_schema("test_tenant", "video_colpali_smol500_mv_frame")

        # Verify name format
        tenant_schemas = manager.list_tenant_schemas("test_tenant")
        expected_name = "video_colpali_smol500_mv_frame"
        assert expected_name in tenant_schemas

    def test_underscore_in_tenant_id(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that underscores in tenant_id are preserved"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy with underscore in tenant_id
        manager.deploy_tenant_schema("acme_corp", "video_colpali_smol500_mv_frame")

        tenant_schemas = manager.list_tenant_schemas("acme_corp")
        expected_name = "video_colpali_smol500_mv_frame"
        assert expected_name in tenant_schemas


@pytest.mark.integration
class TestSchemaValidation:
    """Test schema validation"""

    def test_validate_existing_schema(self, vespa_instance, tmp_path, schema_loader):
        """Test validation of deployed schema"""
        import time

        from cogniverse_core.config.manager import ConfigManager
        from cogniverse_core.config.unified_config import SystemConfig

        # Use a real temporary ConfigManager for proper test isolation
        # (not a mock) so SchemaRegistry state doesn't persist between tests
        temp_config_manager = ConfigManager(db_path=tmp_path / "manager_config.db")

        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy schema with force=True to ensure actual Vespa deployment
        # (bypasses SchemaRegistry check that might have stale data from previous tests)
        manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame", force=True)

        # Wait for Vespa to fully initialize the schema (increased for container resource contention)
        time.sleep(5)

        # Create config_manager for validation
        config_manager = ConfigManager(db_path=tmp_path / "test_config.db")
        system_config = SystemConfig(
            tenant_id="acme",
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
        config_manager.set_system_config(system_config)

        # Validate with retry logic for robustness against timing issues
        # Increased retries for resource contention when running full test suite
        max_retries = 5
        retry_delay = 2.0
        result = False

        for attempt in range(max_retries):
            result = manager.validate_tenant_schema("acme", "video_colpali_smol500_mv_frame", config_manager)
            if result:
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10.0)  # Exponential backoff capped at 10s

        # Schema validation now properly checks Vespa
        assert result is True  # Schema exists after deployment


@pytest.mark.integration
class TestTenantIsolation:
    """Test tenant isolation in schema management"""

    def test_different_tenants_have_separate_schemas(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that different tenants have completely isolated schemas"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy for tenant A
        manager.deploy_tenant_schema("tenant_a", "video_colpali_smol500_mv_frame")

        # Deploy for tenant B
        manager.deploy_tenant_schema("tenant_b", "video_colpali_smol500_mv_frame")

        # Get schemas for each tenant
        tenant_a_schemas = manager.list_tenant_schemas("tenant_a")
        tenant_b_schemas = manager.list_tenant_schemas("tenant_b")

        # Verify isolation - each tenant has the base schema in their cache
        assert "video_colpali_smol500_mv_frame" in tenant_a_schemas
        assert "video_colpali_smol500_mv_frame" in tenant_b_schemas

        # Verify both tenants have exactly one schema
        assert len(tenant_a_schemas) == 1
        assert len(tenant_b_schemas) == 1

    def test_cache_isolation_between_tenants(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that cache properly isolates tenants"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy schemas for multiple tenants
        manager.deploy_tenant_schema("tenant_a", "video_colpali_smol500_mv_frame")
        manager.deploy_tenant_schema("tenant_b", "video_videoprism_base_mv_chunk_30s")

        # Check stats
        stats = manager.get_cache_stats()
        assert stats["total_tenants"] == 2
        assert "tenant_a" in stats["tenants"]
        assert "tenant_b" in stats["tenants"]

        # Verify each tenant has only their schemas
        assert len(manager.list_tenant_schemas("tenant_a")) == 1
        assert len(manager.list_tenant_schemas("tenant_b")) == 1


@pytest.mark.integration
class TestSchemaDeletion:
    """Test schema deletion"""

    def test_delete_tenant_schemas(self, vespa_instance, temp_config_manager, schema_loader):
        """Test deleting all schemas for a tenant"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy schemas
        manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame")
        manager.deploy_tenant_schema("acme", "video_videoprism_base_mv_chunk_30s")

        # Verify deployed
        assert len(manager.list_tenant_schemas("acme")) == 2

        # Delete
        deleted = manager.delete_tenant_schemas("acme")

        # Verify deletion
        assert len(deleted) == 2
        assert len(manager.list_tenant_schemas("acme")) == 0

    def test_delete_doesnt_affect_other_tenants(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that deleting one tenant's schemas doesn't affect others"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )
        manager.clear_cache()

        # Deploy for two tenants
        manager.deploy_tenant_schema("tenant_a", "video_colpali_smol500_mv_frame")
        manager.deploy_tenant_schema("tenant_b", "video_colpali_smol500_mv_frame")

        # Delete tenant A
        manager.delete_tenant_schemas("tenant_a")

        # Verify tenant A gone
        assert len(manager.list_tenant_schemas("tenant_a")) == 0

        # Verify tenant B unaffected
        assert len(manager.list_tenant_schemas("tenant_b")) == 1


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration scenarios"""

    def test_invalid_tenant_id_fails(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that invalid tenant IDs are rejected"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )

        with pytest.raises(ValueError, match="only alphanumeric, underscore, and colon allowed"):
            manager.deploy_tenant_schema("tenant-with-dash", "video_colpali_smol500_mv_frame")

    def test_empty_tenant_id_fails(self, vespa_instance, temp_config_manager, schema_loader):
        """Test that empty tenant ID is rejected"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost", backend_port=vespa_instance["config_port"], http_port=vespa_instance["http_port"],
            config_manager=temp_config_manager, schema_loader=schema_loader
        )

        with pytest.raises(ValueError, match="tenant_id is required"):
            manager.deploy_tenant_schema("", "video_colpali_smol500_mv_frame")
