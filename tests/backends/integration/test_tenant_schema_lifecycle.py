"""
Integration tests for SchemaRegistry with real Vespa Docker instance.

Tests actual schema deployment, deletion, and tenant isolation via SchemaRegistry.
Requires Docker to be running.
"""

import logging
from pathlib import Path

import pytest
from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def temp_config_manager(tmp_path_factory):
    """
    Provide a temporary ConfigManager for test isolation.

    Scope is "module" because Vespa state persists across test classes in this module.
    This ensures SchemaRegistry tracks all schemas deployed by any test class, preventing
    Vespa schema-removal errors.
    """
    tmp_path = tmp_path_factory.mktemp("config")
    return ConfigManager(db_path=tmp_path / "test_config.db")


@pytest.fixture(scope="module")
def schema_loader():
    """Provide FilesystemSchemaLoader for tests (module-scoped for reuse)."""
    return FilesystemSchemaLoader(Path("configs/schemas"))


@pytest.fixture(scope="module")
def get_backend(vespa_instance, temp_config_manager, schema_loader):
    """
    Factory function to get backend for a tenant.

    Returns a function that creates backend instances for different tenants.
    Module-scoped to reuse backend instances across tests.
    """
    def _get_backend(tenant_id: str):
        registry = BackendRegistry.get_instance()
        config = {
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        }
        return registry.get_search_backend(
            name="vespa",
            tenant_id=tenant_id,
            config=config,
            config_manager=temp_config_manager,
            schema_loader=schema_loader
        )
    return _get_backend


@pytest.mark.integration
class TestSchemaRegistryDeployment:
    """Test schema deployment via SchemaRegistry"""

    def test_deploy_single_schema(self, get_backend):
        """Test deploying a single schema for a tenant"""
        backend = get_backend("acme")

        # Deploy schema
        backend.schema_registry.deploy_schema("acme", "video_colpali_smol500_mv_frame")

        # Verify registered
        schemas = backend.schema_registry.get_tenant_schemas("acme")
        assert len(schemas) == 1
        assert schemas[0].base_schema_name == "video_colpali_smol500_mv_frame"
        assert schemas[0].full_schema_name == "video_colpali_smol500_mv_frame_acme"

    def test_deploy_multiple_schemas_same_tenant(self, get_backend):
        """Test deploying multiple schemas for the same tenant"""
        backend = get_backend("startup")

        # Deploy two schemas
        backend.schema_registry.deploy_schema("startup", "video_colpali_smol500_mv_frame")
        backend.schema_registry.deploy_schema("startup", "video_videoprism_base_mv_chunk_30s")

        # Verify both registered
        schemas = backend.schema_registry.get_tenant_schemas("startup")
        assert len(schemas) == 2
        base_names = {s.base_schema_name for s in schemas}
        assert "video_colpali_smol500_mv_frame" in base_names
        assert "video_videoprism_base_mv_chunk_30s" in base_names

    def test_deploy_same_schema_multiple_tenants(self, get_backend):
        """Test deploying the same base schema for different tenants"""
        # Use same backend but deploy for different tenants
        # (SchemaRegistry is per-backend instance, so both deployments share registry)
        backend = get_backend("multi_tenant_test")

        # Deploy for both tenants via same SchemaRegistry
        backend.schema_registry.deploy_schema("tenant_a", "video_colpali_smol500_mv_frame")
        backend.schema_registry.deploy_schema("tenant_b", "video_colpali_smol500_mv_frame")

        # Verify isolation - each tenant has their own schema
        schemas_a = backend.schema_registry.get_tenant_schemas("tenant_a")
        schemas_b = backend.schema_registry.get_tenant_schemas("tenant_b")

        assert len(schemas_a) == 1
        assert len(schemas_b) == 1
        assert schemas_a[0].full_schema_name == "video_colpali_smol500_mv_frame_tenant_a"
        assert schemas_b[0].full_schema_name == "video_colpali_smol500_mv_frame_tenant_b"

    def test_idempotent_deployment(self, get_backend):
        """Test that deploying same schema twice is idempotent"""
        backend = get_backend("idempotent_test")

        # Deploy twice
        result1 = backend.schema_registry.deploy_schema("idempotent_test", "video_colpali_smol500_mv_frame")
        result2 = backend.schema_registry.deploy_schema("idempotent_test", "video_colpali_smol500_mv_frame")

        # Both should succeed and return same name
        assert result1 == result2
        assert result1 == "video_colpali_smol500_mv_frame_idempotent_test"

        # Should only have one schema registered
        schemas = backend.schema_registry.get_tenant_schemas("idempotent_test")
        assert len(schemas) == 1

    def test_invalid_tenant_id_rejected(self, get_backend):
        """Test that invalid tenant IDs are rejected"""
        backend = get_backend("valid_tenant")

        # Invalid characters
        with pytest.raises(ValueError, match="only alphanumeric, underscore, and colon allowed"):
            backend.schema_registry.deploy_schema("tenant-with-dash", "video_colpali_smol500_mv_frame")

        # Empty tenant_id
        with pytest.raises(ValueError, match="tenant_id is required"):
            backend.schema_registry.deploy_schema("", "video_colpali_smol500_mv_frame")

    def test_invalid_schema_name_rejected(self, get_backend):
        """Test that invalid schema names are rejected"""
        backend = get_backend("test_tenant")

        # Empty schema name
        with pytest.raises(ValueError, match="schema_name is required"):
            backend.schema_registry.deploy_schema("test_tenant", "")

    def test_nonexistent_schema_fails(self, get_backend):
        """Test that deploying nonexistent schema raises exception"""
        backend = get_backend("test_tenant_nonexistent")

        with pytest.raises(Exception, match="Failed to load base schema"):
            backend.schema_registry.deploy_schema("test_tenant_nonexistent", "nonexistent_schema_xyz")
