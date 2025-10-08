"""
Unit tests for TenantSchemaManager

Tests schema routing logic, template loading, and transformation WITHOUT actual Vespa deployment.
Uses mocks to test logic in isolation.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

from src.backends.vespa.tenant_schema_manager import (
    TenantSchemaManager,
    SchemaNotFoundException,
    SchemaDeploymentException,
    get_tenant_schema_manager,
)


class TestSchemaNameRouting:
    """Test tenant schema name generation and validation"""

    def test_schema_name_routing_basic(self):
        """Test basic schema name routing"""
        manager = TenantSchemaManager()

        result = manager.get_tenant_schema_name("acme", "video_colpali_smol500_mv_frame")
        assert result == "video_colpali_smol500_mv_frame_acme"

    def test_schema_name_routing_multiple_tenants(self):
        """Test schema name routing for different tenants"""
        manager = TenantSchemaManager()

        assert manager.get_tenant_schema_name("acme", "video_colpali") == "video_colpali_acme"
        assert manager.get_tenant_schema_name("startup", "video_colpali") == "video_colpali_startup"
        assert manager.get_tenant_schema_name("enterprise", "video_colpali") == "video_colpali_enterprise"

    def test_schema_name_routing_different_schemas(self):
        """Test schema name routing for different base schemas"""
        manager = TenantSchemaManager()

        assert (
            manager.get_tenant_schema_name("acme", "video_colpali_smol500_mv_frame")
            == "video_colpali_smol500_mv_frame_acme"
        )
        assert (
            manager.get_tenant_schema_name("acme", "video_videoprism_base_mv_chunk_30s")
            == "video_videoprism_base_mv_chunk_30s_acme"
        )

    def test_tenant_id_validation_empty(self):
        """Test that empty tenant_id raises ValueError"""
        manager = TenantSchemaManager()

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            manager.get_tenant_schema_name("", "video_colpali")

    def test_tenant_id_validation_type(self):
        """Test that non-string tenant_id raises ValueError"""
        manager = TenantSchemaManager()

        with pytest.raises(ValueError, match="tenant_id must be string"):
            manager.get_tenant_schema_name(123, "video_colpali")

    def test_tenant_id_validation_invalid_chars(self):
        """Test that invalid characters in tenant_id raise ValueError"""
        manager = TenantSchemaManager()

        # Hyphens and colons are now allowed for org:tenant format
        # Only test truly invalid characters
        with pytest.raises(ValueError, match="only alphanumeric"):
            manager.get_tenant_schema_name("acme@corp", "video_colpali")

        with pytest.raises(ValueError, match="only alphanumeric"):
            manager.get_tenant_schema_name("acme corp", "video_colpali")

    def test_tenant_id_validation_valid_underscore(self):
        """Test that underscores are allowed in tenant_id"""
        manager = TenantSchemaManager()

        result = manager.get_tenant_schema_name("acme_corp", "video_colpali")
        assert result == "video_colpali_acme_corp"

    def test_schema_name_validation_empty(self):
        """Test that empty schema_name raises ValueError"""
        manager = TenantSchemaManager()

        with pytest.raises(ValueError, match="schema_name cannot be empty"):
            manager.get_tenant_schema_name("acme", "")

    def test_schema_name_validation_type(self):
        """Test that non-string schema_name raises ValueError"""
        manager = TenantSchemaManager()

        with pytest.raises(ValueError, match="schema_name must be string"):
            manager.get_tenant_schema_name("acme", None)


class TestSchemaTemplateLoading:
    """Test loading and transformation of base schema templates"""

    def test_load_base_schema_success(self):
        """Test successful loading of base schema JSON"""
        manager = TenantSchemaManager()

        # Mock file reading
        mock_schema = {
            "name": "video_colpali_smol500_mv_frame",
            "document": {"name": "video_colpali_smol500_mv_frame", "fields": []},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_schema))
        ):
            result = manager._load_base_schema_json("video_colpali_smol500_mv_frame")

        assert result["name"] == "video_colpali_smol500_mv_frame"
        assert "document" in result

    def test_load_base_schema_not_found(self):
        """Test that missing base schema raises SchemaNotFoundException"""
        manager = TenantSchemaManager()

        with patch("pathlib.Path.exists", return_value=False), patch.object(
            manager, "list_available_base_schemas", return_value=["schema1", "schema2"]
        ):
            with pytest.raises(SchemaNotFoundException, match="not found"):
                manager._load_base_schema_json("nonexistent_schema")

    def test_transform_schema_for_tenant(self):
        """Test schema transformation includes tenant suffix"""
        manager = TenantSchemaManager()

        base_schema = {
            "name": "video_colpali",
            "document": {"name": "video_colpali", "fields": []},
        }

        result = manager._transform_schema_for_tenant(base_schema, "acme", "video_colpali")

        assert result["name"] == "video_colpali_acme"
        assert result["document"]["name"] == "video_colpali_acme"

    def test_transform_schema_preserves_fields(self):
        """Test that schema transformation preserves all other fields"""
        manager = TenantSchemaManager()

        base_schema = {
            "name": "video_colpali",
            "document": {
                "name": "video_colpali",
                "fields": [
                    {"name": "video_id", "type": "string"},
                    {"name": "embedding", "type": "tensor<float>(x[128])"},
                ],
            },
            "fieldsets": [{"name": "default", "fields": ["video_id"]}],
        }

        result = manager._transform_schema_for_tenant(base_schema, "acme", "video_colpali")

        # Names updated
        assert result["name"] == "video_colpali_acme"
        assert result["document"]["name"] == "video_colpali_acme"

        # Fields preserved
        assert len(result["document"]["fields"]) == 2
        assert result["document"]["fields"][0]["name"] == "video_id"
        assert result["document"]["fields"][1]["type"] == "tensor<float>(x[128])"

        # Fieldsets preserved
        assert result["fieldsets"] == base_schema["fieldsets"]

    def test_list_available_base_schemas(self):
        """Test listing available base schema templates"""
        manager = TenantSchemaManager()

        # Mock glob to return fake schema files
        mock_files = [
            Path("video_colpali_smol500_mv_frame_schema.json"),
            Path("video_videoprism_base_mv_chunk_30s_schema.json"),
            Path("video_colqwen_omni_mv_chunk_30s_schema.json"),
        ]

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.glob", return_value=mock_files
        ):
            result = manager.list_available_base_schemas()

        assert "video_colpali_smol500_mv_frame" in result
        assert "video_videoprism_base_mv_chunk_30s" in result
        assert "video_colqwen_omni_mv_chunk_30s" in result

    def test_list_available_base_schemas_empty_dir(self):
        """Test listing schemas when directory doesn't exist"""
        manager = TenantSchemaManager()

        with patch("pathlib.Path.exists", return_value=False):
            result = manager.list_available_base_schemas()

        assert result == []


class TestSchemaCaching:
    """Test schema deployment caching"""

    def test_cache_deployed_schema(self):
        """Test that deployed schemas are cached"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")

        assert "acme" in manager._deployed_schemas
        assert "video_colpali" in manager._deployed_schemas["acme"]

    def test_cache_multiple_schemas_same_tenant(self):
        """Test caching multiple schemas for same tenant"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")

        assert len(manager._deployed_schemas["acme"]) == 2
        assert "video_colpali" in manager._deployed_schemas["acme"]
        assert "video_videoprism" in manager._deployed_schemas["acme"]

    def test_cache_multiple_tenants(self):
        """Test caching schemas for multiple tenants"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("startup", "video_colpali")

        assert len(manager._deployed_schemas) == 2
        assert "acme" in manager._deployed_schemas
        assert "startup" in manager._deployed_schemas

    def test_list_tenant_schemas(self):
        """Test listing all schemas for a tenant"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")

        result = manager.list_tenant_schemas("acme")

        assert len(result) == 2
        assert "video_colpali_acme" in result
        assert "video_videoprism_acme" in result

    def test_list_tenant_schemas_empty(self):
        """Test listing schemas for tenant with none deployed"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        result = manager.list_tenant_schemas("nonexistent")

        assert result == []

    def test_get_cache_stats(self):
        """Test cache statistics"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")
        manager._cache_deployed_schema("startup", "video_colpali")

        stats = manager.get_cache_stats()

        assert stats["tenants_cached"] == 2
        assert stats["total_schemas_cached"] == 3
        assert "acme" in stats["tenants"]
        assert "startup" in stats["tenants"]
        assert len(stats["tenants"]["acme"]) == 2

    def test_clear_cache(self):
        """Test clearing the cache"""
        manager = TenantSchemaManager()

        manager._cache_deployed_schema("acme", "video_colpali")
        assert len(manager._deployed_schemas) > 0

        manager.clear_cache()
        assert len(manager._deployed_schemas) == 0


class TestSchemaDeployment:
    """Test schema deployment with mocked VespaSchemaManager"""

    def test_deploy_tenant_schema_success(self):
        """Test successful tenant schema deployment"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        # Mock all the internal methods that do actual work
        base_schema = {"name": "video_colpali", "document": {"name": "video_colpali"}}

        with patch.object(manager, "_load_base_schema_json", return_value=base_schema):
            with patch.object(manager, "_parse_schema_from_json", return_value=Mock()):
                # Mock the actual deployment call
                with patch.object(manager.schema_manager, "_deploy_package") as mock_deploy:
                    manager.deploy_tenant_schema("acme", "video_colpali")

                    # Verify deployment was called
                    mock_deploy.assert_called_once()

        # Verify cache updated
        assert "acme" in manager._deployed_schemas
        assert "video_colpali" in manager._deployed_schemas["acme"]

    def test_ensure_tenant_schema_exists_deploys_if_needed(self):
        """Test ensure_tenant_schema_exists deploys schema if not cached"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        base_schema = {"name": "video_colpali", "document": {"name": "video_colpali"}}

        with patch.object(manager, "_load_base_schema_json", return_value=base_schema):
            with patch.object(manager, "_parse_schema_from_json", return_value=Mock()):
                with patch.object(manager, "_schema_exists_in_vespa", return_value=False):
                    with patch.object(manager.schema_manager, "_deploy_package") as mock_deploy:
                        result = manager.ensure_tenant_schema_exists("acme", "video_colpali")
                        mock_deploy.assert_called_once()

        assert result is True

    def test_ensure_tenant_schema_exists_skips_if_cached(self):
        """Test ensure_tenant_schema_exists skips deployment if already cached"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        # Pre-cache schema
        manager._cache_deployed_schema("acme", "video_colpali")

        with patch.object(manager.schema_manager, "_deploy_package") as mock_deploy:
            result = manager.ensure_tenant_schema_exists("acme", "video_colpali")

            # Should not deploy
            mock_deploy.assert_not_called()

        assert result is True


class TestSchemaValidation:
    """Test schema validation"""

    def test_validate_tenant_schema_exists(self):
        """Test validation when schema exists"""
        manager = TenantSchemaManager()

        with patch.object(manager, "_schema_exists_in_vespa", return_value=True):
            result = manager.validate_tenant_schema("acme", "video_colpali")

        assert result is True

    def test_validate_tenant_schema_not_exists(self):
        """Test validation when schema doesn't exist"""
        manager = TenantSchemaManager()

        with patch.object(manager, "_schema_exists_in_vespa", return_value=False):
            result = manager.validate_tenant_schema("acme", "video_colpali")

        assert result is False


class TestSingletonPattern:
    """Test singleton behavior"""

    def test_singleton_returns_same_instance(self):
        """Test that TenantSchemaManager is a singleton"""
        manager1 = TenantSchemaManager()
        manager2 = TenantSchemaManager()

        assert manager1 is manager2

    def test_get_tenant_schema_manager_returns_singleton(self):
        """Test helper function returns singleton"""
        manager1 = get_tenant_schema_manager()
        manager2 = get_tenant_schema_manager()

        assert manager1 is manager2


class TestDeleteSchemas:
    """Test schema deletion"""

    def test_delete_tenant_schemas_removes_from_cache(self):
        """Test that deleting schemas removes them from cache"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")

        deleted = manager.delete_tenant_schemas("acme")

        assert len(deleted) == 2
        assert "acme" not in manager._deployed_schemas

    def test_delete_tenant_schemas_nonexistent_tenant(self):
        """Test deleting schemas for tenant with none deployed"""
        manager = TenantSchemaManager()
        manager.clear_cache()

        deleted = manager.delete_tenant_schemas("nonexistent")

        assert len(deleted) == 0
