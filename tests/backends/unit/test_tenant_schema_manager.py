"""
Unit tests for TenantSchemaManager

Tests schema routing logic, template loading, and transformation WITHOUT actual Vespa deployment.
Uses mocks to test logic in isolation.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from cogniverse_vespa.tenant_schema_manager import (
    TenantSchemaManager,
    get_tenant_schema_manager,
)


class TestSchemaNameRouting:
    """Test tenant schema name generation and validation"""

    def test_schema_name_routing_basic(self):
        """Test basic schema name routing"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        result = manager.get_tenant_schema_name("acme", "video_colpali_smol500_mv_frame")
        assert result == "video_colpali_smol500_mv_frame_acme"

    def test_schema_name_routing_multiple_tenants(self):
        """Test schema name routing for different tenants"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        assert manager.get_tenant_schema_name("acme", "video_colpali") == "video_colpali_acme"
        assert manager.get_tenant_schema_name("startup", "video_colpali") == "video_colpali_startup"
        assert manager.get_tenant_schema_name("enterprise", "video_colpali") == "video_colpali_enterprise"

    def test_schema_name_routing_different_schemas(self):
        """Test schema name routing for different base schemas"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

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
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            manager.get_tenant_schema_name("", "video_colpali")

    def test_tenant_id_validation_type(self):
        """Test that non-string tenant_id raises ValueError"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        with pytest.raises(ValueError, match="tenant_id must be string"):
            manager.get_tenant_schema_name(123, "video_colpali")

    def test_tenant_id_validation_invalid_chars(self):
        """Test that invalid characters in tenant_id raise ValueError"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        # Hyphens and colons are now allowed for org:tenant format
        # Only test truly invalid characters
        with pytest.raises(ValueError, match="only alphanumeric"):
            manager.get_tenant_schema_name("acme@corp", "video_colpali")

        with pytest.raises(ValueError, match="only alphanumeric"):
            manager.get_tenant_schema_name("acme corp", "video_colpali")

    def test_tenant_id_validation_valid_underscore(self):
        """Test that underscores are allowed in tenant_id"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        result = manager.get_tenant_schema_name("acme_corp", "video_colpali")
        assert result == "video_colpali_acme_corp"

    def test_schema_name_validation_empty(self):
        """Test that empty schema_name raises ValueError"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        with pytest.raises(ValueError, match="schema_name cannot be empty"):
            manager.get_tenant_schema_name("acme", "")

    def test_schema_name_validation_type(self):
        """Test that non-string schema_name raises ValueError"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        with pytest.raises(ValueError, match="schema_name must be string"):
            manager.get_tenant_schema_name("acme", None)


class TestSchemaTemplateLoading:
    """Test loading and transformation of base schema templates"""

    def test_load_base_schema_success(self):
        """Test successful loading of base schema JSON via SchemaLoader"""
        # Clear singleton before test
        TenantSchemaManager._instance = None

        mock_schema_loader = MagicMock()
        mock_schema = {
            "name": "video_colpali_smol500_mv_frame",
            "document": {"name": "video_colpali_smol500_mv_frame", "fields": []},
        }
        mock_schema_loader.load_schema.return_value = mock_schema

        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=mock_schema_loader)

        result = manager.schema_loader.load_schema("video_colpali_smol500_mv_frame")

        assert result["name"] == "video_colpali_smol500_mv_frame"
        assert "document" in result
        mock_schema_loader.load_schema.assert_called_once_with("video_colpali_smol500_mv_frame")

    def test_load_base_schema_not_found(self):
        """Test that missing base schema raises SchemaNotFoundException via SchemaLoader"""
        from cogniverse_core.interfaces.schema_loader import (
            SchemaNotFoundException as SchemaLoaderNotFoundException,
        )

        #  Clear singleton before test
        TenantSchemaManager._instance = None

        mock_schema_loader = MagicMock()
        mock_schema_loader.load_schema.side_effect = SchemaLoaderNotFoundException("Schema not found")
        mock_schema_loader.list_available_schemas.return_value = ["schema1", "schema2"]

        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=mock_schema_loader)

        # Test that schema_loader properly raises exception
        with pytest.raises(SchemaLoaderNotFoundException, match="not found"):
            manager.schema_loader.load_schema("nonexistent_schema")

    def test_transform_schema_for_tenant(self):
        """Test schema transformation includes tenant suffix"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        base_schema = {
            "name": "video_colpali",
            "document": {"name": "video_colpali", "fields": []},
        }

        result = manager._transform_schema_for_tenant(base_schema, "acme", "video_colpali")

        assert result["name"] == "video_colpali_acme"
        assert result["document"]["name"] == "video_colpali_acme"

    def test_transform_schema_preserves_fields(self):
        """Test that schema transformation preserves all other fields"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

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
        """Test listing available base schema templates via SchemaLoader"""
        # Clear singleton before test
        TenantSchemaManager._instance = None

        mock_schema_loader = MagicMock()
        mock_schema_loader.list_available_schemas.return_value = [
            "video_colpali_smol500_mv_frame",
            "video_videoprism_base_mv_chunk_30s",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=mock_schema_loader)

        result = manager.list_available_base_schemas()

        assert "video_colpali_smol500_mv_frame" in result
        assert "video_videoprism_base_mv_chunk_30s" in result
        assert "video_colqwen_omni_mv_chunk_30s" in result
        mock_schema_loader.list_available_schemas.assert_called_once()

    def test_list_available_base_schemas_empty_dir(self):
        """Test listing schemas when directory is empty via SchemaLoader"""
        # Clear singleton before test
        TenantSchemaManager._instance = None

        mock_schema_loader = MagicMock()
        mock_schema_loader.list_available_schemas.return_value = []

        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=mock_schema_loader)

        result = manager.list_available_base_schemas()

        assert result == []
        mock_schema_loader.list_available_schemas.assert_called_once()


class TestSchemaCaching:
    """Test schema deployment caching"""

    def test_cache_deployed_schema(self):
        """Test that deployed schemas are cached"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")

        assert "acme" in manager._deployed_schemas
        assert "video_colpali" in manager._deployed_schemas["acme"]

    def test_cache_multiple_schemas_same_tenant(self):
        """Test caching multiple schemas for same tenant"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")

        assert len(manager._deployed_schemas["acme"]) == 2
        assert "video_colpali" in manager._deployed_schemas["acme"]
        assert "video_videoprism" in manager._deployed_schemas["acme"]

    def test_cache_multiple_tenants(self):
        """Test caching schemas for multiple tenants"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("startup", "video_colpali")

        assert len(manager._deployed_schemas) == 2
        assert "acme" in manager._deployed_schemas
        assert "startup" in manager._deployed_schemas

    def test_list_tenant_schemas(self):
        """Test listing all schemas for a tenant"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")

        result = manager.list_tenant_schemas("acme")

        assert len(result) == 2
        assert "video_colpali_acme" in result
        assert "video_videoprism_acme" in result

    def test_list_tenant_schemas_empty(self):
        """Test listing schemas for tenant with none deployed"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager.clear_cache()

        result = manager.list_tenant_schemas("nonexistent")

        assert result == []

    def test_get_cache_stats(self):
        """Test cache statistics"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
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
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        manager._cache_deployed_schema("acme", "video_colpali")
        assert len(manager._deployed_schemas) > 0

        manager.clear_cache()
        assert len(manager._deployed_schemas) == 0


class TestSchemaDeployment:
    """Test schema deployment with mocked VespaSchemaManager"""

    def test_deploy_tenant_schema_success(self):
        """Test successful tenant schema deployment"""
        # Clear singleton before test
        TenantSchemaManager._instance = None

        mock_schema_loader = MagicMock()
        base_schema = {"name": "video_colpali", "document": {"name": "video_colpali"}}
        mock_schema_loader.load_schema.return_value = base_schema

        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=mock_schema_loader)
        manager.clear_cache()

        with patch.object(manager, "_parse_schema_from_json", return_value=Mock()):
            # Mock schema registry to return False (schema not registered yet)
            with patch.object(manager.schema_registry, "schema_exists", return_value=False):
                # Mock the actual deployment call
                with patch.object(manager.schema_manager, "_deploy_package") as mock_deploy:
                    manager.deploy_tenant_schema("acme", "video_colpali")

                    # Verify deployment was called
                    mock_deploy.assert_called_once()

        # Verify cache updated
        assert "acme" in manager._deployed_schemas
        assert "video_colpali" in manager._deployed_schemas["acme"]
        # Verify schema_loader was called
        mock_schema_loader.load_schema.assert_called_with("video_colpali")

    def test_ensure_tenant_schema_exists_deploys_if_needed(self):
        """Test ensure_tenant_schema_exists deploys schema if not cached and not in registry"""
        # Clear singleton before test
        TenantSchemaManager._instance = None

        mock_schema_loader = MagicMock()
        base_schema = {"name": "video_colpali", "document": {"name": "video_colpali"}}
        mock_schema_loader.load_schema.return_value = base_schema

        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=mock_schema_loader)
        manager.clear_cache()

        with patch.object(manager, "_parse_schema_from_json", return_value=Mock()):
            with patch.object(manager, "_schema_exists_in_vespa", return_value=False):
                # Mock schema registry to return False (schema not registered)
                with patch.object(manager.schema_registry, "schema_exists", return_value=False):
                    with patch.object(manager.schema_manager, "_deploy_package") as mock_deploy:
                        result = manager.ensure_tenant_schema_exists("acme", "video_colpali")
                        mock_deploy.assert_called_once()

        assert result is True
        # Verify schema_loader was called
        mock_schema_loader.load_schema.assert_called_with("video_colpali")

    def test_ensure_tenant_schema_exists_skips_if_cached(self):
        """Test ensure_tenant_schema_exists skips deployment if already cached"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
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
        """Test validation when schema exists (backend query succeeds)"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        mock_config_manager = Mock()

        # Mock backend.search to succeed (schema exists)
        with patch("cogniverse_core.registries.backend_registry.get_backend_registry") as mock_registry:
            mock_backend = Mock()
            mock_backend.search.return_value = {"results": []}  # Empty results but query succeeds
            mock_registry.return_value.get_search_backend.return_value = mock_backend

            result = manager.validate_tenant_schema("acme", "video_colpali", mock_config_manager)

        assert result is True

    def test_validate_tenant_schema_not_exists(self):
        """Test validation when schema doesn't exist (backend query fails)"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        mock_config_manager = Mock()

        # Mock backend.search to raise exception (schema doesn't exist)
        with patch("cogniverse_core.registries.backend_registry.get_backend_registry") as mock_registry:
            mock_backend = Mock()
            mock_backend.search.side_effect = Exception("Schema not found")
            mock_registry.return_value.get_search_backend.return_value = mock_backend

            result = manager.validate_tenant_schema("acme", "video_colpali", mock_config_manager)

        assert result is False


class TestSingletonPattern:
    """Test singleton behavior"""

    def test_singleton_returns_same_instance(self):
        """Test that TenantSchemaManager returns same instance for same parameters"""
        manager1 = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager2 = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        assert manager1 is manager2

    def test_get_tenant_schema_manager_returns_singleton(self):
        """Test helper function returns singleton for same parameters"""
        manager1 = get_tenant_schema_manager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager2 = get_tenant_schema_manager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())

        assert manager1 is manager2


class TestDeleteSchemas:
    """Test schema deletion"""

    def test_delete_tenant_schemas_removes_from_cache(self):
        """Test that deleting schemas removes them from cache"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")

        deleted = manager.delete_tenant_schemas("acme")

        assert len(deleted) == 2
        assert "acme" not in manager._deployed_schemas

    def test_delete_tenant_schemas_nonexistent_tenant(self):
        """Test deleting schemas for tenant with none deployed"""
        manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080, config_manager=MagicMock(), schema_loader=MagicMock())
        manager.clear_cache()

        deleted = manager.delete_tenant_schemas("nonexistent")

        assert len(deleted) == 0
