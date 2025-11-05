"""
Unit tests for TenantSchemaManager (New Architecture)

Tests the backend-agnostic tenant schema management layer.
After refactoring, TenantSchemaManager delegates operations to backends.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401

from cogniverse_core.backends.tenant_schema_manager import (
    TenantSchemaManager,
    SchemaDeploymentException,
)


class TestSchemaNameRouting:
    """Test tenant schema name generation"""

    def test_schema_name_routing_basic(self):
        """Test basic schema name routing delegates to backend"""
        mock_backend = MagicMock()
        mock_backend.get_tenant_schema_name.return_value = "video_colpali_smol500_mv_frame_acme"

        with patch("cogniverse_core.backends.tenant_schema_manager.BackendRegistry") as mock_registry:
            mock_registry.get_instance.return_value.get_search_backend.return_value = mock_backend
            manager = TenantSchemaManager(
                backend_name="vespa",
                backend_url="http://localhost",
                backend_port=19071,
                http_port=8080,
                config_manager=MagicMock(),
                schema_loader=MagicMock()
            )

            result = manager.get_tenant_schema_name("acme", "video_colpali_smol500_mv_frame")

        assert result == "video_colpali_smol500_mv_frame_acme"
        mock_backend.get_tenant_schema_name.assert_called_once_with("acme", "video_colpali_smol500_mv_frame")

    def test_tenant_id_validation_empty(self):
        """Test that empty tenant_id raises ValueError"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost",
            backend_port=19071,
            http_port=8080,
            config_manager=MagicMock(),
            schema_loader=MagicMock()
        )

        with pytest.raises(ValueError, match="tenant_id is required"):
            manager.get_tenant_schema_name("", "video_colpali")

    def test_schema_name_validation_empty(self):
        """Test that empty schema_name raises ValueError"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost",
            backend_port=19071,
            http_port=8080,
            config_manager=MagicMock(),
            schema_loader=MagicMock()
        )

        with pytest.raises(ValueError, match="schema_name is required"):
            manager.get_tenant_schema_name("acme", "")


class TestSchemaDeployment:
    """Test schema deployment delegation to backend"""

    def test_deploy_tenant_schema_success(self):
        """Test successful tenant schema deployment delegates to backend"""
        mock_backend = MagicMock()
        mock_backend.schema_exists.return_value = False
        mock_backend.deploy_schema.return_value = True
        mock_backend.get_tenant_schema_name.return_value = "video_colpali_acme"

        with patch("cogniverse_core.backends.tenant_schema_manager.BackendRegistry") as mock_registry:
            mock_registry.get_instance.return_value.get_search_backend.return_value = mock_backend

            manager = TenantSchemaManager(
                backend_name="vespa",
                backend_url="http://localhost",
                backend_port=19071,
                http_port=8080,
                config_manager=MagicMock(),
                schema_loader=MagicMock()
            )
            manager.clear_cache()

            result = manager.deploy_tenant_schema("acme", "video_colpali")

        assert result == "video_colpali_acme"
        mock_backend.deploy_schema.assert_called_once_with(
            schema_name="video_colpali",
            tenant_id="acme"
        )

    def test_deploy_tenant_schema_skips_if_exists(self):
        """Test schema deployment skipped if already exists"""
        mock_backend = MagicMock()
        mock_backend.schema_exists.return_value = True
        mock_backend.get_tenant_schema_name.return_value = "video_colpali_acme"

        with patch("cogniverse_core.backends.tenant_schema_manager.BackendRegistry") as mock_registry:
            mock_registry.get_instance.return_value.get_search_backend.return_value = mock_backend

            manager = TenantSchemaManager(
                backend_name="vespa",
                backend_url="http://localhost",
                backend_port=19071,
                http_port=8080,
                config_manager=MagicMock(),
                schema_loader=MagicMock()
            )
            manager.clear_cache()

            result = manager.deploy_tenant_schema("acme", "video_colpali", force=False)

        assert result == "video_colpali_acme"
        mock_backend.deploy_schema.assert_not_called()


class TestSchemaCaching:
    """Test schema deployment caching"""

    def test_list_tenant_schemas(self):
        """Test listing cached schemas for a tenant"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost",
            backend_port=19071,
            http_port=8080,
            config_manager=MagicMock(),
            schema_loader=MagicMock()
        )
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")

        result = manager.list_tenant_schemas("acme")

        assert len(result) == 2
        assert "video_colpali" in result  # Returns base schema names
        assert "video_videoprism" in result

    def test_get_cache_stats(self):
        """Test cache statistics"""
        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost",
            backend_port=19071,
            http_port=8080,
            config_manager=MagicMock(),
            schema_loader=MagicMock()
        )
        manager.clear_cache()

        manager._cache_deployed_schema("acme", "video_colpali")
        manager._cache_deployed_schema("acme", "video_videoprism")
        manager._cache_deployed_schema("startup", "video_colpali")

        stats = manager.get_cache_stats()

        assert stats["total_tenants"] == 2
        assert stats["total_schemas"] == 3
        assert "acme" in stats["tenants"]
        assert "startup" in stats["tenants"]


class TestSchemaLoader:
    """Test schema loader integration"""

    def test_list_available_base_schemas(self):
        """Test listing available base schema templates via SchemaLoader"""
        mock_schema_loader = MagicMock()
        mock_schema_loader.list_schemas.return_value = [
            "video_colpali_smol500_mv_frame",
            "video_videoprism_base_mv_chunk_30s",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        manager = TenantSchemaManager(
            backend_name="vespa",
            backend_url="http://localhost",
            backend_port=19071,
            http_port=8080,
            config_manager=MagicMock(),
            schema_loader=mock_schema_loader
        )

        result = manager.list_available_base_schemas()

        assert "video_colpali_smol500_mv_frame" in result
        assert "video_videoprism_base_mv_chunk_30s" in result
        assert "video_colqwen_omni_mv_chunk_30s" in result
        mock_schema_loader.list_schemas.assert_called_once()


class TestSchemaValidation:
    """Test schema validation"""

    def test_validate_tenant_schema_exists(self):
        """Test validation delegates to backend"""
        mock_backend = MagicMock()
        mock_backend.schema_exists.return_value = True

        with patch("cogniverse_core.backends.tenant_schema_manager.BackendRegistry") as mock_registry:
            mock_registry.get_instance.return_value.get_search_backend.return_value = mock_backend

            manager = TenantSchemaManager(
                backend_name="vespa",
                backend_url="http://localhost",
                backend_port=19071,
                http_port=8080,
                config_manager=MagicMock(),
                schema_loader=MagicMock()
            )

            result = manager.validate_tenant_schema("acme", "video_colpali", MagicMock())

        assert result is True
        mock_backend.schema_exists.assert_called_once()


class TestDeleteSchemas:
    """Test schema deletion"""

    def test_delete_tenant_schemas(self):
        """Test deleting schemas delegates to backend and clears cache"""
        mock_backend = MagicMock()
        mock_backend.delete_schema.return_value = ["video_colpali_acme", "video_videoprism_acme"]

        with patch("cogniverse_core.backends.tenant_schema_manager.BackendRegistry") as mock_registry:
            mock_registry.get_instance.return_value.get_search_backend.return_value = mock_backend

            manager = TenantSchemaManager(
                backend_name="vespa",
                backend_url="http://localhost",
                backend_port=19071,
                http_port=8080,
                config_manager=MagicMock(),
                schema_loader=MagicMock()
            )
            manager.clear_cache()

            manager._cache_deployed_schema("acme", "video_colpali")
            manager._cache_deployed_schema("acme", "video_videoprism")

            deleted = manager.delete_tenant_schemas("acme")

        # Cache should be cleared
        assert "acme" not in manager._deployed_schemas
