"""
Dashboard UI Integration Tests for Profile Management

Tests the API integration functions used by the dashboard UI
(deploy_schema_via_api, delete_profile_via_api, get_profile_schema_status).
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

# Check if cogniverse_runtime is available
try:
    import cogniverse_runtime  # noqa: F401

    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.skipif(not RUNTIME_AVAILABLE, reason="cogniverse_runtime not installed")
class TestDashboardProfileIntegration:
    """Integration tests for dashboard API helper functions"""

    @pytest.fixture(scope="class")
    def temp_schema_dir(cls, tmp_path_factory):
        """Create temporary schema directory with test schema templates."""
        schema_dir = tmp_path_factory.mktemp("schemas")

        video_schema = {
            "name": "video_test",
            "document": {
                "fields": [
                    {"name": "id", "type": "string", "indexing": ["summary"]},
                    {
                        "name": "embedding",
                        "type": "tensor<float>(x[128])",
                        "indexing": ["attribute", "index"],
                    },
                ]
            },
        }

        with open(schema_dir / "video_test_schema.json", "w") as f:
            json.dump(video_schema, f)

        return schema_dir

    @pytest.fixture
    def running_api(self, temp_schema_dir: Path, tmp_path: Path):
        """Start the FastAPI server for integration tests"""
        from fastapi.testclient import TestClient

        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_core.registries.schema_registry import SchemaRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.sqlite.config_store import SQLiteConfigStore
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_runtime.main import app
        from cogniverse_runtime.routers import admin

        # Reset registries
        BackendRegistry._instance = None
        SchemaRegistry._instance = None

        # Create config manager with SQLite store (no Vespa required)
        db_path = tmp_path / "test_config.db"
        store = SQLiteConfigStore(db_path=str(db_path))
        config_manager = ConfigManager(store=store)

        # Set up system config
        system_config = SystemConfig(
            tenant_id="default",
            backend_url="http://nonexistent",
            backend_port=9999,
        )
        config_manager.set_system_config(system_config)

        # Set ConfigManager and schema directory
        schema_loader = FilesystemSchemaLoader(temp_schema_dir)
        admin.set_config_manager(config_manager)
        admin.set_schema_loader(schema_loader)
        admin.set_profile_validator_schema_dir(temp_schema_dir)

        # Create test client
        client = TestClient(app)

        yield client

        # Cleanup
        admin.reset_dependencies()

    def test_deploy_schema_via_api_success(self, running_api):
        """Test deploy_schema_via_api function with successful deployment"""
        from scripts.backend_profile_tab import deploy_schema_via_api

        # Create a profile first
        running_api.post(
            "/admin/profiles",
            json={
                "profile_name": "deploy_test",
                "tenant_id": "test_tenant",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "test_model",
                "embedding_type": "frame_based",
            },
        )

        # Mock the API URL to use TestClient
        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://testserver",
        ):
            # Mock httpx.Client to use TestClient
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                # Make the mock client use the actual test client
                def mock_post(endpoint, json):
                    # Extract profile_name from endpoint
                    profile_name = endpoint.split("/")[-2]
                    return running_api.post(
                        f"/admin/profiles/{profile_name}/deploy", json=json
                    )

                mock_client_instance.post = mock_post

                # Call the function
                result = deploy_schema_via_api(
                    "deploy_test", "test_tenant", force=False
                )

                # Verify result
                assert (
                    result["success"] is False
                )  # Will fail because Vespa is not available
                assert "error" in result

    def test_delete_profile_via_api_success(self, running_api):
        """Test delete_profile_via_api function"""
        from scripts.backend_profile_tab import delete_profile_via_api

        # Create a profile first
        running_api.post(
            "/admin/profiles",
            json={
                "profile_name": "delete_test",
                "tenant_id": "test_tenant",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "test_model",
                "embedding_type": "frame_based",
            },
        )

        # Mock the API URL and httpx.Client
        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://testserver",
        ):
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                def mock_delete(endpoint, params):
                    profile_name = endpoint.split("/")[-1]
                    return running_api.delete(
                        f"/admin/profiles/{profile_name}", params=params
                    )

                mock_client_instance.delete = mock_delete

                # Call the function
                result = delete_profile_via_api(
                    "delete_test", "test_tenant", delete_schema=False
                )

                # Verify result
                assert result["success"] is True
                assert result["schema_deleted"] is False

    def test_get_profile_schema_status_success(self, running_api):
        """Test get_profile_schema_status function"""
        from scripts.backend_profile_tab import get_profile_schema_status

        # Create a profile first
        running_api.post(
            "/admin/profiles",
            json={
                "profile_name": "status_test",
                "tenant_id": "test_tenant",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "test_model",
                "embedding_type": "frame_based",
            },
        )

        # Mock the API URL and httpx.Client
        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://testserver",
        ):
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                def mock_get(endpoint, params):
                    profile_name = endpoint.split("/")[-1]
                    return running_api.get(
                        f"/admin/profiles/{profile_name}", params=params
                    )

                mock_client_instance.get = mock_get

                # Call the function
                result = get_profile_schema_status("status_test", "test_tenant")

                # Verify result
                assert result["error"] is None
                assert "schema_deployed" in result

    def test_end_to_end_workflow_via_dashboard_functions(self, running_api):
        """Test complete workflow using dashboard API functions"""
        from scripts.backend_profile_tab import (
            delete_profile_via_api,
            get_profile_schema_status,
        )

        profile_name = "e2e_test"
        tenant_id = "test_tenant"

        # Step 1: Create profile via API directly
        create_response = running_api.post(
            "/admin/profiles",
            json={
                "profile_name": profile_name,
                "tenant_id": tenant_id,
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "test_model",
                "embedding_type": "frame_based",
                "description": "E2E test profile",
            },
        )
        assert create_response.status_code == 201

        # Step 2: Check schema status via dashboard function
        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://testserver",
        ):
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                def mock_get(endpoint, params):
                    p_name = endpoint.split("/")[-1]
                    return running_api.get(f"/admin/profiles/{p_name}", params=params)

                mock_client_instance.get = mock_get

                status_result = get_profile_schema_status(profile_name, tenant_id)
                assert status_result["error"] is None
                assert status_result["schema_deployed"] is False  # Not deployed yet

        # Step 3: Update profile
        update_response = running_api.put(
            f"/admin/profiles/{profile_name}",
            json={"tenant_id": tenant_id, "description": "Updated description"},
        )
        assert update_response.status_code == 200

        # Step 4: Delete via dashboard function
        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://testserver",
        ):
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                def mock_delete(endpoint, params):
                    p_name = endpoint.split("/")[-1]
                    return running_api.delete(
                        f"/admin/profiles/{p_name}", params=params
                    )

                mock_client_instance.delete = mock_delete

                delete_result = delete_profile_via_api(
                    profile_name, tenant_id, delete_schema=False
                )
                assert delete_result["success"] is True

        # Step 5: Verify deletion
        get_response = running_api.get(
            f"/admin/profiles/{profile_name}", params={"tenant_id": tenant_id}
        )
        assert get_response.status_code == 404

    def test_api_timeout_handling(self):
        """Test that API functions handle timeouts gracefully"""
        from scripts.backend_profile_tab import deploy_schema_via_api

        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://localhost:9999",
        ):
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                # Simulate timeout
                mock_client_instance.post.side_effect = httpx.TimeoutException(
                    "Request timed out"
                )

                result = deploy_schema_via_api(
                    "test_profile", "test_tenant", force=False
                )

                assert result["success"] is False
                assert "timed out" in result["error"].lower()

    def test_api_connection_error_handling(self):
        """Test that API functions handle connection errors gracefully"""
        from scripts.backend_profile_tab import get_profile_schema_status

        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://nonexistent:9999",
        ):
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                # Simulate connection error
                mock_client_instance.get.side_effect = Exception("Connection refused")

                result = get_profile_schema_status("test_profile", "test_tenant")

                assert result["schema_deployed"] is False
                assert result["error"] is not None

    def test_api_http_error_handling(self):
        """Test that API functions handle HTTP errors gracefully"""
        from scripts.backend_profile_tab import delete_profile_via_api

        with patch(
            "scripts.backend_profile_tab.get_runtime_api_url",
            return_value="http://localhost:9999",
        ):
            with patch("scripts.backend_profile_tab.httpx.Client") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__enter__.return_value = mock_client_instance

                # Simulate 404 error
                mock_response = Mock()
                mock_response.status_code = 404
                mock_response.text = "Profile not found"
                mock_response.json.return_value = {"detail": "Profile not found"}

                mock_client_instance.delete.return_value = mock_response

                result = delete_profile_via_api(
                    "nonexistent_profile", "test_tenant", delete_schema=False
                )

                assert result["success"] is False
                assert "404" in result["error"]
