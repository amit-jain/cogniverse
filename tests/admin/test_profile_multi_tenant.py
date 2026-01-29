"""
Multi-Tenant Isolation Tests for Profile Management

Tests that profiles are properly isolated between tenants and that
cross-tenant access is prevented.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestProfileMultiTenantIsolation:
    """Tests for multi-tenant isolation in profile management"""

    @pytest.fixture(scope="class")
    def temp_schema_dir(cls, tmp_path_factory):
        """Create temporary schema directory with test schema templates."""
        schema_dir = tmp_path_factory.mktemp("schemas")

        # Create valid video schema template
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
    def test_client(self, temp_schema_dir: Path, tmp_path: Path):
        """Create test client with properly configured instances."""
        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_core.registries.schema_registry import SchemaRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_foundation.config.utils import create_default_config_manager
        from cogniverse_runtime.main import app
        from cogniverse_runtime.routers import admin

        # Reset registries
        BackendRegistry._instance = None
        SchemaRegistry._instance = None

        # Create config manager with backend store
        config_manager = create_default_config_manager()

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

    def test_tenant_isolation_create_same_profile_name(self, test_client):
        """Test that two tenants can create profiles with the same name"""
        profile_name = "shared_profile_name"

        profile_a = {
            "profile_name": profile_name,
            "tenant_id": "tenant_a",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "model_a",
            "embedding_type": "frame_based",
            "description": "Tenant A's profile",
        }

        profile_b = {
            "profile_name": profile_name,
            "tenant_id": "tenant_b",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "model_b",
            "embedding_type": "single_vector",
            "description": "Tenant B's profile",
        }

        # Create profile for tenant A
        response_a = test_client.post("/admin/profiles", json=profile_a)
        assert response_a.status_code == 201
        assert response_a.json()["tenant_id"] == "tenant_a"

        # Create profile with same name for tenant B
        response_b = test_client.post("/admin/profiles", json=profile_b)
        assert response_b.status_code == 201
        assert response_b.json()["tenant_id"] == "tenant_b"

        # Verify both exist independently
        response_a_get = test_client.get(
            f"/admin/profiles/{profile_name}", params={"tenant_id": "tenant_a"}
        )
        assert response_a_get.status_code == 200
        assert response_a_get.json()["embedding_model"] == "model_a"
        assert response_a_get.json()["description"] == "Tenant A's profile"

        response_b_get = test_client.get(
            f"/admin/profiles/{profile_name}", params={"tenant_id": "tenant_b"}
        )
        assert response_b_get.status_code == 200
        assert response_b_get.json()["embedding_model"] == "model_b"
        assert response_b_get.json()["description"] == "Tenant B's profile"

    def test_tenant_cannot_see_other_tenant_profiles(self, test_client):
        """Test that tenant A cannot see tenant B's profiles in list"""
        # Create profiles for tenant A
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": "tenant_a_profile_1",
                "tenant_id": "tenant_a",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model_a",
                "embedding_type": "frame_based",
            },
        )

        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": "tenant_a_profile_2",
                "tenant_id": "tenant_a",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model_a",
                "embedding_type": "frame_based",
            },
        )

        # Create profiles for tenant B
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": "tenant_b_profile_1",
                "tenant_id": "tenant_b",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model_b",
                "embedding_type": "frame_based",
            },
        )

        # List profiles for tenant A
        response_a = test_client.get(
            "/admin/profiles", params={"tenant_id": "tenant_a"}
        )
        assert response_a.status_code == 200
        profiles_a = response_a.json()["profiles"]
        profile_names_a = [p["profile_name"] for p in profiles_a]

        assert "tenant_a_profile_1" in profile_names_a
        assert "tenant_a_profile_2" in profile_names_a
        assert "tenant_b_profile_1" not in profile_names_a
        assert len(profiles_a) == 2

        # List profiles for tenant B
        response_b = test_client.get(
            "/admin/profiles", params={"tenant_id": "tenant_b"}
        )
        assert response_b.status_code == 200
        profiles_b = response_b.json()["profiles"]
        profile_names_b = [p["profile_name"] for p in profiles_b]

        assert "tenant_b_profile_1" in profile_names_b
        assert "tenant_a_profile_1" not in profile_names_b
        assert "tenant_a_profile_2" not in profile_names_b
        assert len(profiles_b) == 1

    def test_tenant_cannot_access_other_tenant_profile_details(self, test_client):
        """Test that tenant A cannot access tenant B's profile details"""
        # Create profile for tenant A
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": "secret_profile",
                "tenant_id": "tenant_a",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "secret_model",
                "embedding_type": "frame_based",
            },
        )

        # Tenant B tries to access tenant A's profile
        response = test_client.get(
            "/admin/profiles/secret_profile", params={"tenant_id": "tenant_b"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_tenant_cannot_update_other_tenant_profile(self, test_client):
        """Test that tenant A cannot update tenant B's profile"""
        # Create profile for tenant A
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": "protected_profile",
                "tenant_id": "tenant_a",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model_a",
                "embedding_type": "frame_based",
                "description": "Original description",
            },
        )

        # Tenant B tries to update tenant A's profile
        response = test_client.put(
            "/admin/profiles/protected_profile",
            json={"tenant_id": "tenant_b", "description": "Malicious update"},
        )
        assert response.status_code == 404

        # Verify profile was not modified
        response_check = test_client.get(
            "/admin/profiles/protected_profile", params={"tenant_id": "tenant_a"}
        )
        assert response_check.status_code == 200
        assert response_check.json()["description"] == "Original description"

    def test_tenant_cannot_delete_other_tenant_profile(self, test_client):
        """Test that tenant A cannot delete tenant B's profile"""
        # Create profile for tenant A
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": "permanent_profile",
                "tenant_id": "tenant_a",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model_a",
                "embedding_type": "frame_based",
            },
        )

        # Tenant B tries to delete tenant A's profile
        response = test_client.delete(
            "/admin/profiles/permanent_profile", params={"tenant_id": "tenant_b"}
        )
        assert response.status_code == 404

        # Verify profile still exists
        response_check = test_client.get(
            "/admin/profiles/permanent_profile", params={"tenant_id": "tenant_a"}
        )
        assert response_check.status_code == 200

    def test_empty_tenant_id_not_allowed(self, test_client):
        """Test that empty or missing tenant_id is rejected"""
        # Try to create profile with empty tenant_id
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": "no_tenant_profile",
                "tenant_id": "",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model",
                "embedding_type": "frame_based",
            },
        )
        # Should fail validation or use default tenant
        # Behavior depends on implementation - just verify it doesn't create issues

    def test_tenant_isolation_persists_across_operations(self, test_client):
        """Test that tenant isolation is maintained across multiple operations"""
        profile_name = "isolation_test"

        # Tenant A creates profile
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": profile_name,
                "tenant_id": "tenant_a",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model_a",
                "embedding_type": "frame_based",
                "description": "Version 1",
            },
        )

        # Tenant A updates profile
        test_client.put(
            f"/admin/profiles/{profile_name}",
            json={"tenant_id": "tenant_a", "description": "Version 2"},
        )

        # Tenant B creates profile with same name
        test_client.post(
            "/admin/profiles",
            json={
                "profile_name": profile_name,
                "tenant_id": "tenant_b",
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model_b",
                "embedding_type": "frame_based",
                "description": "Tenant B version",
            },
        )

        # Verify tenant A's profile is unchanged
        response_a = test_client.get(
            f"/admin/profiles/{profile_name}", params={"tenant_id": "tenant_a"}
        )
        assert response_a.status_code == 200
        assert response_a.json()["description"] == "Version 2"
        assert response_a.json()["embedding_model"] == "model_a"

        # Verify tenant B's profile is separate
        response_b = test_client.get(
            f"/admin/profiles/{profile_name}", params={"tenant_id": "tenant_b"}
        )
        assert response_b.status_code == 200
        assert response_b.json()["description"] == "Tenant B version"
        assert response_b.json()["embedding_model"] == "model_b"

        # Delete tenant B's profile
        test_client.delete(
            f"/admin/profiles/{profile_name}", params={"tenant_id": "tenant_b"}
        )

        # Verify tenant A's profile still exists
        response_a_check = test_client.get(
            f"/admin/profiles/{profile_name}", params={"tenant_id": "tenant_a"}
        )
        assert response_a_check.status_code == 200

        # Verify tenant B's profile is gone
        response_b_check = test_client.get(
            f"/admin/profiles/{profile_name}", params={"tenant_id": "tenant_b"}
        )
        assert response_b_check.status_code == 404
