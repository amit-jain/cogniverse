"""
Integration Tests for Profile Management API

Tests backend profile CRUD operations and schema deployment.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.async_polling import wait_for_vespa_indexing


@pytest.mark.integration
class TestProfileAPICRUD:
    """Integration tests for profile CRUD operations (no Vespa required)"""

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
                    {"name": "title", "type": "string", "indexing": ["summary"]},
                ]
            },
        }

        with open(schema_dir / "video_test_schema.json", "w") as f:
            json.dump(video_schema, f)

        # Create colpali schema template (used in example)
        colpali_schema = {
            "name": "video_colpali",
            "document": {
                "fields": [
                    {"name": "id", "type": "string", "indexing": ["summary"]},
                    {
                        "name": "patches",
                        "type": "tensor<float>(patch[1024], x[128])",
                        "indexing": ["attribute"],
                    },
                ]
            },
        }

        with open(schema_dir / "video_colpali_smol500_mv_frame_schema.json", "w") as f:
            json.dump(colpali_schema, f)

        return schema_dir

    @pytest.fixture
    def test_client(self, temp_schema_dir: Path, tmp_path: Path):
        """Create test client for profile API with properly configured test instances."""
        from cogniverse_core.config.manager import ConfigManager
        from cogniverse_core.validation.profile_validator import ProfileValidator
        from cogniverse_runtime.routers import admin

        # Create temporary database for ConfigManager
        temp_db = tmp_path / "test_config.db"
        config_manager = ConfigManager(db_path=temp_db)

        # Create ProfileValidator with temp schema directory
        profile_validator = ProfileValidator(
            config_manager=config_manager, schema_templates_dir=temp_schema_dir
        )

        # Reset and set global instances in admin router
        admin._config_manager = config_manager
        admin._profile_validator = profile_validator

        try:
            # Import app and create client
            from cogniverse_runtime.main import app

            client = TestClient(app)
            yield client
        finally:
            # Reset globals after test
            admin._config_manager = None
            admin._profile_validator = None

    def test_create_profile_minimal(self, test_client: TestClient):
        """Test creating a profile with minimal required fields."""
        profile_data = {
            "profile_name": "test_minimal",
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        response = test_client.post("/admin/profiles", json=profile_data)
        assert response.status_code == 201

        data = response.json()
        assert data["profile_name"] == "test_minimal"
        assert data["tenant_id"] == "test_tenant"
        assert data["schema_deployed"] is False
        assert data["tenant_schema_name"] is None
        assert "created_at" in data
        assert data["version"] == 1

    def test_create_profile_full(self, test_client: TestClient):
        """Test creating a profile with all fields."""

        profile_data = {
            "profile_name": "test_full",
            "tenant_id": "test_tenant",
            "type": "video",
            "description": "Full test profile",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "pipeline_config": {"keyframe_fps": 30.0, "transcribe_audio": True},
            "strategies": {
                "segmentation": {
                    "class": "FrameSegmentationStrategy",
                    "params": {"fps": 30.0},
                }
            },
            "embedding_type": "frame_based",
            "schema_config": {"embedding_dim": 128, "num_patches": 1024},
            "model_specific": {"batch_size": 32},
            "deploy_schema": False,
        }

        # Mock strategy class check to return True
        with patch(
            "cogniverse_core.validation.profile_validator.ProfileValidator._strategy_class_exists",
            return_value=True,
        ):
            response = test_client.post("/admin/profiles", json=profile_data)
            assert response.status_code == 201

            data = response.json()
            assert data["profile_name"] == "test_full"
            assert data["tenant_id"] == "test_tenant"

    def test_create_profile_validation_errors(self, test_client: TestClient):
        """Test profile creation with validation errors."""
        # Invalid profile name (spaces not allowed)
        invalid_data = {
            "profile_name": "invalid name",
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        response = test_client.post("/admin/profiles", json=invalid_data)
        assert response.status_code == 400
        assert "validation failed" in response.json()["detail"]["message"].lower()
        assert "errors" in response.json()["detail"]

    def test_create_profile_missing_schema_template(self, test_client: TestClient):
        """Test profile creation with nonexistent schema template."""
        profile_data = {
            "profile_name": "test_missing_schema",
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "nonexistent_schema",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        response = test_client.post("/admin/profiles", json=profile_data)
        assert response.status_code == 400
        assert "Schema template not found" in str(response.json()["detail"]["errors"])

    def test_create_duplicate_profile(self, test_client: TestClient):
        """Test creating a duplicate profile."""
        profile_data = {
            "profile_name": "test_duplicate",
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        # Create first profile
        response1 = test_client.post("/admin/profiles", json=profile_data)
        assert response1.status_code == 201

        # Try to create duplicate
        response2 = test_client.post("/admin/profiles", json=profile_data)
        assert response2.status_code == 400
        assert "already exists" in str(response2.json()["detail"]["errors"])

    def test_list_profiles_empty(self, test_client: TestClient):
        """Test listing profiles when none exist."""
        response = test_client.get("/admin/profiles?tenant_id=empty_tenant")
        assert response.status_code == 200

        data = response.json()
        assert data["profiles"] == []
        assert data["total_count"] == 0
        assert data["tenant_id"] == "empty_tenant"

    def test_list_profiles_with_data(self, test_client: TestClient):
        """Test listing profiles after creating some."""
        tenant_id = "test_list_tenant"

        # Create multiple profiles
        for i in range(3):
            profile_data = {
                "profile_name": f"test_profile_{i}",
                "tenant_id": tenant_id,
                "type": "video",
                "description": f"Test profile {i}",
                "schema_name": "video_test",
                "embedding_model": "vidore/colsmol-500m",
                "embedding_type": "frame_based",
            }
            response = test_client.post("/admin/profiles", json=profile_data)
            assert response.status_code == 201

        # List profiles
        response = test_client.get(f"/admin/profiles?tenant_id={tenant_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["total_count"] == 3
        assert len(data["profiles"]) == 3
        assert data["tenant_id"] == tenant_id

        # Check profile summaries
        profile_names = {p["profile_name"] for p in data["profiles"]}
        assert profile_names == {"test_profile_0", "test_profile_1", "test_profile_2"}

        # Check all summaries have required fields
        for profile in data["profiles"]:
            assert "profile_name" in profile
            assert "type" in profile
            assert "description" in profile
            assert "schema_name" in profile
            assert "embedding_model" in profile
            assert "schema_deployed" in profile
            assert "created_at" in profile

    def test_get_profile(self, test_client: TestClient):
        """Test getting a specific profile."""

        # Create profile
        profile_data = {
            "profile_name": "test_get",
            "tenant_id": "test_tenant",
            "type": "video",
            "description": "Test get profile",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "pipeline_config": {"keyframe_fps": 30.0},
            "strategies": {"segmentation": {"class": "FrameSegmentationStrategy"}},
            "embedding_type": "frame_based",
            "schema_config": {"embedding_dim": 128},
        }

        # Mock strategy class check to return True
        with patch(
            "cogniverse_core.validation.profile_validator.ProfileValidator._strategy_class_exists",
            return_value=True,
        ):
            create_response = test_client.post("/admin/profiles", json=profile_data)
            assert create_response.status_code == 201

        # Get profile
        response = test_client.get("/admin/profiles/test_get?tenant_id=test_tenant")
        assert response.status_code == 200

        data = response.json()
        assert data["profile_name"] == "test_get"
        assert data["tenant_id"] == "test_tenant"
        assert data["type"] == "video"
        assert data["description"] == "Test get profile"
        assert data["schema_name"] == "video_test"
        assert data["embedding_model"] == "vidore/colsmol-500m"
        assert data["pipeline_config"] == {"keyframe_fps": 30.0}
        assert data["strategies"] == {"segmentation": {"class": "FrameSegmentationStrategy"}}
        assert data["embedding_type"] == "frame_based"
        assert data["schema_config"] == {"embedding_dim": 128}
        assert data["schema_deployed"] is False

    def test_get_nonexistent_profile(self, test_client: TestClient):
        """Test getting a profile that doesn't exist."""
        response = test_client.get("/admin/profiles/nonexistent?tenant_id=test_tenant")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_profile(self, test_client: TestClient):
        """Test updating a profile."""
        # Create profile
        profile_data = {
            "profile_name": "test_update",
            "tenant_id": "test_tenant",
            "type": "video",
            "description": "Original description",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "pipeline_config": {"keyframe_fps": 30.0},
            "embedding_type": "frame_based",
        }

        create_response = test_client.post("/admin/profiles", json=profile_data)
        assert create_response.status_code == 201

        # Update profile
        update_data = {
            "tenant_id": "test_tenant",
            "description": "Updated description",
            "pipeline_config": {"keyframe_fps": 60.0, "transcribe_audio": True},
        }

        response = test_client.put("/admin/profiles/test_update", json=update_data)
        assert response.status_code == 200

        data = response.json()
        assert data["profile_name"] == "test_update"
        assert data["tenant_id"] == "test_tenant"
        assert set(data["updated_fields"]) == {"description", "pipeline_config"}
        assert data["version"] == 2

        # Verify update by getting profile
        get_response = test_client.get("/admin/profiles/test_update?tenant_id=test_tenant")
        get_data = get_response.json()
        assert get_data["description"] == "Updated description"
        assert get_data["pipeline_config"] == {"keyframe_fps": 60.0, "transcribe_audio": True}

    def test_update_immutable_fields(self, test_client: TestClient):
        """
        Test that immutable fields cannot be updated.

        Note: ProfileUpdateRequest Pydantic model doesn't include immutable fields,
        so attempts to update them are ignored (filtered out), resulting in "No fields to update".
        This is a design choice - the API prevents immutable field updates at the schema level.
        """
        # Create profile
        profile_data = {
            "profile_name": "test_immutable",
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        create_response = test_client.post("/admin/profiles", json=profile_data)
        assert create_response.status_code == 201

        # Try to update schema_name (immutable field not in ProfileUpdateRequest)
        # Pydantic will ignore it, leaving no fields to update
        update_data = {"tenant_id": "test_tenant", "schema_name": "different_schema"}

        response = test_client.put("/admin/profiles/test_immutable", json=update_data)
        assert response.status_code == 400
        assert "No fields to update" in response.json()["detail"]

    def test_update_nonexistent_profile(self, test_client: TestClient):
        """Test updating a profile that doesn't exist."""
        update_data = {"tenant_id": "test_tenant", "description": "New description"}

        response = test_client.put("/admin/profiles/nonexistent", json=update_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_profile(self, test_client: TestClient):
        """Test deleting a profile."""
        # Create profile
        profile_data = {
            "profile_name": "test_delete",
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        create_response = test_client.post("/admin/profiles", json=profile_data)
        assert create_response.status_code == 201

        # Delete profile
        response = test_client.delete("/admin/profiles/test_delete?tenant_id=test_tenant")
        assert response.status_code == 200

        data = response.json()
        assert data["profile_name"] == "test_delete"
        assert data["tenant_id"] == "test_tenant"
        assert data["schema_deleted"] is False
        assert "deleted_at" in data

        # Verify deletion
        get_response = test_client.get("/admin/profiles/test_delete?tenant_id=test_tenant")
        assert get_response.status_code == 404

    def test_delete_nonexistent_profile(self, test_client: TestClient):
        """Test deleting a profile that doesn't exist."""
        response = test_client.delete("/admin/profiles/nonexistent?tenant_id=test_tenant")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_tenant_isolation(self, test_client: TestClient):
        """Test that profiles are isolated by tenant."""
        # Create profile for tenant1
        profile_data_1 = {
            "profile_name": "shared_name",
            "tenant_id": "tenant1",
            "type": "video",
            "description": "Tenant 1 profile",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        response1 = test_client.post("/admin/profiles", json=profile_data_1)
        assert response1.status_code == 201

        # Create profile with same name for tenant2
        profile_data_2 = {
            "profile_name": "shared_name",
            "tenant_id": "tenant2",
            "type": "video",
            "description": "Tenant 2 profile",
            "schema_name": "video_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
        }

        response2 = test_client.post("/admin/profiles", json=profile_data_2)
        assert response2.status_code == 201

        # Get profiles for each tenant
        get1 = test_client.get("/admin/profiles/shared_name?tenant_id=tenant1")
        get2 = test_client.get("/admin/profiles/shared_name?tenant_id=tenant2")

        assert get1.status_code == 200
        assert get2.status_code == 200
        assert get1.json()["description"] == "Tenant 1 profile"
        assert get2.json()["description"] == "Tenant 2 profile"

        # List profiles for tenant1
        list1 = test_client.get("/admin/profiles?tenant_id=tenant1")
        assert list1.json()["total_count"] == 1

        # List profiles for tenant2
        list2 = test_client.get("/admin/profiles?tenant_id=tenant2")
        assert list2.json()["total_count"] == 1


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestProfileAPISchemaDeployment:
    """Integration tests for schema deployment (requires Vespa)"""

    @pytest.fixture(scope="class")
    def vespa_backend(self):
        """Start Vespa Docker container for schema deployment tests."""
        manager = VespaTestManager(app_name="test-profile-schema", http_port=8085)

        if not manager.setup_application_directory():
            pytest.skip("Failed to setup application directory")

        if not manager.deploy_test_application():
            pytest.skip("Failed to deploy Vespa test application")

        yield manager
        manager.cleanup()

    @pytest.fixture
    def test_client(self, vespa_backend, tmp_path: Path):
        """Create test client with Vespa backend."""

        from cogniverse_core.config.manager import ConfigManager
        from cogniverse_core.validation.profile_validator import ProfileValidator
        from cogniverse_runtime.backends.registry import BackendRegistry
        from cogniverse_runtime.routers import admin

        wait_for_vespa_indexing(delay=1, description="Vespa startup")

        # Create schema directory with test schema
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()

        video_schema = {
            "name": "video_deploy_test",
            "document": {
                "fields": [
                    {"name": "id", "type": "string", "indexing": ["summary"]},
                    {
                        "name": "embedding",
                        "type": "tensor<float>(x[128])",
                        "indexing": ["attribute"],
                    },
                ]
            },
        }

        with open(schema_dir / "video_deploy_test_schema.json", "w") as f:
            json.dump(video_schema, f)

        # Mock config to use test Vespa
        mock_config = {
            "vespa_url": "http://localhost",
            "vespa_port": vespa_backend.http_port,
            "vespa_config_port": vespa_backend.config_port,
        }

        with patch("cogniverse_core.config.utils.load_config_file", return_value=mock_config):
            # Create temporary database for ConfigManager
            temp_db = tmp_path / "test_config.db"
            config_manager = ConfigManager(db_path=temp_db)

            # Create ProfileValidator with temp schema directory
            profile_validator = ProfileValidator(
                config_manager=config_manager, schema_templates_dir=schema_dir
            )

            # Reset and set global instances in admin router
            admin._config_manager = config_manager
            admin._profile_validator = profile_validator

            # Reset backend registry to pick up new Vespa config
            BackendRegistry._instance = None

            try:
                from cogniverse_runtime.main import app

                client = TestClient(app)
                yield client
            finally:
                # Reset globals after test
                admin._config_manager = None
                admin._profile_validator = None
                BackendRegistry._instance = None

    def test_create_profile_with_schema_deployment(self, test_client: TestClient):
        """Test creating a profile and deploying its schema."""
        profile_data = {
            "profile_name": "test_with_deploy",
            "tenant_id": "deploy_tenant",
            "type": "video",
            "schema_name": "video_deploy_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
            "deploy_schema": True,
        }

        response = test_client.post("/admin/profiles", json=profile_data)
        assert response.status_code == 201

        data = response.json()
        assert data["schema_deployed"] is True
        assert data["tenant_schema_name"] is not None
        assert "deploy_tenant" in data["tenant_schema_name"]

    def test_deploy_schema_for_existing_profile(self, test_client: TestClient):
        """Test deploying schema for an existing profile."""
        # Create profile without schema deployment
        profile_data = {
            "profile_name": "test_deploy_later",
            "tenant_id": "deploy_tenant",
            "type": "video",
            "schema_name": "video_deploy_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
            "deploy_schema": False,
        }

        create_response = test_client.post("/admin/profiles", json=profile_data)
        assert create_response.status_code == 201
        assert create_response.json()["schema_deployed"] is False

        # Deploy schema
        deploy_data = {"tenant_id": "deploy_tenant", "force": False}

        response = test_client.post("/admin/profiles/test_deploy_later/deploy", json=deploy_data)
        assert response.status_code == 200

        data = response.json()
        assert data["deployment_status"] == "success"
        assert data["schema_name"] == "video_deploy_test"
        assert "deploy_tenant" in data["tenant_schema_name"]

    def test_deploy_schema_already_deployed(self, test_client: TestClient):
        """Test deploying a schema that's already deployed."""
        # Create and deploy profile
        profile_data = {
            "profile_name": "test_already_deployed",
            "tenant_id": "deploy_tenant",
            "type": "video",
            "schema_name": "video_deploy_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
            "deploy_schema": True,
        }

        test_client.post("/admin/profiles", json=profile_data)

        # Try to deploy again without force
        deploy_data = {"tenant_id": "deploy_tenant", "force": False}

        response = test_client.post("/admin/profiles/test_already_deployed/deploy", json=deploy_data)
        assert response.status_code == 200

        data = response.json()
        assert data["deployment_status"] == "already_deployed"

    def test_force_redeploy_schema(self, test_client: TestClient):
        """Test force redeploying a schema."""
        # Create and deploy profile
        profile_data = {
            "profile_name": "test_force_redeploy",
            "tenant_id": "deploy_tenant",
            "type": "video",
            "schema_name": "video_deploy_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
            "deploy_schema": True,
        }

        test_client.post("/admin/profiles", json=profile_data)

        # Force redeploy
        deploy_data = {"tenant_id": "deploy_tenant", "force": True}

        response = test_client.post("/admin/profiles/test_force_redeploy/deploy", json=deploy_data)
        assert response.status_code == 200

        data = response.json()
        assert data["deployment_status"] == "success"

    def test_end_to_end_schema_deployment_and_ingestion(self, test_client: TestClient, vespa_backend):
        """
        Comprehensive end-to-end test:
        1. Create profile and deploy schema via API
        2. Verify schema exists in Vespa
        3. Ingest a sample document
        4. Query for the document and verify results
        """
        from cogniverse_runtime.backends.registry import BackendRegistry
        from pyvespa import Vespa

        tenant_id = "e2e_tenant"
        profile_name = "e2e_test_profile"

        # Step 1: Create profile and deploy schema via API
        profile_data = {
            "profile_name": profile_name,
            "tenant_id": tenant_id,
            "type": "video",
            "description": "E2E test profile",
            "schema_name": "video_deploy_test",
            "embedding_model": "vidore/colsmol-500m",
            "embedding_type": "frame_based",
            "schema_config": {"embedding_dim": 128},
            "deploy_schema": True,
        }

        create_response = test_client.post("/admin/profiles", json=profile_data)
        assert create_response.status_code == 201
        assert create_response.json()["schema_deployed"] is True

        tenant_schema_name = create_response.json()["tenant_schema_name"]
        assert tenant_schema_name is not None

        # Step 2: Verify schema exists in Vespa
        backend_registry = BackendRegistry.get_instance()
        vespa_backend_obj = backend_registry.get_backend("vespa", tenant_id=tenant_id)
        assert vespa_backend_obj is not None

        schema_exists = vespa_backend_obj.schema_exists(
            schema_name="video_deploy_test", tenant_id=tenant_id
        )
        assert schema_exists is True

        # Get Vespa client directly
        vespa_app = Vespa(
            url=f"http://localhost:{vespa_backend.http_port}",
            application_package=vespa_backend_obj.app_package,
        )

        # Step 3: Ingest a sample document
        sample_doc = {
            "id": "test_video_001",
            "fields": {
                "id": "test_video_001",
                "embedding": [0.1] * 128,  # 128-dim embedding
            },
        }

        # Feed document
        feed_response = vespa_app.feed_data_point(
            schema=tenant_schema_name,
            data_id=sample_doc["id"],
            fields=sample_doc["fields"],
        )

        # Verify feed was successful
        assert feed_response.status_code in [200, 201]

        # Wait for indexing
        wait_for_vespa_indexing(delay=2, description="Document indexing")

        # Step 4: Query for the document and verify results
        yql_query = f"select * from sources {tenant_schema_name} where true"
        query_response = vespa_app.query(yql=yql_query)

        # Verify query succeeded
        assert query_response.status_code == 200
        query_json = query_response.json

        # Verify document was found
        assert "root" in query_json
        assert "children" in query_json["root"]
        assert len(query_json["root"]["children"]) >= 1

        # Verify document content
        found_doc = None
        for child in query_json["root"]["children"]:
            if child["fields"]["id"] == "test_video_001":
                found_doc = child
                break

        assert found_doc is not None, "Ingested document not found in query results"
        assert found_doc["fields"]["id"] == "test_video_001"

        # Verify the profile GET endpoint reflects deployed status
        get_response = test_client.get(f"/admin/profiles/{profile_name}?tenant_id={tenant_id}")
        assert get_response.status_code == 200
        get_data = get_response.json()
        assert get_data["schema_deployed"] is True
        assert get_data["tenant_schema_name"] == tenant_schema_name
