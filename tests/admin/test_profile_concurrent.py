"""
Concurrent Operations Tests for Profile Management

Tests that concurrent profile operations don't cause race conditions,
data corruption, or version conflicts.
"""

import json
import threading
from pathlib import Path
from typing import List

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestProfileConcurrentOperations:
    """Tests for concurrent operations on profile management"""

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

    def test_concurrent_profile_creation(self, test_client):
        """Test that multiple threads can create different profiles simultaneously"""
        num_threads = 10
        results: List[dict] = []
        errors: List[Exception] = []

        def create_profile(thread_id: int):
            try:
                response = test_client.post("/admin/profiles", json={
                    "profile_name": f"concurrent_profile_{thread_id}",
                    "tenant_id": "test_tenant",
                    "type": "video",
                    "schema_name": "video_test",
                    "embedding_model": f"model_{thread_id}",
                    "embedding_type": "frame_based"
                })
                results.append({
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 201 else None
                })
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = [threading.Thread(target=create_profile, args=(i,)) for i in range(num_threads)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads

        # All should succeed
        success_count = sum(1 for r in results if r["status_code"] == 201)
        assert success_count == num_threads

        # Verify all profiles were created
        list_response = test_client.get("/admin/profiles", params={"tenant_id": "test_tenant"})
        assert list_response.status_code == 200
        profile_names = [p["profile_name"] for p in list_response.json()["profiles"]]

        for i in range(num_threads):
            assert f"concurrent_profile_{i}" in profile_names

    def test_concurrent_profile_updates(self, test_client):
        """Test that concurrent updates to same profile are handled correctly"""
        profile_name = "update_test_profile"

        # Create initial profile
        test_client.post("/admin/profiles", json={
            "profile_name": profile_name,
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "model",
            "embedding_type": "frame_based",
            "pipeline_config": {"initial": "value"}
        })

        num_threads = 5
        results: List[dict] = []
        errors: List[Exception] = []

        def update_profile(thread_id: int):
            try:
                response = test_client.put(f"/admin/profiles/{profile_name}", json={
                    "tenant_id": "test_tenant",
                    "pipeline_config": {f"thread_{thread_id}": f"value_{thread_id}"}
                })
                results.append({
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "version": response.json().get("version") if response.status_code == 200 else None
                })
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = [threading.Thread(target=update_profile, args=(i,)) for i in range(num_threads)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All updates should succeed
        success_count = sum(1 for r in results if r["status_code"] == 200)
        assert success_count == num_threads

        # Verify version increments are correct
        versions = [r["version"] for r in results if r["version"] is not None]
        assert len(set(versions)) == num_threads, "Versions should be unique"

        # Final profile should have one of the updates
        final_response = test_client.get(f"/admin/profiles/{profile_name}", params={"tenant_id": "test_tenant"})
        assert final_response.status_code == 200
        final_config = final_response.json()["pipeline_config"]

        # Should have exactly one thread's update
        thread_keys = [k for k in final_config.keys() if k.startswith("thread_")]
        assert len(thread_keys) >= 1, "At least one update should persist"

    def test_concurrent_same_profile_creation_different_tenants(self, test_client):
        """Test concurrent creation of same profile name across different tenants"""
        profile_name = "shared_name"
        num_tenants = 10
        results: List[dict] = []
        errors: List[Exception] = []

        def create_for_tenant(tenant_id: int):
            try:
                response = test_client.post("/admin/profiles", json={
                    "profile_name": profile_name,
                    "tenant_id": f"tenant_{tenant_id}",
                    "type": "video",
                    "schema_name": "video_test",
                    "embedding_model": f"model_{tenant_id}",
                    "embedding_type": "frame_based"
                })
                results.append({
                    "tenant_id": tenant_id,
                    "status_code": response.status_code
                })
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = [threading.Thread(target=create_for_tenant, args=(i,)) for i in range(num_tenants)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        success_count = sum(1 for r in results if r["status_code"] == 201)
        assert success_count == num_tenants

        # Verify each tenant has their own profile
        for i in range(num_tenants):
            response = test_client.get(f"/admin/profiles/{profile_name}", params={"tenant_id": f"tenant_{i}"})
            assert response.status_code == 200
            assert response.json()["embedding_model"] == f"model_{i}"

    def test_concurrent_read_while_update(self, test_client):
        """Test that reads can happen concurrently with updates"""
        profile_name = "read_update_test"

        # Create initial profile
        test_client.post("/admin/profiles", json={
            "profile_name": profile_name,
            "tenant_id": "test_tenant",
            "type": "video",
            "schema_name": "video_test",
            "embedding_model": "model",
            "embedding_type": "frame_based",
            "description": "Initial"
        })

        num_readers = 10
        num_updaters = 5
        results: List[dict] = []
        errors: List[Exception] = []

        def read_profile(thread_id: int):
            try:
                response = test_client.get(f"/admin/profiles/{profile_name}", params={"tenant_id": "test_tenant"})
                results.append({
                    "type": "read",
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "description": response.json().get("description") if response.status_code == 200 else None
                })
            except Exception as e:
                errors.append(e)

        def update_profile(thread_id: int):
            try:
                response = test_client.put(f"/admin/profiles/{profile_name}", json={
                    "tenant_id": "test_tenant",
                    "description": f"Updated by thread {thread_id}"
                })
                results.append({
                    "type": "update",
                    "thread_id": thread_id,
                    "status_code": response.status_code
                })
            except Exception as e:
                errors.append(e)

        # Create mixed threads
        threads = []
        for i in range(num_readers):
            threads.append(threading.Thread(target=read_profile, args=(i,)))
        for i in range(num_updaters):
            threads.append(threading.Thread(target=update_profile, args=(i,)))

        # Shuffle and start
        import random
        random.shuffle(threads)
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All reads should succeed
        read_results = [r for r in results if r["type"] == "read"]
        assert all(r["status_code"] == 200 for r in read_results)

        # All updates should succeed
        update_results = [r for r in results if r["type"] == "update"]
        assert all(r["status_code"] == 200 for r in update_results)

    def test_concurrent_list_operations(self, test_client):
        """Test that list operations are consistent during concurrent modifications"""
        tenant_id = "list_test_tenant"

        # Create initial profiles
        for i in range(5):
            test_client.post("/admin/profiles", json={
                "profile_name": f"initial_profile_{i}",
                "tenant_id": tenant_id,
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model",
                "embedding_type": "frame_based"
            })

        results: List[dict] = []
        errors: List[Exception] = []

        def list_profiles(thread_id: int):
            try:
                response = test_client.get("/admin/profiles", params={"tenant_id": tenant_id})
                results.append({
                    "type": "list",
                    "thread_id": thread_id,
                    "count": len(response.json()["profiles"]) if response.status_code == 200 else 0
                })
            except Exception as e:
                errors.append(e)

        def add_profile(thread_id: int):
            try:
                test_client.post("/admin/profiles", json={
                    "profile_name": f"added_profile_{thread_id}",
                    "tenant_id": tenant_id,
                    "type": "video",
                    "schema_name": "video_test",
                    "embedding_model": "model",
                    "embedding_type": "frame_based"
                })
                results.append({"type": "add", "thread_id": thread_id})
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=list_profiles, args=(i,)))
        for i in range(5):
            threads.append(threading.Thread(target=add_profile, args=(i,)))

        # Start all
        for thread in threads:
            thread.start()

        # Wait
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Final count should be consistent
        final_response = test_client.get("/admin/profiles", params={"tenant_id": tenant_id})
        final_count = len(final_response.json()["profiles"])

        # Should have 5 initial + 5 added = 10 profiles
        assert final_count == 10

    def test_concurrent_delete_operations(self, test_client):
        """Test concurrent deletion of different profiles"""
        tenant_id = "delete_test_tenant"
        num_profiles = 10

        # Create profiles
        for i in range(num_profiles):
            test_client.post("/admin/profiles", json={
                "profile_name": f"deletable_profile_{i}",
                "tenant_id": tenant_id,
                "type": "video",
                "schema_name": "video_test",
                "embedding_model": "model",
                "embedding_type": "frame_based"
            })

        results: List[dict] = []
        errors: List[Exception] = []

        def delete_profile(profile_id: int):
            try:
                response = test_client.delete(
                    f"/admin/profiles/deletable_profile_{profile_id}",
                    params={"tenant_id": tenant_id}
                )
                results.append({
                    "profile_id": profile_id,
                    "status_code": response.status_code
                })
            except Exception as e:
                errors.append(e)

        # Create delete threads
        threads = [threading.Thread(target=delete_profile, args=(i,)) for i in range(num_profiles)]

        # Start all
        for thread in threads:
            thread.start()

        # Wait
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All deletes should succeed
        assert all(r["status_code"] == 200 for r in results)

        # Verify all profiles are gone
        list_response = test_client.get("/admin/profiles", params={"tenant_id": tenant_id})
        assert list_response.status_code == 200
        assert len(list_response.json()["profiles"]) == 0
