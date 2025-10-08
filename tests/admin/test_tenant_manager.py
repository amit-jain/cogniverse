"""
Integration tests for Tenant Management API.

Tests organization and tenant CRUD operations with Vespa backend.
"""

import time

import pytest
from fastapi.testclient import TestClient

from tests.system.vespa_test_manager import VespaTestManager


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestTenantManagerAPI:
    """Integration tests for tenant management API"""

    @pytest.fixture(scope="class")
    def vespa_backend(self):
        """Start Vespa Docker container, deploy metadata schemas, yield, cleanup"""
        manager = VespaTestManager(app_name="test-tenant-manager", http_port=8084)
        yield manager
        manager.cleanup()

    @pytest.fixture
    def test_client(self, vespa_backend):
        """Create test client for tenant manager API"""
        import os

        # Set environment variables for Vespa configuration
        os.environ["VESPA_ENDPOINT"] = f"http://localhost:{vespa_backend.http_port}"
        os.environ["VESPA_PORT"] = str(vespa_backend.http_port)

        # Deploy metadata schemas
        from src.backends.vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            vespa_endpoint=f"http://localhost",
            vespa_port=vespa_backend.http_port,
        )
        schema_manager.upload_metadata_schemas()

        # Import app after setting env vars
        from src.admin.tenant_manager import app

        return TestClient(app)

    @pytest.mark.ci_fast
    def test_health_check(self):
        """Test health check endpoint without Vespa"""
        from src.admin.tenant_manager import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tenant_manager"

    def test_create_organization(self, test_client):
        """Test creating a new organization"""
        response = test_client.post(
            "/admin/organizations",
            json={"org_id": "testorg", "org_name": "Test Org", "created_by": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] == "testorg"
        assert data["org_name"] == "Test Org"
        assert data["status"] == "active"
        assert data["tenant_count"] == 0

    def test_create_duplicate_organization(self, test_client):
        """Test creating duplicate organization fails"""
        # Create first org
        test_client.post(
            "/admin/organizations",
            json={
                "org_id": "duporg",
                "org_name": "Duplicate Org",
                "created_by": "test",
            },
        )

        # Try to create same org again
        response = test_client.post(
            "/admin/organizations",
            json={
                "org_id": "duporg",
                "org_name": "Duplicate Org 2",
                "created_by": "test",
            },
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_list_organizations(self, test_client):
        """Test listing organizations"""
        # Create a few orgs
        test_client.post(
            "/admin/organizations",
            json={"org_id": "org1", "org_name": "Org 1", "created_by": "test"},
        )
        test_client.post(
            "/admin/organizations",
            json={"org_id": "org2", "org_name": "Org 2", "created_by": "test"},
        )

        time.sleep(1)  # Wait for Vespa indexing

        response = test_client.get("/admin/organizations")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 2
        assert len(data["organizations"]) >= 2

    def test_get_organization(self, test_client):
        """Test getting single organization"""
        # Create org
        test_client.post(
            "/admin/organizations",
            json={"org_id": "getorg", "org_name": "Get Org", "created_by": "test"},
        )

        time.sleep(1)  # Wait for Vespa indexing

        response = test_client.get("/admin/organizations/getorg")
        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] == "getorg"
        assert data["org_name"] == "Get Org"

    def test_get_nonexistent_organization(self, test_client):
        """Test getting nonexistent organization"""
        response = test_client.get("/admin/organizations/nonexistent")
        assert response.status_code == 404

    def test_create_tenant_with_org_tenant_format(self, test_client):
        """Test creating tenant with org:tenant format"""
        response = test_client.post(
            "/admin/tenants",
            json={"tenant_id": "acme:production", "created_by": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_full_id"] == "acme:production"
        assert data["org_id"] == "acme"
        assert data["tenant_name"] == "production"
        assert data["status"] == "active"
        assert len(data["schemas_deployed"]) > 0

    def test_create_tenant_auto_creates_org(self, test_client):
        """Test creating tenant auto-creates organization if doesn't exist"""
        response = test_client.post(
            "/admin/tenants",
            json={"tenant_id": "neworg:dev", "created_by": "test"},
        )
        assert response.status_code == 200

        time.sleep(1)  # Wait for Vespa indexing

        # Verify org was created
        org_response = test_client.get("/admin/organizations/neworg")
        assert org_response.status_code == 200
        org_data = org_response.json()
        assert org_data["org_id"] == "neworg"
        assert org_data["tenant_count"] > 0

    def test_create_duplicate_tenant(self, test_client):
        """Test creating duplicate tenant fails"""
        # Create first tenant
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "duptenant:prod", "created_by": "test"},
        )

        # Try to create same tenant again
        response = test_client.post(
            "/admin/tenants",
            json={"tenant_id": "duptenant:prod", "created_by": "test"},
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_list_tenants_for_org(self, test_client):
        """Test listing tenants for an organization"""
        # Create org with multiple tenants
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "listorg:dev", "created_by": "test"},
        )
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "listorg:staging", "created_by": "test"},
        )
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "listorg:prod", "created_by": "test"},
        )

        time.sleep(1)  # Wait for Vespa indexing

        response = test_client.get("/admin/organizations/listorg/tenants")
        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] == "listorg"
        assert data["total_count"] == 3
        assert len(data["tenants"]) == 3

        tenant_names = {t["tenant_name"] for t in data["tenants"]}
        assert tenant_names == {"dev", "staging", "prod"}

    def test_get_tenant(self, test_client):
        """Test getting single tenant"""
        # Create tenant
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "gettenant:test", "created_by": "test"},
        )

        time.sleep(1)  # Wait for Vespa indexing

        response = test_client.get("/admin/tenants/gettenant:test")
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_full_id"] == "gettenant:test"
        assert data["org_id"] == "gettenant"
        assert data["tenant_name"] == "test"

    def test_get_nonexistent_tenant(self, test_client):
        """Test getting nonexistent tenant"""
        response = test_client.get("/admin/tenants/nonexistent:tenant")
        assert response.status_code == 404

    def test_delete_tenant(self, test_client):
        """Test deleting tenant"""
        # Create tenant
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "deltenant:test", "created_by": "test"},
        )

        time.sleep(1)  # Wait for Vespa indexing

        # Delete tenant
        response = test_client.delete("/admin/tenants/deltenant:test")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["tenant_full_id"] == "deltenant:test"

        # Verify tenant is gone
        get_response = test_client.get("/admin/tenants/deltenant:test")
        assert get_response.status_code == 404

    def test_delete_organization(self, test_client):
        """Test deleting organization and all its tenants"""
        # Create org with tenants
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "delorg:dev", "created_by": "test"},
        )
        test_client.post(
            "/admin/tenants",
            json={"tenant_id": "delorg:prod", "created_by": "test"},
        )

        time.sleep(1)  # Wait for Vespa indexing

        # Delete org
        response = test_client.delete("/admin/organizations/delorg")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["org_id"] == "delorg"
        assert data["tenants_deleted"] == 2

        # Verify org is gone
        get_response = test_client.get("/admin/organizations/delorg")
        assert get_response.status_code == 404

    def test_invalid_tenant_id_format(self, test_client):
        """Test creating tenant with invalid format"""
        response = test_client.post(
            "/admin/tenants",
            json={"tenant_id": "invalid@tenant", "created_by": "test"},
        )
        assert response.status_code == 400

    def test_invalid_org_id_format(self, test_client):
        """Test creating org with invalid format"""
        response = test_client.post(
            "/admin/organizations",
            json={"org_id": "invalid@org", "org_name": "Invalid", "created_by": "test"},
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
