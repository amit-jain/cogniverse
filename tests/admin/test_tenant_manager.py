"""
Integration tests for Tenant Management API.

Tests organization and tenant CRUD operations with Vespa backend.
"""

import logging

import pytest
from fastapi.testclient import TestClient

import cogniverse_vespa  # noqa: F401 - trigger Vespa backend self-registration
from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.async_polling import wait_for_vespa_indexing

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestTenantManagerAPI:
    """Integration tests for tenant management API"""

    @pytest.fixture(scope="module")
    def vespa_backend(self):
        """Start Vespa Docker container (NO schemas pre-deployed)"""
        manager = VespaTestManager(app_name="test-tenant-manager", http_port=8084)

        if not manager.setup_application_directory():
            pytest.skip("Failed to setup application directory")

        # Start Vespa WITHOUT deploying schemas - let SchemaRegistry handle it
        from tests.utils.vespa_docker import VespaDockerManager

        docker_mgr = VespaDockerManager()
        container_info = docker_mgr.start_container(
            module_name="tenant_manager_tests", use_module_ports=False
        )
        manager.container_name = container_info["container_name"]
        manager.http_port = container_info["http_port"]
        manager.config_port = container_info["config_port"]
        manager.docker_manager = docker_mgr

        docker_mgr.wait_for_config_ready(container_info)
        manager.is_deployed = True

        logger.info(f"Vespa ready on port {manager.http_port}")
        yield manager
        manager.cleanup()

    @pytest.fixture(scope="module")
    def shared_test_db(self, tmp_path_factory):
        """Create shared database for all tests in this module"""
        db_dir = tmp_path_factory.mktemp("test_db")
        db_path = db_dir / "test_tenant_config.db"
        return db_path

    @pytest.fixture(scope="module")
    def config_manager(self, vespa_backend, shared_test_db):
        """Create class-scoped ConfigManager"""
        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_vespa.config.config_store import VespaConfigStore

        wait_for_vespa_indexing(delay=1, description="Vespa startup")

        # Create ConfigManager with VespaConfigStore pointing to test Docker container
        store = VespaConfigStore(
            vespa_url="http://localhost",
            vespa_port=vespa_backend.http_port,
        )
        config_manager = ConfigManager(store=store)

        # FIRST: Deploy metadata schemas via backend initialization
        # This must happen BEFORE set_system_config() which writes to config_metadata schema
        schema_loader = FilesystemSchemaLoader(vespa_backend.test_schemas_resource_dir)
        logger.info(
            f"Creating backend for system tenant on port {vespa_backend.http_port}"
        )
        BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id="system",
            config={
                "backend": {
                    "url": "http://localhost",
                    "port": vespa_backend.http_port,
                    "config_port": vespa_backend.config_port,
                }
            },
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        logger.info("Backend created successfully - metadata schemas deployed")

        # Wait for Vespa to be ready and for schemas to be fully activated
        from tests.utils.vespa_health import wait_for_vespa_ready

        wait_for_vespa_ready(port=vespa_backend.http_port, max_timeout=30)

        # Additional wait for metadata schema activation
        wait_for_vespa_indexing(delay=3, description="metadata schema activation")

        # NOW set system config (schemas are deployed, config_metadata exists)
        logger.info(
            f"Setting up config with Vespa on port {vespa_backend.http_port} (config port {vespa_backend.config_port})"
        )
        system_config = SystemConfig(
            tenant_id="system",
            backend_url="http://localhost",
            backend_port=vespa_backend.http_port,
        )
        config_manager.set_system_config(system_config)
        logger.info(f"System config set: backend_port={system_config.backend_port}")

        yield config_manager

    @pytest.fixture
    def test_client(self, vespa_backend, config_manager):
        """Create function-scoped test client reusing backend from config_manager"""
        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.admin import tenant_manager

        # DO NOT clear BackendRegistry cache - reuse backend with deployed metadata schemas
        logger.info(
            "Reusing backend from config_manager fixture (with metadata schemas)"
        )

        schema_loader = FilesystemSchemaLoader(vespa_backend.test_schemas_resource_dir)

        # Get existing backend (which already has metadata schemas deployed)
        backend = BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id="system",
            config={
                "backend": {
                    "url": "http://localhost",
                    "port": vespa_backend.http_port,
                    "config_port": vespa_backend.config_port,
                }
            },
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Set up tenant_manager with fresh backend
        tenant_manager.backend = backend
        tenant_manager.set_config_manager(config_manager)
        tenant_manager.set_schema_loader(schema_loader)

        client = TestClient(tenant_manager.app)
        yield client

        # Cleanup after each test to prevent state leakage
        logger.info("Cleaning up test_client fixture")
        tenant_manager.backend = None
        tenant_manager._config_manager = None
        tenant_manager._schema_loader = None

        # Small delay to let Vespa settle between tests
        wait_for_vespa_indexing(delay=1, description="post-test cleanup")

    @pytest.mark.ci_fast
    def test_health_check(self):
        """Test health check endpoint without Vespa"""
        from cogniverse_runtime.admin.tenant_manager import app

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
        response1 = test_client.post(
            "/admin/organizations",
            json={
                "org_id": "duporg",
                "org_name": "Duplicate Org",
                "created_by": "test",
            },
        )
        assert (
            response1.status_code == 200
        ), f"First org creation failed: {response1.status_code} - {response1.text}"

        # Wait for Vespa to index the document
        wait_for_vespa_indexing(delay=6, description="organization indexing")

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

        wait_for_vespa_indexing(delay=7, description="multiple organization documents")

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

        wait_for_vespa_indexing(delay=4, description="organization indexing")

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

        # Wait for schema deployment to complete
        wait_for_vespa_indexing(delay=3, description="tenant schema deployment")

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

        wait_for_vespa_indexing(delay=2, description="organization and tenant indexing")

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

        # Wait for Vespa to index both tenant and organization documents
        wait_for_vespa_indexing(delay=5, description="tenant and organization indexing")

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

        wait_for_vespa_indexing(delay=3, description="multiple tenant documents")

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

        wait_for_vespa_indexing(delay=2, description="tenant indexing")

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

        wait_for_vespa_indexing(delay=2, description="tenant indexing")

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

        wait_for_vespa_indexing(delay=3, description="multiple tenant documents")

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
