"""
Integration test configuration for runtime integration tests.

Provides shared Vespa Docker instance with metadata schemas deployed,
plus ConfigManager, SchemaLoader, and FastAPI TestClient fixtures
wired with real dependencies (only QueryEncoder is mocked).
"""

import json
import logging
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_runtime.routers import health, search
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)

# Path to schema JSON files
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for runtime integration tests.

    Module-scoped to share across all tests in this module.
    Deploys metadata schemas so VespaConfigStore can be used immediately.

    Yields:
        dict: Vespa connection info with http_port, config_port, base_url, container_name
    """
    manager = VespaDockerManager()

    try:
        # Use default Vespa ports (8080/19071) so the merged config from
        # configs/config.json (backend.port=8080) matches the test container.
        container_info = manager.start_container(
            module_name="runtime_integration_tests",
            use_module_ports=False,
            http_port=8080,
            config_port=19071,
        )

        # Wait for config server
        manager.wait_for_config_ready(container_info, timeout=180)

        # Wait for internal services to initialize
        logger.info("Waiting 15 seconds for Vespa internal services to initialize...")
        time.sleep(15)

        # Deploy metadata + data schemas in a single application package.
        # Vespa rejects deployments that remove existing schemas, so all
        # schemas must be included together.
        from vespa.package import ApplicationPackage

        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]

        # Parse the data schema used by search tests.
        # Rename to include "_default" tenant suffix to match the multi-tenant
        # naming convention (BackendRegistry generates tenant-specific schema names).
        schema_file = SCHEMAS_DIR / "video_colpali_smol500_mv_frame_schema.json"
        with open(schema_file) as f:
            schema_json = json.load(f)
        schema_json["name"] = "video_colpali_smol500_mv_frame_default"
        schema_json["document"]["name"] = "video_colpali_smol500_mv_frame_default"
        parser = JsonSchemaParser()
        data_schema = parser.parse_schema(schema_json)

        all_schemas = metadata_schemas + [data_schema]
        app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager._deploy_package(app_package)
        logger.info("Deployed metadata + data schemas in single package")

        # Wait for application readiness after schema deployment
        manager.wait_for_application_ready(container_info, timeout=120)

        logger.info("Vespa initialization complete - ready for runtime integration tests")

        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        manager.stop_container()

        # Clear singleton state to avoid interference with other test modules
        try:
            BackendRegistry._instance = None
            BackendRegistry._backend_instances.clear()
        except Exception:
            pass


@pytest.fixture(scope="module")
def config_manager(vespa_instance):
    """
    ConfigManager backed by real VespaConfigStore.

    Seeds SystemConfig with backend_url/port pointing at the test Vespa container,
    and adds two test profiles for the default tenant.
    """
    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )

    cm = ConfigManager(store=store)

    # Seed system config pointing at the test Vespa
    system_config = SystemConfig(
        tenant_id="default",
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm.set_system_config(system_config)

    # Add test profiles for the default tenant
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_colpali",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="vidore/colpali-v1.2",
        ),
        tenant_id="default",
    )
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_videoprism",
            type="video",
            schema_name="video_videoprism_base_mv_chunk_30s",
            embedding_model="google/videoprism-base",
        ),
        tenant_id="default",
    )

    # Seed a second tenant with different profiles
    tenant_b_config = SystemConfig(
        tenant_id="tenant_b",
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm.set_system_config(tenant_b_config, tenant_id="tenant_b")

    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="tenant_b_profile",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="vidore/colpali-v1.2",
        ),
        tenant_id="tenant_b",
    )

    return cm


@pytest.fixture(scope="module")
def schema_loader():
    """FilesystemSchemaLoader from configs/schemas/."""
    return FilesystemSchemaLoader(SCHEMAS_DIR)


@pytest.fixture(scope="module")
def search_client(vespa_instance, config_manager, schema_loader):
    """
    FastAPI TestClient with search router wired to real ConfigManager and SchemaLoader.

    Only the search router is mounted (no lifespan needed).
    Dependency overrides point at the real Vespa-backed ConfigManager.
    """
    test_app = FastAPI()
    test_app.include_router(search.router, prefix="/search")

    test_app.dependency_overrides[search.get_config_manager_dependency] = (
        lambda: config_manager
    )
    test_app.dependency_overrides[search.get_schema_loader_dependency] = (
        lambda: schema_loader
    )

    with TestClient(test_app) as client:
        yield client


@pytest.fixture(scope="module")
def health_client(vespa_instance, config_manager):
    """
    FastAPI TestClient with health router.

    Mounts the health router for integration testing against real registries.
    """
    test_app = FastAPI()
    test_app.include_router(health.router)

    with TestClient(test_app) as client:
        yield client
