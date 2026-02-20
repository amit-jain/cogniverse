"""
Integration test configuration for admin profile tests.

Provides shared Vespa Docker instance with metadata schemas deployed.
"""

import logging
import time

import pytest

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for admin integration tests.

    Session-scoped to share across all admin test modules.
    Deploys metadata schemas (config_metadata, organization_metadata, etc.)
    so VespaConfigStore can be used immediately.

    Yields:
        dict: Vespa connection info with http_port, config_port, base_url, container_name
    """
    manager = VespaDockerManager()

    try:
        container_info = manager.start_container(
            module_name="admin_tests", use_module_ports=True
        )

        # Wait for config server
        manager.wait_for_config_ready(container_info, timeout=180)

        # Wait for internal services to initialize
        logger.info("Waiting 15 seconds for Vespa internal services to initialize...")
        time.sleep(15)

        # Deploy metadata schemas (config_metadata, organization_metadata, tenant_metadata)
        # MUST happen before VespaConfigStore can be used
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager.upload_metadata_schemas(app_name="cogniverse")
        logger.info("Deployed metadata schemas (organization, tenant, config)")

        # Wait for application readiness after schema deployment
        manager.wait_for_application_ready(container_info, timeout=120)

        logger.info("Vespa initialization complete - ready for admin tests")

        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        manager.stop_container()

        # Clear singleton state to avoid interference with other test modules
        try:
            from cogniverse_core.registries.backend_registry import BackendRegistry

            BackendRegistry._instance = None
            BackendRegistry._backend_instances.clear()
        except Exception:
            pass
