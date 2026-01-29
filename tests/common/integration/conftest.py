"""
Integration test configuration for common tests.

Provides Vespa Docker instance fixture for testing configuration persistence.
"""

import logging

import pytest

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for common integration tests.

    Module-scoped to reuse the container across tests in this module.
    Deploys metadata schemas before yielding to ensure config store works.

    Yields:
        dict: Vespa connection info with keys:
            - http_port: Vespa HTTP port (unique per module)
            - config_port: Vespa config server port (unique per module)
            - base_url: Full HTTP URL
            - container_name: Docker container name
    """
    manager = VespaDockerManager()

    try:
        # Start container with module-specific ports
        container_info = manager.start_container(
            module_name=__name__, use_module_ports=True
        )

        # Wait for config server to be ready (with longer timeout for slow startups)
        manager.wait_for_config_ready(container_info, timeout=180)

        # Give Vespa additional time for internal services to initialize
        import time

        logger.info("Waiting 15 seconds for Vespa internal services to initialize...")
        time.sleep(15)

        # Deploy metadata schemas (organization, tenant, config) before any ConfigStore is used
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager.upload_metadata_schemas(app_name="cogniverse")
        logger.info("Deployed metadata schemas (organization, tenant, config)")

        # Wait for Vespa HTTP/application endpoint to be ready after schema deployment
        manager.wait_for_application_ready(container_info, timeout=120)

        logger.info("Vespa initialization complete - ready for integration tests")

        # Yield instance info
        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        # Cleanup container
        manager.stop_container()

        # Clear singleton state to avoid interference with other test modules
        try:
            from cogniverse_core.registries.backend_registry import get_backend_registry
            from cogniverse_foundation.config.manager import ConfigManager

            # Clear backend registry instances
            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()

            # Clear ConfigManager singleton
            if hasattr(ConfigManager, "_instance"):
                ConfigManager._instance = None

            logger.info("Cleared singleton state")
        except Exception as e:
            logger.warning(f"Error clearing singleton state: {e}")
