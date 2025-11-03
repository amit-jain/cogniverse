"""
Integration test configuration for backend tests.

Provides Vespa Docker instance fixture for testing schema lifecycle.
"""

import logging

import pytest

from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for backend integration tests.

    Uses unique ports per test module to avoid conflicts with:
    - Main Vespa (8080)
    - System tests (different module, different ports)
    - Other test modules (deterministic hash-based port assignment)

    Yields:
        dict: Vespa connection info with keys:
            - http_port: Vespa HTTP port (unique per module)
            - config_port: Vespa config server port (unique per module)
            - base_url: Full HTTP URL
            - container_name: Docker container name

    Example:
        def test_schema_deployment(vespa_instance):
            manager = TenantSchemaManager(
                backend_url="http://localhost",
                backend_port=vespa_instance["http_port"]
            )
    """
    manager = VespaDockerManager()

    try:
        # Start container with module-specific ports
        container_info = manager.start_container(module_name=__name__, use_module_ports=True)

        # Wait for config server to be ready
        manager.wait_for_config_ready(container_info)

        # Deploy base schemas (must be AFTER config ready, BEFORE application ready)
        logger.info("Deploying base schemas...")
        try:
            manager.deploy_schemas(container_info, include_metadata=True)
        except Exception as e:
            logger.warning(f"Schema deployment failed: {e}")
            # Continue anyway - some tests might not need schemas

        # Wait for application to be ready (must be AFTER schemas deployed)
        manager.wait_for_application_ready(container_info)

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
            from cogniverse_core.config.manager import ConfigManager
            from cogniverse_core.registries.backend_registry import get_backend_registry
            from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

            # Clear TenantSchemaManager singleton
            TenantSchemaManager._clear_instance()

            # Clear backend registry instances
            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()

            # Clear ConfigManager singleton
            if hasattr(ConfigManager, "_instance"):
                ConfigManager._instance = None

            logger.info("✅ Cleared singleton state")
        except Exception as e:
            logger.warning(f"⚠️  Error clearing singleton state: {e}")
