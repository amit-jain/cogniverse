"""
Integration test configuration for backend tests.

Provides Vespa Docker instance fixture for testing schema lifecycle.
"""

import logging

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
import pytest

from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for backend integration tests.

    Module-scoped to match temp_config_manager and schema_loader fixtures.
    This ensures SchemaRegistry state stays consistent with Vespa state
    across all test classes in this module.

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
            from pathlib import Path
            from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
            from unittest.mock import MagicMock

            # Example shows direct instantiation (use fixture in actual tests)
            schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
            manager = TenantSchemaManager(
                backend_url="http://localhost",
                backend_port=vespa_instance["config_port"],
                http_port=vespa_instance["http_port"],
                config_manager=MagicMock(),
                schema_loader=schema_loader
            )
    """
    manager = VespaDockerManager()

    try:
        # Start container with module-specific ports
        container_info = manager.start_container(module_name=__name__, use_module_ports=True)

        # Wait for config server to be ready (with longer timeout for slow startups)
        manager.wait_for_config_ready(container_info, timeout=180)

        # Give Vespa additional time for internal services to initialize
        import time
        logger.info("Waiting 15 seconds for Vespa internal services to initialize...")
        time.sleep(15)

        # Deploy metadata schemas (organization, tenant, config) before any ConfigStore is used
        # This breaks the circular dependency: VespaConfigStore needs config_metadata schema
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info['config_port'],
        )
        schema_manager.upload_metadata_schemas(app_name="cogniverse")
        logger.info("Deployed metadata schemas (organization, tenant, config)")

        # Wait for Vespa HTTP/application endpoint to be ready after schema deployment
        # The config server deployment is async - HTTP service needs time to initialize
        manager.wait_for_application_ready(container_info, timeout=120)

        logger.info("Vespa initialization complete - ready for schema tests")

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

            logger.info("✅ Cleared singleton state")
        except Exception as e:
            logger.warning(f"⚠️  Error clearing singleton state: {e}")
