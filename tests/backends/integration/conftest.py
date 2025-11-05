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

        # Wait for config server to be ready
        manager.wait_for_config_ready(container_info)

        # Give Vespa additional time to fully initialize all services
        # Config port being ready doesn't mean data port is ready
        import time
        logger.info("Waiting additional 10 seconds for Vespa services to fully initialize...")
        time.sleep(10)
        logger.info("Vespa initialization complete")

        # NOTE: Schema deployment is intentionally SKIPPED for backend schema lifecycle tests
        # These tests are designed to TEST schema deployment, so they deploy schemas themselves
        # Pre-deploying schemas would interfere with the tests and cause timeout issues

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
