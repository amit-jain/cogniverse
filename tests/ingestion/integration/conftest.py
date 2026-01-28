"""
Ingestion integration test configuration and fixtures.

Provides module-scoped Vespa instance for ingestion tests.
Sets up BACKEND_URL environment variable required by BootstrapConfig.
"""

import os

import pytest

from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.docker_utils import generate_unique_ports

# Generate unique ports based on this module name
INGESTION_VESPA_PORT, INGESTION_VESPA_CONFIG_PORT = generate_unique_ports(__name__)


@pytest.fixture(scope="module")
def ingestion_vespa_backend():
    """
    Module-scoped Vespa instance for ingestion integration tests.

    Sets up:
    - Vespa Docker container with unique ports
    - BACKEND_URL environment variable
    - Cleans up after module tests complete
    """
    manager = VespaTestManager(
        app_name="test-ingestion-module",
        http_port=INGESTION_VESPA_PORT,
        config_port=INGESTION_VESPA_CONFIG_PORT,
    )

    # Save old environment
    old_backend_url = os.environ.get("BACKEND_URL")
    old_backend_port = os.environ.get("BACKEND_PORT")

    try:
        # Start Vespa container
        if not manager.setup_application_directory():
            pytest.skip("Failed to setup application directory")

        if not manager.deploy_test_application():
            pytest.skip("Failed to deploy Vespa test application")

        # Set environment for tests
        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(manager.http_port)

        yield {
            "manager": manager,
            "http_port": manager.http_port,
            "config_port": manager.config_port,
            "backend_url": f"http://localhost:{manager.http_port}",
        }

    finally:
        # Restore environment
        if old_backend_url is not None:
            os.environ["BACKEND_URL"] = old_backend_url
        elif "BACKEND_URL" in os.environ:
            del os.environ["BACKEND_URL"]

        if old_backend_port is not None:
            os.environ["BACKEND_PORT"] = old_backend_port
        elif "BACKEND_PORT" in os.environ:
            del os.environ["BACKEND_PORT"]

        # Cleanup container
        manager.cleanup()
