"""
Shared fixtures and utilities for agent integration tests.
"""

import logging

import httpx
import pytest

logger = logging.getLogger(__name__)


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama service is available."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def is_teacher_api_available() -> bool:
    """Check if router optimizer teacher API key is available."""
    import os

    return bool(os.getenv("ROUTER_OPTIMIZER_TEACHER_KEY"))


# Skip markers for integration tests
skip_if_no_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama service not available at http://localhost:11434",
)

skip_if_no_teacher_api = pytest.mark.skipif(
    not is_teacher_api_available(),
    reason="ROUTER_OPTIMIZER_TEACHER_KEY environment variable not set",
)


@pytest.fixture(autouse=True)
def clear_singleton_state_between_tests():
    """
    Function-scoped autouse fixture to clear singleton state between each test.

    This prevents test isolation issues when using module-scoped fixtures like vespa_with_schema.
    Runs automatically before each test to ensure clean state.
    """
    # Clear before each test
    from cogniverse_core.registries.backend_registry import get_backend_registry
    from cogniverse_foundation.config.manager import ConfigManager

    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        initial_count = len(registry._backend_instances)
        registry._backend_instances.clear()
        if initial_count > 0:
            logger.debug(
                f"🧹 Cleared {initial_count} cached backend instances before test"
            )

    if hasattr(ConfigManager, "_instance"):
        if ConfigManager._instance is not None:
            logger.debug("🧹 Cleared ConfigManager singleton before test")
        ConfigManager._instance = None

    import cogniverse_vespa.search_backend as _sb

    with _sb._CACHE_LOCK:
        _sb._RANKING_STRATEGIES_CACHE = None

    yield

    # Clear after each test as well
    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

    with _sb._CACHE_LOCK:
        _sb._RANKING_STRATEGIES_CACHE = None


@pytest.fixture(scope="module")
def vespa_with_schema():
    """
    Module-scoped Vespa instance with deployed schemas for agent integration tests.

    Similar to system tests - deploys minimal video search schema for testing.

    Yields:
        dict: Vespa connection info with keys:
            - http_port: Vespa HTTP port
            - config_port: Vespa config server port
            - base_url: Full HTTP URL
            - manager: VespaTestManager instance
            - default_schema: Default schema name
    """

    # Import after ensuring no prior state
    from tests.utils.docker_utils import generate_unique_ports

    # Generate unique ports for this test module
    agent_http_port, agent_config_port = generate_unique_ports(__name__)

    logger.info(
        f"Agent tests using ports: {agent_http_port} (http), {agent_config_port} (config)"
    )

    # Clear singletons before setup
    from cogniverse_core.registries.backend_registry import get_backend_registry
    from cogniverse_foundation.config.manager import ConfigManager

    logger.info("🧹 Clearing singleton state before Vespa setup...")

    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()

    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

    # Clear ranking strategies cache — unit tests may have poisoned it
    # with empty results from Mock schema_loaders
    import cogniverse_vespa.search_backend as _sb

    with _sb._CACHE_LOCK:
        _sb._RANKING_STRATEGIES_CACHE = None

    logger.info("✅ Singleton state cleared")

    # Import VespaTestManager (creates temp config)
    from tests.system.vespa_test_manager import VespaTestManager

    # Create manager with test ports
    manager = VespaTestManager(http_port=agent_http_port, config_port=agent_config_port)

    try:
        # Full setup: start container, deploy schema, ingest test data
        logger.info("Setting up Vespa with test schema and data...")
        if not manager.full_setup():
            pytest.fail("Failed to setup Vespa test environment")

        logger.info(f"✅ Vespa ready at http://localhost:{agent_http_port}")

        # Yield with manager for agent fixture access
        yield {
            "http_port": agent_http_port,
            "config_port": agent_config_port,
            "base_url": f"http://localhost:{agent_http_port}",
            "manager": manager,
            "default_schema": manager.default_test_schema,
        }

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.fail(f"Failed to start Vespa: {e}")

    finally:
        # Cleanup
        logger.info("Tearing down Vespa test instance...")
        manager.cleanup()

        # Clear singleton state
        try:
            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()

            if hasattr(ConfigManager, "_instance"):
                ConfigManager._instance = None

            with _sb._CACHE_LOCK:
                _sb._RANKING_STRATEGIES_CACHE = None

            logger.info("✅ Cleared singleton state after teardown")
        except Exception as e:
            logger.warning(f"⚠️  Error clearing singleton state: {e}")
