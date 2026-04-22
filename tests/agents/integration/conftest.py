"""
Shared fixtures and utilities for agent integration tests.
"""

import logging
import os

import dspy
import httpx
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager, get_config

logger = logging.getLogger(__name__)


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama service is available."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def is_llm_available() -> bool:
    """Check if the configured LLM endpoint is reachable.

    Reads api_base from configs/config.json directly (no ConfigManager
    needed — avoids BACKEND_URL env var requirement at import time).
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        config_path = _Path(__file__).resolve().parents[3] / "configs" / "config.json"
        with open(config_path) as f:
            config = _json.load(f)
        api_base = (
            config.get("llm_config", {})
            .get("primary", {})
            .get("api_base", "http://localhost:11434")
        )
        response = httpx.get(f"{api_base}/api/tags", timeout=5.0)
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

skip_if_no_llm = pytest.mark.skipif(
    not is_llm_available(),
    reason="Configured LLM endpoint not reachable",
)

skip_if_no_teacher_api = pytest.mark.skipif(
    not is_teacher_api_available(),
    reason="ROUTER_OPTIMIZER_TEACHER_KEY environment variable not set",
)


@pytest.fixture(scope="module")
def _dspy_lm_instance():
    """Module-scoped: create the LM once per module (expensive)."""
    cm = create_default_config_manager()
    config = get_config(tenant_id="test:unit", config_manager=cm)
    llm_cfg = config.get("llm_config", {}).get("primary", {})

    # Disable qwen3 thinking mode — it puts output in a 'thinking' field
    # that DSPy can't read, leaving content empty.
    extra_body = None
    model = llm_cfg["model"]
    if "qwen3" in model or "qwen-3" in model:
        extra_body = {"think": False}

    endpoint = LLMEndpointConfig(
        model=model,
        api_base=llm_cfg.get("api_base"),
        temperature=0.1,
        max_tokens=200,
        extra_body=extra_body,
    )
    return create_dspy_lm(endpoint)


@pytest.fixture
def dspy_lm(_dspy_lm_instance):
    """Function-scoped: re-apply dspy.configure before each test.

    The root conftest cleanup_dspy_state clears dspy.settings.lm after
    each test, so we must re-configure before every test that needs an LLM.
    """
    dspy.configure(lm=_dspy_lm_instance)
    return _dspy_lm_instance


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

        # Set env vars so create_default_config_manager() and agent code
        # resolve to the test Vespa container ports
        original_url = os.environ.get("BACKEND_URL")
        original_port = os.environ.get("BACKEND_PORT")
        original_config_port = os.environ.get("VESPA_CONFIG_PORT")
        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(agent_http_port)
        os.environ["VESPA_CONFIG_PORT"] = str(agent_config_port)

        from cogniverse_foundation.config import utils as config_utils
        config_utils._config_manager_singleton = None

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
        # Restore env vars
        config_utils._config_manager_singleton = None
        for var, orig in [
            ("BACKEND_URL", original_url),
            ("BACKEND_PORT", original_port),
            ("VESPA_CONFIG_PORT", original_config_port),
        ]:
            if orig is not None:
                os.environ[var] = orig
            else:
                os.environ.pop(var, None)

        # Cleanup: stop Docker container and clear state
        logger.info("Tearing down Vespa test instance...")
        try:
            manager.docker_manager.stop_container()
        except Exception as cleanup_err:
            logger.warning(f"Vespa container cleanup failed: {cleanup_err}")
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


@pytest.fixture(scope="module")
def real_telemetry(phoenix_container):
    """Module-scoped real TelemetryManager backed by Phoenix Docker.

    Depends on the root-conftest phoenix_container fixture which starts
    Phoenix on ports 16006 (HTTP) and 14317 (gRPC). Exposes a live
    TelemetryManager so agent telemetry span tests can emit and query
    real spans.
    """
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    config = TelemetryConfig(
        otlp_endpoint=os.getenv("TELEMETRY_OTLP_ENDPOINT", "localhost:4317"),
        provider_config={
            "http_endpoint": "http://localhost:16006",
            "grpc_endpoint": "http://localhost:14317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
