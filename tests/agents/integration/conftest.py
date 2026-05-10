"""
Shared fixtures and utilities for agent integration tests.

Re-exports ``shared_memory_vespa`` from tests/memory/conftest.py so agent
integration tests that need a real Mem0+Vespa backend can request the
same module-scoped Vespa instance the memory tests use, without spinning
up a duplicate.
"""

import logging
import os
import platform
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import dspy
import httpx
import pytest

from cogniverse_agents.inference.deno_check import is_deno_available
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager, get_config

# Re-export the canonical shared_memory_vespa fixture so it's discoverable
# by tests under tests/agents/integration/ (pytest only walks UP from a
# test file's directory, not laterally into siblings).
from tests.memory.conftest import shared_memory_vespa  # noqa: F401

logger = logging.getLogger(__name__)


def is_llm_available() -> bool:
    """Check if the LM endpoint configured in ``configs/config.json`` is reachable.

    Reads ``llm_config.primary.api_base`` directly (no ConfigManager — avoids the
    BACKEND_URL env var requirement at import time) and probes both ``/api/tags``
    (native LM-server tag listing) and ``/v1/models`` (OAI-compat). Either
    returning HTTP 200 means the endpoint is up.
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        config_path = _Path(__file__).resolve().parents[3] / "configs" / "config.json"
        with open(config_path) as f:
            config = _json.load(f)
        api_base = (
            config.get("llm_config", {}).get("primary", {}).get("api_base") or ""
        ).rstrip("/")
        if not api_base:
            return False
        if api_base.endswith("/v1"):
            api_base = api_base[: -len("/v1")]
        for path in ("/api/tags", "/v1/models"):
            try:
                response = httpx.get(f"{api_base}{path}", timeout=5.0)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False
    except Exception:
        return False


def is_teacher_api_available() -> bool:
    """Check if router optimizer teacher API key is available."""
    import os

    return bool(os.getenv("ROUTER_OPTIMIZER_TEACHER_KEY"))


skip_if_no_lm = pytest.mark.skipif(
    not is_llm_available(),
    reason="Configured LLM endpoint not reachable",
)

skip_if_no_teacher_api = pytest.mark.skipif(
    not is_teacher_api_available(),
    reason="ROUTER_OPTIMIZER_TEACHER_KEY environment variable not set",
)


_DENO_RELEASE_BASE = "https://github.com/denoland/deno/releases/latest/download"


def _resolve_deno_artefact() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "deno-x86_64-unknown-linux-gnu.zip"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return "deno-aarch64-unknown-linux-gnu.zip"
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "deno-aarch64-apple-darwin.zip"
    if system == "Darwin" and machine in ("x86_64", "amd64"):
        return "deno-x86_64-apple-darwin.zip"
    raise RuntimeError(f"Unsupported platform for Deno install: {system}/{machine}")


def _install_deno_to_home() -> Path:
    """Download the latest Deno release zip into ~/.deno/bin/ and chmod +x.

    The cogniverse Deno probe (``is_deno_available``) already checks
    ``~/.deno/bin/deno`` and amends ``PATH`` when found, so this matches the
    install location the production code expects.
    """
    home_bin = Path.home() / ".deno" / "bin"
    home_bin.mkdir(parents=True, exist_ok=True)
    deno_path = home_bin / "deno"
    if deno_path.exists():
        return deno_path

    artefact = _resolve_deno_artefact()
    url = f"{_DENO_RELEASE_BASE}/{artefact}"
    logger.info("Downloading Deno from %s", url)
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / artefact
        with (
            urllib.request.urlopen(url, timeout=120) as resp,
            open(zip_path, "wb") as f,
        ):
            shutil.copyfileobj(resp, f)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(home_bin)

    if not deno_path.exists():
        raise RuntimeError(
            f"Deno install completed but binary missing at {deno_path}; "
            f"zip layout may have changed"
        )
    deno_path.chmod(
        deno_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    return deno_path


@pytest.fixture(scope="session")
def ensure_deno() -> Path:
    """Session-scoped fixture: guarantees Deno is reachable for RLM REPL tests.

    Installs the latest Deno release into ``~/.deno/bin/`` if missing
    (single binary, no package manager). Tests use this instead of
    skipping when Deno is absent — infrastructure managed inside the
    test suite, per project policy that infra-skips count as bugs.
    """
    if is_deno_available():
        return Path(shutil.which("deno") or Path.home() / ".deno" / "bin" / "deno")
    logger.info("Deno not on PATH or in ~/.deno/bin — installing for test session")
    deno_path = _install_deno_to_home()
    if not is_deno_available():
        raise RuntimeError(
            f"Installed Deno at {deno_path} but is_deno_available() still False"
        )
    return deno_path


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
        # Local OAI-compat LM servers accept any bearer token but litellm
        # refuses to construct an OpenAI client without an api_key set, so
        # it falls back to OPENAI_API_KEY and raises AuthenticationError
        # when neither is present. Test endpoints don't validate the key —
        # pass the cogniverse convention sentinel.
        api_key=llm_cfg.get("api_key") or "not-required",
        temperature=0.1,
        # 200 tokens was too small: synthesis / summarisation tests
        # truncate mid-sentence, masking real failures behind a
        # plausible-looking partial answer (the 3-doc synthesis test
        # would only ever surface 1 of 3 distinguishing facts before
        # hitting the cap). 800 is enough headroom for the largest
        # synthesis prompts in the test suite without slowing simple
        # Q&A tests appreciably (local LMs stop early on short answers).
        max_tokens=800,
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
