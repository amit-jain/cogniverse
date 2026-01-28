"""Global pytest configuration for test isolation"""

import gc
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from tests.utils.async_polling import simulate_processing_delay

# Configure torch and tokenizers to avoid threading issues in pytest
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Import torch and configure threading before any tests run
try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass


def cleanup_background_threads():
    """
    Clean up background threads from tqdm (transformers) and posthog (mem0ai).

    These libraries create daemon threads that can cause segfaults during pytest
    cleanup in async tests. We need to give them time to finish and exit cleanly.
    """
    max_wait = 2.0  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        background_threads = [
            t
            for t in threading.enumerate()
            if t != threading.current_thread()
            and t.daemon
            and any(name in t.name.lower() for name in ["tqdm", "posthog", "monitor"])
        ]

        if not background_threads:
            break

        # Give threads time to finish their work
        simulate_processing_delay(delay=0.1, description="test processing")

    # Force garbage collection to clean up any remaining references
    gc.collect()


@pytest.fixture(autouse=True, scope="function")
def cleanup_dspy_state():
    """Clean up DSPy state between tests to prevent isolation issues"""
    yield

    # Clean up any DSPy state after each test
    try:
        import dspy

        # Reset ALL DSPy settings attributes to prevent any state pollution
        if hasattr(dspy, "settings"):
            # Clear the LM
            if hasattr(dspy.settings, "lm"):
                dspy.settings.lm = None

            # Clear adapters if they exist
            if hasattr(dspy.settings, "adapter"):
                dspy.settings.adapter = None

            # Clear any other cached settings
            if hasattr(dspy.settings, "rm"):
                dspy.settings.rm = None

            # Clear experimental settings
            if hasattr(dspy.settings, "experimental"):
                dspy.settings.experimental = False

        # Clear any context stack from async tests
        if hasattr(dspy, "_context_stack"):
            if hasattr(dspy._context_stack, "clear"):
                dspy._context_stack.clear()
            elif isinstance(dspy._context_stack, list):
                dspy._context_stack.clear()

    except (ImportError, AttributeError, RuntimeError):
        # RuntimeError can occur if called from different async context
        pass

    # Clean up background threads from tqdm and posthog
    cleanup_background_threads()


def pytest_collection_modifyitems(session, config, items):
    """
    Clean up sys.modules pollution from test_composing_agents_main.py immediately after collection.

    This hook runs after all tests are collected but before any tests execute.
    test_composing_agents_main.py pollutes sys.modules with mocks at import time,
    which causes integration tests to fail when they try to import real modules.
    """
    # List of module names that test_composing_agents_main.py mocks
    mocked_modules = [
        "google",
        "google.adk",
        "google.adk.agents",
        "google.adk.runners",
        "google.adk.sessions",
        "google.adk.tools",
        "google.genai",
        "google.genai.types",
        "gliner",
    ]

    # Clean up mocked modules from sys.modules
    for module_name in mocked_modules:
        if module_name in sys.modules:
            # Only remove if it's a MagicMock (from test_composing_agents_main.py)
            from unittest.mock import MagicMock

            if isinstance(sys.modules[module_name], MagicMock):
                del sys.modules[module_name]


@pytest.fixture(autouse=True, scope="function")
def cleanup_vlm_state():
    """Clean up VLM interface state between tests"""
    yield
    # Clean up any cached VLM instances
    try:
        from cogniverse_core.common.vlm_interface import VLMInterface

        # Clear any class-level state if it exists
        if hasattr(VLMInterface, "_instance"):
            VLMInterface._instance = None
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True, scope="session")
def test_output_dir():
    """
    Configure output directory for test artifacts (logs, databases, etc.).

    Overrides config's output_base_dir to use temporary directory.
    Automatically cleans up after test session completes.
    """
    # Create temp directory for all test artifacts
    temp_dir = tempfile.mkdtemp(prefix="cogniverse_test_")
    artifacts_dir = Path(temp_dir)

    # Override config's output_base_dir for tests
    # OutputManager reads from config.get("output_base_dir", "outputs")
    os.environ["TEST_OUTPUT_BASE_DIR"] = str(artifacts_dir)

    print(f"\n🗂️  Test output directory: {artifacts_dir}")

    yield artifacts_dir

    # Cleanup: Remove entire test output directory
    try:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test artifacts: {artifacts_dir}")
    except Exception as e:
        print(f"\n⚠️  Failed to cleanup {artifacts_dir}: {e}")
    finally:
        os.environ.pop("TEST_OUTPUT_BASE_DIR", None)


@pytest.fixture(autouse=True, scope="function")
def cleanup_environment():
    """Clean up environment variables that might pollute tests"""
    # Save current environment
    saved_env = {}
    env_vars_to_track = ["VESPA_SCHEMA", "MLFLOW_TRACKING_URI", "OTLP_ENDPOINT"]
    for var in env_vars_to_track:
        if var in os.environ:
            saved_env[var] = os.environ[var]

    yield

    # Restore saved environment variables
    for var in env_vars_to_track:
        if var in saved_env:
            os.environ[var] = saved_env[var]
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def telemetry_manager_without_phoenix():
    """
    Standard telemetry manager fixture for tests that don't need real Phoenix.

    Sets up telemetry with mock endpoints - tests can use real telemetry components
    without connecting to Phoenix. Use this for unit and integration tests that
    just need telemetry configured but don't export/query real spans.
    """
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    # Reset TelemetryManager singleton AND clear provider cache
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    # Create config with mock endpoints (tests don't actually connect)
    config = TelemetryConfig(
        otlp_endpoint="http://localhost:24317",  # gRPC endpoint for span export
        provider_config={
            "http_endpoint": "http://localhost:26006",  # HTTP endpoint for queries
            "grpc_endpoint": "http://localhost:24317",  # gRPC endpoint (same as OTLP)
        },
        batch_config=BatchExportConfig(
            use_sync_export=True
        ),  # Synchronous export for tests
    )

    # Set as the global singleton
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    # Cleanup
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture(scope="module")
def phoenix_container():
    """
    Start Phoenix Docker container with gRPC support for integration tests.

    Uses non-default ports to avoid conflicts:
    - HTTP: 16006 (instead of 6006)
    - gRPC: 14317 (instead of 4317)

    Sets OTLP_ENDPOINT env var for tests and resets TelemetryManager.
    """
    import subprocess

    import requests

    from cogniverse_foundation.telemetry.manager import TelemetryManager

    original_endpoint = os.environ.get("OTLP_ENDPOINT")
    original_sync_export = os.environ.get("TELEMETRY_SYNC_EXPORT")

    # Set environment for tests
    os.environ["OTLP_ENDPOINT"] = "http://localhost:14317"
    os.environ["TELEMETRY_SYNC_EXPORT"] = "true"

    # Reset TelemetryManager to pick up new env vars
    TelemetryManager.reset()

    container_name = f"phoenix_test_{int(time.time() * 1000)}"

    try:
        # Start Phoenix container
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "16006:6006",  # HTTP port
                "-p",
                "14317:4317",  # gRPC port
                "-e",
                "PHOENIX_WORKING_DIR=/phoenix",
                "arizephoenix/phoenix:latest",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

        # Wait for Phoenix to be ready
        max_wait_time = 60
        poll_interval = 2
        start_time = time.time()
        phoenix_ready = False

        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get("http://localhost:16006", timeout=2)
                if response.status_code == 200:
                    phoenix_ready = True
                    break
            except Exception:
                pass
            time.sleep(poll_interval)

        if not phoenix_ready:
            logs_result = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            raise RuntimeError(
                f"Phoenix failed to start after {max_wait_time} seconds. Logs:\n{logs_result.stdout}\n{logs_result.stderr}"
            )

        yield container_name

    finally:
        # Cleanup
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=False,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "rm", container_name],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass

        # Restore environment
        if original_endpoint:
            os.environ["OTLP_ENDPOINT"] = original_endpoint
        else:
            os.environ.pop("OTLP_ENDPOINT", None)

        if original_sync_export:
            os.environ["TELEMETRY_SYNC_EXPORT"] = original_sync_export
        else:
            os.environ.pop("TELEMETRY_SYNC_EXPORT", None)


@pytest.fixture
def phoenix_client(phoenix_container):
    """Phoenix client for querying spans from Docker container"""
    import phoenix as px

    return px.Client(endpoint="http://localhost:16006")


@pytest.fixture
def telemetry_config_with_phoenix(phoenix_container):
    """
    Telemetry config for tests using real Phoenix Docker container.

    Depends on phoenix_container to ensure env vars are set.
    """
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )

    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "localhost:4317")
    config = TelemetryConfig(
        otlp_endpoint=otlp_endpoint,
        provider_config={
            "http_endpoint": "http://localhost:16006",
            "grpc_endpoint": "http://localhost:14317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    return config


@pytest.fixture
def telemetry_manager_with_phoenix(telemetry_config_with_phoenix):
    """
    Telemetry manager for tests using real Phoenix Docker container.

    Sets up telemetry manager as global singleton for the test.
    """
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    manager = TelemetryManager(config=telemetry_config_with_phoenix)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


# ==================== Backend Configuration Fixtures ====================


@pytest.fixture(scope="session", autouse=True)
def backend_config_env():
    """
    Set environment variables for backend configuration.

    Sets BACKEND_URL and BACKEND_PORT environment variables
    required by create_default_config_manager().

    Uses TEST_BACKEND_URL and TEST_BACKEND_PORT if available,
    otherwise defaults to localhost:8080.

    This fixture is autouse=True so it applies to all tests automatically.
    """
    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")

    # Set test values
    os.environ["BACKEND_URL"] = os.environ.get("TEST_BACKEND_URL", "http://localhost")
    os.environ["BACKEND_PORT"] = os.environ.get("TEST_BACKEND_PORT", "8080")

    yield

    # Restore original values
    if original_url is not None:
        os.environ["BACKEND_URL"] = original_url
    elif "BACKEND_URL" in os.environ:
        del os.environ["BACKEND_URL"]

    if original_port is not None:
        os.environ["BACKEND_PORT"] = original_port
    elif "BACKEND_PORT" in os.environ:
        del os.environ["BACKEND_PORT"]


@pytest.fixture
def config_manager(backend_config_env):
    """
    Create ConfigManager with backend store for testing.

    Requires backend_config_env fixture to set environment variables.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager

    return create_default_config_manager()


@pytest.fixture
def config_manager_memory():
    """
    Create ConfigManager with in-memory store for unit testing.

    Does not require any backend infrastructure (Vespa, etc.).
    Use this for unit tests that test business logic without
    needing real backend connectivity.
    """
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


@pytest.fixture
def workflow_store(backend_config_env):
    """
    Create VespaWorkflowStore for testing.

    Requires backend_config_env fixture to set environment variables.
    """
    from cogniverse_vespa.workflow.workflow_store import VespaWorkflowStore

    store = VespaWorkflowStore(
        vespa_url=os.environ.get("BACKEND_URL", "http://localhost"),
        vespa_port=int(os.environ.get("BACKEND_PORT", "8080")),
    )
    store.initialize()
    return store
