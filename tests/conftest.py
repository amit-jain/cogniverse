"""Global pytest configuration for test isolation"""

import gc
import os
import sys
import threading
import time

import pytest

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


@pytest.fixture(autouse=True, scope="function")
def cleanup_environment():
    """Clean up environment variables that might pollute tests"""
    # Save current environment
    saved_env = {}
    env_vars_to_track = ["VESPA_SCHEMA", "MLFLOW_TRACKING_URI", "PHOENIX_COLLECTOR_ENDPOINT"]
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
