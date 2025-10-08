"""Global pytest configuration for test isolation"""

import gc
import os
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
        time.sleep(0.1)

    # Force garbage collection to clean up any remaining references
    gc.collect()


@pytest.fixture(autouse=True, scope="function")
def cleanup_dspy_state():
    """Clean up DSPy state between tests to prevent isolation issues"""
    yield
    # Clean up any DSPy state after each test
    try:
        import dspy

        # Reset settings to prevent state pollution
        # Don't call configure() as it breaks async task isolation
        # Instead, directly clear the internal state
        if hasattr(dspy.settings, "_instance"):
            dspy.settings._instance = None
    except (ImportError, AttributeError, RuntimeError):
        # RuntimeError can occur if called from different async context
        pass

    # Clean up background threads from tqdm and posthog
    cleanup_background_threads()


@pytest.fixture(autouse=True, scope="function")
def cleanup_vlm_state():
    """Clean up VLM interface state between tests"""
    yield
    # Clean up any cached VLM instances
    try:
        from src.common.vlm_interface import VLMInterface

        # Clear any class-level state if it exists
        if hasattr(VLMInterface, "_instance"):
            VLMInterface._instance = None
    except (ImportError, AttributeError):
        pass
