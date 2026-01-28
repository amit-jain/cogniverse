"""
Pytest configuration for routing tests.

Provides cleanup hooks to prevent model memory leaks and segfaults.
"""

import gc
import os
import threading
import time

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
    Clean up background threads from tqdm (transformers) and posthog.

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


@pytest.fixture(scope="module", autouse=True)
def cleanup_models_after_module():
    """Cleanup models after each test module to prevent segfaults.

    This fixture runs automatically after each test module completes.
    It forces garbage collection to clean up heavy ML models (GLiNER, transformers)
    that may cause segfaults when multiple test modules load them.
    """
    yield
    # Clean up background threads and force garbage collection
    cleanup_background_threads()
    gc.collect()
    gc.collect()  # Call twice to ensure cleanup of circular references


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each individual test to prevent resource leaks."""
    yield
    # Clean up background threads after each test
    cleanup_background_threads()


# Import standardized telemetry and Phoenix fixtures from parent conftest
# Note: Due to pytest.ini testpaths configuration, we need to import these here
# rather than relying on parent conftest.py discovery
import sys
from pathlib import Path

# Add parent tests directory to path to import from tests/conftest.py
parent_tests_dir = Path(__file__).parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

# Import fixtures from tests/conftest.py (not root conftest.py)
# Use absolute import to avoid confusion with root conftest.py
sys.path.insert(0, str(parent_tests_dir.parent))  # Add project root

# Import Phoenix fixtures from parent tests/conftest.py
from tests.conftest import (
    phoenix_client,
    phoenix_container,
    telemetry_config_with_phoenix,
    telemetry_manager_with_phoenix,
    telemetry_manager_without_phoenix,
)

__all__ = [
    "phoenix_client",
    "phoenix_container",
    "telemetry_config_with_phoenix",
    "telemetry_manager_with_phoenix",
    "telemetry_manager_without_phoenix",
]
