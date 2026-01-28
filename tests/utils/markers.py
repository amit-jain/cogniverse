"""
Test markers and utilities for conditional test execution.

Provides functions to check availability of backends, models, and dependencies.
"""

import importlib
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest
import requests


def is_ci_environment() -> bool:
    """Check if running in CI environment."""
    ci_indicators = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS_URL",
    ]
    return any(os.getenv(indicator) for indicator in ci_indicators)


def is_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_vespa_running() -> bool:
    """Check if Vespa is running locally."""
    try:
        response = requests.get("http://localhost:8080/ApplicationStatus", timeout=5)
        return response.status_code == 200
    except (requests.RequestException, requests.ConnectionError):
        return False


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_module_available(module_name: str) -> bool:
    """Check if a Python module is available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def is_videoprism_available() -> bool:
    """Check if VideoPrism module is available."""
    try:
        # First try to import videoprism directly
        importlib.import_module("videoprism")
        return True
    except ImportError:
        # Check if it exists in the adjacent directory structure
        import sys
        from pathlib import Path

        project_root = Path(
            __file__
        ).parent.parent.parent  # From tests/utils/ to project root
        videoprism_path = project_root.parent / "videoprism"  # Adjacent to cogniverse

        if (
            videoprism_path.exists()
            and (videoprism_path / "videoprism" / "__init__.py").exists()
        ):
            # Add to path temporarily to check if it's loadable
            videoprism_parent = str(videoprism_path)
            if videoprism_parent not in sys.path:
                sys.path.insert(0, videoprism_parent)
                try:
                    importlib.import_module("videoprism")
                    return True
                except ImportError:
                    pass
                finally:
                    # Clean up path
                    if videoprism_parent in sys.path:
                        sys.path.remove(videoprism_parent)

        return False


def is_cv2_available() -> bool:
    """Check if OpenCV (cv2) is available."""
    return is_module_available("cv2")


def is_whisper_available() -> bool:
    """Check if Whisper is available."""
    return is_module_available("whisper")


def has_gpu_support() -> bool:
    """Check if GPU support is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def model_exists(model_path: str) -> bool:
    """Check if a model file/directory exists."""
    return Path(model_path).exists()


def has_sufficient_memory(min_gb: float = 8.0) -> bool:
    """Check if system has sufficient memory."""
    try:
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        return memory_gb >= min_gb
    except ImportError:
        # Conservative assumption if psutil not available
        return True


# Skip conditions for different scenarios
skip_if_no_vespa = pytest.mark.skipif(
    not is_vespa_running(), reason="Vespa backend not running"
)

skip_if_no_docker = pytest.mark.skipif(
    not is_docker_available(), reason="Docker not available"
)

skip_if_no_ffmpeg = pytest.mark.skipif(
    not is_ffmpeg_available(), reason="FFmpeg not available"
)

skip_if_no_cv2 = pytest.mark.skipif(
    not is_cv2_available(), reason="OpenCV (cv2) not available"
)

skip_if_no_whisper = pytest.mark.skipif(
    not is_whisper_available(), reason="Whisper not available"
)

skip_if_no_videoprism = pytest.mark.skipif(
    not is_videoprism_available(),
    reason="VideoPrism not available in adjacent directory",
)

skip_if_no_gpu = pytest.mark.skipif(not has_gpu_support(), reason="GPU not available")

skip_if_ci = pytest.mark.skipif(
    is_ci_environment(), reason="Test skipped in CI environment"
)

skip_if_low_memory = pytest.mark.skipif(
    not has_sufficient_memory(8.0), reason="Insufficient memory (requires 8GB+)"
)

# Combined markers for heavy model tests
skip_heavy_models_in_ci = pytest.mark.skipif(
    is_ci_environment() and not os.getenv("RUN_HEAVY_TESTS"),
    reason="Heavy model tests skipped in CI (set RUN_HEAVY_TESTS=1 to override)",
)


def pytest_configure(config):
    """Configure pytest with our custom markers."""
    # This ensures our markers are recognized
    pass


def get_available_models() -> Dict[str, bool]:
    """Get availability status of different models."""
    return {
        "colpali": is_module_available("colpali_engine"),
        "videoprism": is_videoprism_available()
        and has_sufficient_memory(8.0),  # Check adjacent directory
        "colqwen": is_module_available("transformers") and has_sufficient_memory(8.0),
        "whisper": is_whisper_available(),
        "cv2": is_cv2_available(),
    }


def get_test_environment_info() -> Dict[str, Any]:
    """Get comprehensive test environment information."""
    return {
        "ci_environment": is_ci_environment(),
        "docker_available": is_docker_available(),
        "vespa_running": is_vespa_running(),
        "ffmpeg_available": is_ffmpeg_available(),
        "gpu_available": has_gpu_support(),
        "sufficient_memory": has_sufficient_memory(),
        "available_models": get_available_models(),
    }
