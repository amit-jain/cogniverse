"""
Docker test utilities for Vespa container management.

Provides utilities for:
1. Generating unique ports per test module to avoid conflicts
2. Waiting for Docker containers to be fully removed before proceeding
"""

import hashlib
import subprocess
import time
from typing import Tuple

from tests.utils.async_polling import simulate_processing_delay, wait_for_vespa_indexing


def generate_unique_ports(
    module_name: str, base_http_port: int = 50000
) -> Tuple[int, int]:
    """
    Generate unique HTTP and config ports for a test module in the IANA
    ephemeral range (49152-65535) so they don't collide with well-known
    or registered services running on the host.

    The hash is seeded with module_name + the OS PID so concurrent pytest
    invocations from different shells land on distinct ports while retries
    within the same process get the same port.

    Args:
        module_name: Test module name (e.g., __name__ from the test file)
        base_http_port: Starting port for HTTP (default: 50000)

    Returns:
        Tuple of (http_port, config_port). Both in the ephemeral range.
    """
    import os

    seed = f"{module_name}:{os.getpid()}"
    port_hash = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)

    # Range: 50000-60999 with +1000 config offset (standard test ports).
    # Code that derives config_port from http_port uses calculate_config_port()
    # which handles the offset correctly for any port.
    http_port = 50000 + (port_hash % 11000)
    config_port = http_port + 1000

    return http_port, config_port


def wait_for_container_removal(container_name: str, timeout: int = 30) -> bool:
    """
    Wait for Docker container to be fully removed and resources released.

    Actively polls Docker to verify container is gone, rather than blindly sleeping.
    This is more reliable and often faster than fixed sleep delays.

    Args:
        container_name: Name of Docker container to wait for
        timeout: Maximum seconds to wait (default: 30)

    Returns:
        True if container was removed within timeout, False otherwise

    Example:
        >>> subprocess.run(["docker", "stop", "vespa-test"])
        >>> subprocess.run(["docker", "rm", "vespa-test"])
        >>> if wait_for_container_removal("vespa-test"):
        ...     print("Container fully removed, safe to start new one")
    """
    start = time.time()

    while time.time() - start < timeout:
        # Check if container still exists (running or stopped)
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if container_name not in result.stdout:
            # Container fully removed from Docker
            # Give Docker a moment to release ports and resources
            wait_for_vespa_indexing(delay=1)
            return True

        # Container still exists, wait and retry
        simulate_processing_delay(delay=0.5, description="docker container removal")

    # Timeout reached
    return False


def cleanup_vespa_container(container_name: str, timeout: int = 30) -> bool:
    """
    Stop, remove, and wait for Vespa container cleanup.

    Combines stop, remove, and wait operations into single reliable cleanup.

    Args:
        container_name: Name of Docker container to cleanup
        timeout: Maximum seconds to wait for cleanup (default: 30)

    Returns:
        True if cleanup succeeded, False otherwise

    Example:
        >>> if cleanup_vespa_container("vespa-test-8081"):
        ...     print("Ready to start new container")
    """
    try:
        # Stop container
        stop_result = subprocess.run(
            ["docker", "stop", container_name], capture_output=True, timeout=timeout
        )

        # Remove container
        remove_result = subprocess.run(
            ["docker", "rm", container_name], capture_output=True, timeout=timeout
        )

        if stop_result.returncode == 0 and remove_result.returncode == 0:
            # Wait for Docker to fully release resources
            return wait_for_container_removal(container_name, timeout=timeout)

        return False

    except Exception as e:
        print(f"⚠️  Error during container cleanup: {e}")
        return False
