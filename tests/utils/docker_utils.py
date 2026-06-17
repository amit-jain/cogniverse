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


def _port_is_free(port: int) -> bool:
    """True if ``port`` can be bound right now (i.e. nothing holds it)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def generate_unique_ports(
    module_name: str, base_http_port: int = 40000
) -> Tuple[int, int]:
    """
    Return a free ``(http_port, config_port)`` pair for a Vespa test container.

    ``config_port`` is always ``http_port + 10991`` (the standard Vespa offset
    some callers re-derive), and BOTH ports are probed as actually-bindable
    before being returned — so a leftover container from a crashed prior run
    or a concurrent session can't cause an ``address already in use`` bind
    failure. ``http_port`` stays in ``[base_http_port, 54544]`` so the derived
    ``config_port`` stays under 65535.

    Callers invoke this once and cache the result; there is no
    same-port-on-retry guarantee. Falls back to a deterministic
    module_name+PID hash if no free pair is found after probing (e.g. a
    sandbox where bind probing is unreliable), preserving the old behaviour.

    Args:
        module_name: Test module name (e.g., __name__ from the test file)
        base_http_port: Lowest HTTP port to consider (default: 40000)

    Returns:
        Tuple of (http_port, config_port).
    """
    import os
    import random

    for _ in range(200):
        http_port = random.randint(base_http_port, 54544)
        config_port = http_port + 10991
        if _port_is_free(http_port) and _port_is_free(config_port):
            return http_port, config_port

    # Fallback: deterministic hash (original behaviour) when probing fails.
    seed = f"{module_name}:{os.getpid()}"
    port_hash = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    http_port = base_http_port + (port_hash % 14544)
    return http_port, http_port + 10991


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
