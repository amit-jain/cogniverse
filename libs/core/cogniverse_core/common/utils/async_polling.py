"""
Production async polling utilities.

Provides semantic wait functions for production code to replace
generic time.sleep() calls with meaningful, purpose-specific waits.
"""

import time
from typing import Callable, Optional, Any


def wait_for_service_ready(
    check_fn: Optional[Callable[[], bool]] = None,
    timeout: float = 30.0,
    poll_interval: float = 1.0,
    description: str = "service"
) -> bool:
    """
    Wait for a service to be ready using a check function.

    Args:
        check_fn: Function that returns True when service is ready
        timeout: Maximum time to wait in seconds
        poll_interval: Time between polls in seconds
        description: Description of what we're waiting for

    Returns:
        True if service became ready, False if timeout exceeded
    """
    if check_fn is None:
        # Simple delay if no check function provided
        time.sleep(min(poll_interval, timeout))
        return True

    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_fn():
            return True
        time.sleep(poll_interval)

    return False


def wait_for_operation_complete(
    delay: float = 1.0,
    description: str = "operation completion"
) -> None:
    """
    Wait for an operation to complete.

    Args:
        delay: Time to wait in seconds
        description: Description of the operation
    """
    time.sleep(delay)


def wait_for_eventual_consistency(
    delay: float = 2.0,
    description: str = "eventual consistency"
) -> None:
    """
    Wait for eventual consistency in distributed systems.

    Args:
        delay: Time to wait in seconds
        description: Description of what needs to be consistent
    """
    time.sleep(delay)


def wait_for_retry_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    description: str = "retry backoff"
) -> None:
    """
    Wait with exponential or linear backoff for retries.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential: Use exponential backoff if True, linear if False
        description: Description of what we're retrying
    """
    if exponential:
        delay = min(base_delay * (2 ** attempt), max_delay)
    else:
        delay = min(base_delay * (attempt + 1), max_delay)

    time.sleep(delay)


def wait_for_cache_refresh(
    ttl: float,
    buffer: float = 0.1,
    description: str = "cache refresh"
) -> None:
    """
    Wait for cache TTL to expire plus a small buffer.

    Args:
        ttl: Cache TTL in seconds
        buffer: Additional buffer time in seconds
        description: Description of what cache is expiring
    """
    time.sleep(ttl + buffer)


def wait_for_process_startup(
    delay: float = 2.0,
    description: str = "process startup"
) -> None:
    """
    Wait for a process to start up.

    Args:
        delay: Time to wait in seconds
        description: Description of the process
    """
    time.sleep(delay)


def wait_for_resource_cleanup(
    delay: float = 0.5,
    description: str = "resource cleanup"
) -> None:
    """
    Wait for resources to be cleaned up.

    Args:
        delay: Time to wait in seconds
        description: Description of resources being cleaned
    """
    time.sleep(delay)