"""
Production async polling utilities.

Provides semantic wait functions for production code to replace
generic time.sleep() calls with meaningful, purpose-specific waits.
"""

import time
from typing import Callable, Optional


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


def wait_for_retry_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    description: str = "retry backoff"
) -> None:
    """
    Wait with exponential or linear backoff for retries.

    This is the ONLY time.sleep wrapper we keep - it actually calculates
    exponential backoff delays for retry logic.

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
