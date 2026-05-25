"""
Production async polling utilities.

Provides semantic wait functions for production code to replace
generic time.sleep() calls with meaningful, purpose-specific waits.
"""

import time


def wait_for_retry_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    description: str = "retry backoff",
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
        delay = min(base_delay * (2**attempt), max_delay)
    else:
        delay = min(base_delay * (attempt + 1), max_delay)

    time.sleep(delay)
