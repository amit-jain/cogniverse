"""
Vespa health check utilities for tests.
"""

import logging
import time
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


def calculate_reverse_backoff_intervals(
    max_timeout_seconds: int, min_interval: float = 1.0
) -> list[float]:
    """
    Calculate reverse exponential backoff intervals.

    Starts with longer waits (Vespa definitely not ready), then shorter intervals.

    Args:
        max_timeout_seconds: Maximum total timeout in seconds
        min_interval: Minimum interval between checks (default 1.0s)

    Returns:
        List of wait intervals in seconds

    Example:
        >>> calculate_reverse_backoff_intervals(60)
        [10, 7, 5, 3, 2, 1, 1, 1, ...]  # sums to ~60s
    """
    if max_timeout_seconds < min_interval:
        return [max_timeout_seconds]

    intervals = []
    total = 0.0

    # Start with larger intervals: 10, 7, 5, 3, 2
    initial_intervals = [10, 7, 5, 3, 2]
    for interval in initial_intervals:
        if total + interval <= max_timeout_seconds:
            intervals.append(float(interval))
            total += interval
        else:
            break

    # Fill remaining time with min_interval (usually 1s)
    while total + min_interval <= max_timeout_seconds:
        intervals.append(min_interval)
        total += min_interval

    # Add final partial interval if needed
    remaining = max_timeout_seconds - total
    if remaining > 0:
        intervals.append(remaining)

    return intervals


def wait_for_vespa_ready(
    port: int,
    max_timeout: int = 60,
    health_check_url: Optional[str] = None,
    check_fn: Optional[Callable[[], bool]] = None,
) -> bool:
    """
    Wait for Vespa to be ready using reverse exponential backoff.

    Args:
        port: Vespa HTTP port
        max_timeout: Maximum timeout in seconds (default 60)
        health_check_url: Optional custom health check URL
        check_fn: Optional custom check function (returns True if ready)

    Returns:
        True if Vespa became ready, raises RuntimeError otherwise

    Raises:
        RuntimeError: If Vespa did not become ready within max_timeout
    """
    if health_check_url is None:
        health_check_url = (
            f"http://localhost:{port}/document/v1/"
            "organization_metadata/organization_metadata/docid/_healthcheck"
        )

    intervals = calculate_reverse_backoff_intervals(max_timeout)

    for attempt, wait_time in enumerate(intervals):
        time.sleep(wait_time)

        try:
            if check_fn:
                ready = check_fn()
            else:
                # Default: check if document API responds
                response = requests.get(health_check_url, timeout=2)
                ready = True  # Any response means API is alive

            if ready:
                total_wait = sum(intervals[: attempt + 1])
                logger.info(
                    f"Vespa ready after {attempt + 1} attempts "
                    f"(waited {total_wait:.1f}s total)"
                )
                return True

        except Exception as e:
            if attempt < len(intervals) - 1:
                logger.debug(
                    f"Vespa not ready yet (attempt {attempt + 1}, "
                    f"next wait: {intervals[attempt + 1]:.1f}s): {e}"
                )
            else:
                total_wait = sum(intervals)
                raise RuntimeError(
                    f"Vespa did not become ready after {total_wait:.1f}s "
                    f"({attempt + 1} attempts)"
                )

    return False
