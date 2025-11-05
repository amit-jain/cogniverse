"""
Async polling utilities for test infrastructure.

Replaces blocking time.sleep() calls with proper async/await patterns
and exponential backoff for service readiness checks.
"""

import asyncio
import logging
import time
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


class PollingTimeoutError(Exception):
    """Raised when polling times out waiting for a condition"""

    pass


async def wait_for_condition(
    condition_fn: Callable[[], bool],
    timeout: float = 120.0,
    interval: float = 0.5,
    max_interval: float = 5.0,
    backoff_factor: float = 1.5,
    description: str = "condition",
) -> bool:
    """
    Wait for a condition function to return True with exponential backoff.

    Args:
        condition_fn: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Initial polling interval in seconds
        max_interval: Maximum polling interval in seconds
        backoff_factor: Factor to increase interval by after each attempt
        description: Description of what we're waiting for (for logging)

    Returns:
        True if condition met, False if timeout

    Raises:
        PollingTimeoutError: If condition not met within timeout
    """
    start_time = time.time()
    current_interval = interval
    attempt = 0

    while True:
        try:
            if condition_fn():
                elapsed = time.time() - start_time
                logger.info(
                    f"✅ {description} ready after {elapsed:.1f}s ({attempt} attempts)"
                )
                return True
        except Exception as e:
            logger.debug(f"Condition check failed: {e}")

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise PollingTimeoutError(
                f"Timeout waiting for {description} after {elapsed:.1f}s ({attempt} attempts)"
            )

        attempt += 1
        await asyncio.sleep(current_interval)

        # Exponential backoff
        current_interval = min(current_interval * backoff_factor, max_interval)


def wait_for_condition_sync(
    condition_fn: Callable[[], bool],
    timeout: float = 120.0,
    interval: float = 0.5,
    max_interval: float = 5.0,
    backoff_factor: float = 1.5,
    description: str = "condition",
) -> bool:
    """
    Synchronous version of wait_for_condition for non-async contexts.

    Args:
        condition_fn: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Initial polling interval in seconds
        max_interval: Maximum polling interval in seconds
        backoff_factor: Factor to increase interval by after each attempt
        description: Description of what we're waiting for (for logging)

    Returns:
        True if condition met

    Raises:
        PollingTimeoutError: If condition not met within timeout
    """
    start_time = time.time()
    current_interval = interval
    attempt = 0

    while True:
        try:
            if condition_fn():
                elapsed = time.time() - start_time
                logger.info(
                    f"✅ {description} ready after {elapsed:.1f}s ({attempt} attempts)"
                )
                return True
        except Exception as e:
            logger.debug(f"Condition check failed: {e}")

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise PollingTimeoutError(
                f"Timeout waiting for {description} after {elapsed:.1f}s ({attempt} attempts)"
            )

        attempt += 1
        time.sleep(current_interval)

        # Exponential backoff
        current_interval = min(current_interval * backoff_factor, max_interval)


def wait_for_http_ready(
    url: str,
    timeout: float = 120.0,
    expected_status: int = 200,
    description: Optional[str] = None,
) -> bool:
    """
    Wait for HTTP endpoint to return expected status code.

    Args:
        url: URL to check
        timeout: Maximum time to wait in seconds
        expected_status: Expected HTTP status code
        description: Description for logging (defaults to URL)

    Returns:
        True when endpoint is ready

    Raises:
        PollingTimeoutError: If endpoint not ready within timeout
    """
    desc = description or f"HTTP endpoint {url}"

    def check_http():
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == expected_status
        except requests.RequestException:
            return False

    return wait_for_condition_sync(
        condition_fn=check_http,
        timeout=timeout,
        description=desc,
    )


def wait_for_docker_healthy(
    container_name: str,
    timeout: float = 120.0,
) -> bool:
    """
    Wait for Docker container to be healthy.

    Args:
        container_name: Name of Docker container
        timeout: Maximum time to wait in seconds

    Returns:
        True when container is healthy

    Raises:
        PollingTimeoutError: If container not healthy within timeout
    """
    import subprocess

    def check_health():
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format={{.State.Health.Status}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                status = result.stdout.strip()
                # Container might not have health check defined
                if status == "healthy":
                    return True
                # If no health check, check if it's running
                if status == "" or status == "<no value>":
                    result = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format={{.State.Running}}",
                            container_name,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    return result.returncode == 0 and result.stdout.strip() == "true"
            return False
        except Exception:
            return False

    return wait_for_condition_sync(
        condition_fn=check_health,
        timeout=timeout,
        description=f"Docker container {container_name}",
    )


def wait_for_vespa_query_ready(
    vespa_url: str,
    schema_name: str,
    timeout: float = 30.0,
) -> bool:
    """
    Wait for Vespa schema to be queryable.

    Tests that documents can be queried from the schema (handles indexing delays).

    Args:
        vespa_url: Vespa endpoint URL (e.g., http://localhost:8080)
        schema_name: Schema/document type name
        timeout: Maximum time to wait in seconds

    Returns:
        True when schema is queryable

    Raises:
        PollingTimeoutError: If schema not queryable within timeout
    """

    def check_query():
        try:
            query_url = f"{vespa_url}/search/"
            params = {
                "yql": f"select * from {schema_name} where true limit 0",
            }
            response = requests.get(query_url, params=params, timeout=5)
            # Schema is queryable if we get 200 (even with 0 results)
            return response.status_code == 200
        except requests.RequestException:
            return False

    return wait_for_condition_sync(
        condition_fn=check_query,
        timeout=timeout,
        description=f"Vespa schema {schema_name} queryable",
    )


def wait_for_vespa_document_visible(
    vespa_url: str,
    schema_name: str,
    document_id: str,
    timeout: float = 30.0,
) -> bool:
    """
    Wait for a specific document to be visible in Vespa queries.

    Handles Vespa's eventual consistency - document might be fed but not yet indexed.

    Args:
        vespa_url: Vespa endpoint URL (e.g., http://localhost:8080)
        schema_name: Schema/document type name
        document_id: Document ID to check for
        timeout: Maximum time to wait in seconds

    Returns:
        True when document is visible in queries

    Raises:
        PollingTimeoutError: If document not visible within timeout
    """

    def check_document():
        try:
            query_url = f"{vespa_url}/search/"
            # Use matches for string attribute queries (works for UUIDs with hyphens)
            params = {
                "yql": f'select * from {schema_name} where id matches "{document_id}" limit 1',
            }
            response = requests.get(query_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                total_count = (
                    data.get("root", {}).get("fields", {}).get("totalCount", 0)
                )
                return total_count > 0
            return False
        except requests.RequestException:
            return False

    return wait_for_condition_sync(
        condition_fn=check_document,
        timeout=timeout,
        description=f"Vespa document {document_id} in {schema_name}",
    )


def wait_for_vespa_indexing(
    vespa_url: str = "http://localhost:8080",
    delay: float = 2.0,
    description: str = "Vespa indexing",
) -> bool:
    """
    Simple delay replacement for time.sleep() in Vespa indexing waits.

    This is a transitional function that provides a centralized place to wait
    for Vespa indexing, making it easier to optimize in the future.

    Args:
        vespa_url: Vespa endpoint (for future optimization)
        delay: How long to wait (backwards compatible with time.sleep)
        description: What we're waiting for

    Returns:
        True when delay has passed
    """
    logger.debug(f"Waiting {delay}s for {description}")
    time.sleep(delay)
    return True


def wait_for_telemetry_processing(
    delay: float = 2.0,
    description: str = "telemetry span processing",
) -> bool:
    """
    Wait for telemetry backend to process spans.

    Telemetry backends process spans asynchronously, so we need to wait for them
    to be available for queries.

    Args:
        delay: How long to wait
        description: What we're waiting for

    Returns:
        True when delay has passed
    """
    logger.debug(f"Waiting {delay}s for {description}")
    time.sleep(delay)
    return True


# Backwards compatibility alias
wait_for_phoenix_processing = wait_for_telemetry_processing


def wait_for_cache_expiration(
    ttl: float,
    buffer: float = 0.1,
    description: str = "cache TTL expiration",
) -> bool:
    """
    Wait for a cache entry to expire based on its TTL.

    Args:
        ttl: Time-to-live in seconds
        buffer: Additional buffer time to ensure expiration
        description: What cache we're waiting for

    Returns:
        True when TTL + buffer has passed
    """
    wait_time = ttl + buffer
    logger.debug(f"Waiting {wait_time}s for {description}")
    time.sleep(wait_time)
    return True


def wait_for_service_startup(
    check_fn: Optional[Callable[[], bool]] = None,
    delay: float = 0.1,
    description: str = "service startup",
) -> bool:
    """
    Wait for a service to start up.

    Args:
        check_fn: Optional function to check if service is ready
        delay: Fallback delay if no check function
        description: What service we're waiting for

    Returns:
        True when service is ready or delay has passed
    """
    if check_fn:
        return wait_for_condition_sync(
            condition_fn=check_fn,
            timeout=30.0,
            interval=0.1,
            description=description,
        )
    else:
        logger.debug(f"Waiting {delay}s for {description}")
        time.sleep(delay)
        return True


def simulate_processing_delay(
    delay: float = 0.01,
    description: str = "simulated processing",
) -> bool:
    """
    Simulate processing delay in unit tests.

    This is explicitly for test purposes where we need to simulate
    some processing time.

    Args:
        delay: How long to simulate
        description: What we're simulating

    Returns:
        True when delay has passed
    """
    logger.debug(f"Simulating {delay}s delay for {description}")
    time.sleep(delay)
    return True
