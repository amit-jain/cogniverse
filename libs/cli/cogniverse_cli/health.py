"""Health-check utilities for Cogniverse services.

Provides polling helpers used by the CLI to wait for services to become
ready after deployment.
"""

from __future__ import annotations

import time

import httpx


def wait_for_url(
    url: str,
    *,
    timeout: float = 300,
    interval: float = 5,
) -> bool:
    """Poll *url* until it returns HTTP 200 or *timeout* seconds elapse.

    Returns ``True`` when the endpoint is healthy, ``False`` on timeout.
    Connection errors and non-200 responses are silently retried.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=interval, verify=False)
            if resp.status_code in (200, 401, 403):
                # 401/403 means the service is up but requires auth (e.g. Argo)
                return True
        except (httpx.HTTPError, OSError):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    return False


def check_service_health(services: dict[str, str]) -> dict[str, bool]:
    """Check multiple services in a single pass.

    *services* maps a human-readable service name to its health-check
    URL.  Returns a dict with the same keys, where each value is
    ``True`` (healthy / HTTP 200) or ``False``.

    Unlike :func:`wait_for_url` this function makes a **single** attempt
    per service with a short timeout — it is meant for a quick status
    snapshot, not for waiting.
    """
    results: dict[str, bool] = {}
    for name, url in services.items():
        try:
            resp = httpx.get(url, timeout=5, verify=False)
            results[name] = resp.status_code in (200, 401, 403)
        except (httpx.HTTPError, OSError):
            results[name] = False
    return results
