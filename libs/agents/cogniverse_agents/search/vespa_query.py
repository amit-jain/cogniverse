"""Retried Vespa ``/search/`` POST shared by the media search agents.

The image / audio / document agents each POSTed directly to Vespa with a bare
``requests.post`` and turned any non-200 into ``return []`` — so a transient
connection reset or a 5xx during an outage read as "no results" with no retry.
This helper retries transient failures and RAISES on a persistent 5xx or
connection error so the outage surfaces instead of masquerading as empty
results. 4xx responses (a malformed query) are returned for the caller to
handle.
"""

from __future__ import annotations

import logging

import requests

from cogniverse_core.common.utils.retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)

_SEARCH_RETRY = RetryConfig(max_attempts=3, exceptions=(requests.RequestException,))


class VespaSearchDegraded(RuntimeError):
    """A Vespa query returned soft-timeout errors (HTTP 200 + root.errors)."""


class VespaSearchError(RuntimeError):
    """A Vespa ``/search/`` query returned a persistent non-2xx status.

    ``vespa_search_post`` retries transient failures and raises on a persistent
    5xx; a 4xx (a malformed query — schema/dim drift) comes back for the caller
    to surface. Returning [] instead read a hard query error as "no results".
    """


def vespa_search_children(data: dict, correlation_id: str = "") -> list:
    """Return ``root.children``, raising when the query was degraded.

    A Vespa soft-timeout arrives as HTTP 200 + ``root.errors`` (code 12) +
    partial/empty ``children``. Iterating those as if complete silently
    returns degraded results. Mirror ``search_backend._process_results``:
    raise on errors, warn on degraded coverage, else return the hits.
    """
    root = (data or {}).get("root", {}) or {}
    errors = root.get("errors")
    if errors:
        tag = f" [{correlation_id}]" if correlation_id else ""
        raise VespaSearchDegraded(f"Vespa query returned errors{tag}: {errors}")
    coverage = root.get("coverage", {})
    if not isinstance(coverage, dict):
        # A non-object coverage shape must not crash result parsing.
        coverage = {}
    if coverage.get("degraded"):
        logger.warning("Vespa coverage degraded: %s", coverage.get("degraded"))
    return root.get("children", []) or []


@retry_with_backoff(config=_SEARCH_RETRY)
def vespa_search_post(
    endpoint: str, params: dict, timeout: float = 10.0
) -> requests.Response:
    """POST ``params`` to ``{endpoint}/search/``, retrying transient failures."""
    resp = requests.post(
        f"{endpoint.rstrip('/')}/search/", json=params, timeout=timeout
    )
    if resp.status_code >= 500:
        # Server/backend error — raise so the retry wrapper backs off and, if it
        # persists, propagates instead of being flattened to no-results.
        raise requests.HTTPError(
            f"Vespa search returned {resp.status_code}: {resp.text[:200]}"
        )
    return resp
