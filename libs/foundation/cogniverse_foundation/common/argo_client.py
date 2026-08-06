"""Shared HTTP client for talking to Argo Workflows.

``argo-server`` runs in secure mode behind a self-signed in-cluster
certificate, so a default-verifying client fails every request with
``CERTIFICATE_VERIFY_FAILED`` before it reaches the API. The submission paths
each discovered this separately, and the one that did not — the tenant
job-scheduling route — answered 503 on every scheduled job with
``Argo unreachable ... certificate verify failed``.

Build Argo clients here rather than constructing ``httpx`` clients inline, so
that posture is decided once and a new caller inherits it instead of
rediscovering the failure. The connection never leaves the cluster.
"""

from __future__ import annotations

import httpx

# Submissions are small control-plane calls; a request that has not completed
# in this window means argo-server is wedged, and the caller should surface
# that rather than hold its own request open.
DEFAULT_ARGO_TIMEOUT = httpx.Timeout(10.0)


def build_argo_async_client(
    timeout: httpx.Timeout | float | None = None,
) -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` that can reach argo-server over TLS."""
    return httpx.AsyncClient(
        timeout=DEFAULT_ARGO_TIMEOUT if timeout is None else timeout,
        verify=False,
    )
