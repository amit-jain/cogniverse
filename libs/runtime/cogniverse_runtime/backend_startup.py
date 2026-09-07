"""Backend readiness probes and guarded first-install metadata bootstrap."""

import asyncio
import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class BackendStartupState(StrEnum):
    FEED_READY = "feed_ready"
    FRESH_INSTALL = "fresh_install"
    UNAVAILABLE = "unavailable"


# Longest observed Vespa restart (unpruned data): 32 minutes. The wait budget
# carries a 2x margin for a cluster that has not pruned since an upgrade; the
# chart's runtime startupProbe window is pinned above it (tests/charts).
BACKEND_RECOVERY_WORST_CASE_S = 32 * 60
BACKEND_STARTUP_WAIT_BUDGET_S = 2 * BACKEND_RECOVERY_WORST_CASE_S
BACKEND_STARTUP_RETRY_INTERVAL_S = 5.0
BACKEND_STARTUP_PROBE_TIMEOUT_S = 5.0


async def _wait_for_backend_startup(
    vespa_base: str,
    config_server_base: str,
    *,
    budget_s: float = BACKEND_STARTUP_WAIT_BUDGET_S,
    retry_interval: float = BACKEND_STARTUP_RETRY_INTERVAL_S,
    timeout: float = BACKEND_STARTUP_PROBE_TIMEOUT_S,
) -> BackendStartupState:
    """Distinguish a ready data plane from a fresh Vespa installation.

    A fresh config server cannot expose ``/ApplicationStatus`` or document
    endpoints until its first application package is deployed. Polling only
    those endpoints therefore creates a startup cycle. The config-server
    application resource returns 404 only for that fresh state; 200 means an
    application exists and its data plane still needs to converge.

    The wait is bounded by wall clock, not attempts: a refused connection
    fails instantly and a paused backend takes the full probe timeout, so an
    attempt count would bound the wait anywhere between the two. The last
    attempt may start just before the deadline, so the wall-clock ceiling is
    ``budget_s + 3 * timeout`` (status, feed, and config server).
    The process entrypoint passes zero for one attempt; the shared startup
    helper owns its grace window, retry policy, and shutdown checks.
    """
    import httpx

    vespa_feed_probe = (
        f"{vespa_base}/document/v1/config_metadata/config_metadata/docid/probe"
    )
    application_resource = (
        f"{config_server_base}/application/v2/tenant/default/application/default"
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + budget_s
    attempt = 0
    async with httpx.AsyncClient() as client:
        while True:
            attempt += 1
            try:
                resp = await client.get(
                    f"{vespa_base}/ApplicationStatus", timeout=timeout
                )
                if resp.status_code != 200:
                    raise ConnectionError("Container node not ready")
                resp = await client.get(vespa_feed_probe, timeout=timeout)
                if resp.status_code in (200, 404):
                    return BackendStartupState.FEED_READY
            except (httpx.HTTPError, OSError, ConnectionError):
                pass
            try:
                resp = await client.get(application_resource, timeout=timeout)
                if resp.status_code == 404:
                    return BackendStartupState.FRESH_INSTALL
            except (httpx.HTTPError, OSError):
                pass
            now = loop.time()
            logger.info(
                "Backend not ready, retrying (attempt %d, %.0fs of %.0fs budget)...",
                attempt,
                now - started,
                budget_s,
            )
            if now >= deadline:
                return BackendStartupState.UNAVAILABLE
            await asyncio.sleep(min(retry_interval, deadline - now))


def _wait_for_config_server(
    host: str, port: int, *, max_attempts: int = 60, interval: float = 5.0
) -> bool:
    """Poll until the backend's config/deploy server accepts TCP connections.

    A cold Vespa opens its query port (8080) before its config/deploy server
    (19071), so a metadata deploy fired the instant the query port answers
    hits ``Connection refused`` on 19071. Waiting here keeps the retry
    IN-PROCESS — otherwise the deploy raises, the whole app startup exits,
    and the only thing retrying is the kubelet restarting the crashed pod
    (5+ crash-loops with full tracebacks before the config server is up).
    """
    import socket
    import time

    for _ in range(max_attempts):
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except OSError:
            time.sleep(interval)
    return False


def _application_exists(
    host: str, port: int, *, max_attempts: int = 6, interval: float = 5.0
) -> bool:
    """Ask the config server whether an application package is deployed.

    Discriminates a genuinely FRESH backend (404 → safe to bootstrap) from a
    populated one whose config read merely failed (200 → a registry-less
    metadata-only deploy would drop every tenant schema and lose their
    documents). An answer that is neither leaves fresh-vs-populated unknown —
    raise rather than deploy blind.
    """
    import time

    import httpx

    url = f"http://{host}:{port}/application/v2/tenant/default/application/default"
    last: object = None
    for attempt in range(max_attempts):
        try:
            resp = httpx.get(url, timeout=10)
        except httpx.HTTPError as exc:
            last = repr(exc)
        else:
            if resp.status_code == 200:
                return True
            if resp.status_code == 404:
                return False
            last = f"HTTP {resp.status_code}"
        if attempt < max_attempts - 1:
            time.sleep(interval)
    raise RuntimeError(
        f"Cannot determine whether {host}:{port} has an application deployed "
        f"(last answer: {last}) — refusing to bootstrap metadata schemas blind"
    )


def _bootstrap_metadata_schemas(bootstrap, application_name: str) -> None:
    """Deploy the metadata schemas to a backend with no application package.

    A fresh backend serves nothing on the query chain until the first
    application deploys, so every config read fails — the config_metadata
    schema is itself part of the metadata application. Runs BEFORE the
    first config read on first install only. Waits for the config/deploy
    server to accept connections first (a cold backend brings it up after
    the query port), then raises if the deploy itself fails (genuine
    outage / misconfig → fail fast).

    A failed config read is NOT proof of a fresh backend — a populated
    cluster mid cold-start or answering degraded fails the same way, and
    deploying the registry-less metadata-only package over it would remove
    every tenant content schema. Two guards make that impossible: the
    config server must report NO deployed application before anything is
    deployed, and the deploy runs with schema removal disabled so Vespa
    itself refuses a package that would drop schemas.

    Constructs the schema manager DIRECTLY: every registry/backend path
    reads the config store internally, which is exactly what cannot work
    yet on a fresh backend.
    """
    from urllib.parse import urlparse

    from cogniverse_vespa.config_utils import calculate_config_port
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    config_port = calculate_config_port(bootstrap.backend_port)
    host = urlparse(bootstrap.backend_url).hostname or bootstrap.backend_url
    if not _wait_for_config_server(host, config_port):
        raise RuntimeError(
            f"Backend config server {host}:{config_port} never accepted "
            "connections — cannot bootstrap metadata schemas"
        )

    if _application_exists(host, config_port):
        raise RuntimeError(
            f"Backend {host}:{config_port} already has an application deployed "
            "— the failed config read is a real outage, not a fresh install; "
            "a metadata-only deploy here would drop the existing tenant schemas"
        )

    manager = VespaSchemaManager(
        backend_endpoint=bootstrap.backend_url,
        backend_port=config_port,
        schema_registry=None,
    )
    manager.upload_metadata_schemas(
        app_name=application_name, allow_schema_removal=False
    )
    logger.info("Metadata schemas bootstrapped for fresh backend")
