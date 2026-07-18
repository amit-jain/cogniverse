"""Health check endpoints for runtime monitoring."""

import logging
from functools import lru_cache
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

logger = logging.getLogger(__name__)

router = APIRouter()


async def _backend_reachable(
    base_url: str | None, timeout: float = 3.0
) -> tuple[bool, str]:
    """Probe the configured backend's container node so health reflects real
    connectivity.

    ``BackendRegistry.list_backends()`` only counts backend *class*
    registrations, and the Vespa backend self-registers at import time — so it
    is never empty and cannot indicate whether Vespa is actually reachable.
    Without this probe /health and /health/ready report healthy/ready through a
    total backend outage and k8s routes traffic to a runtime whose every query
    fails. ``base_url`` is the backend base the lifespan resolved from config
    (``app.state.backend_base_url``); None means startup has not wired it yet.
    """
    if not base_url:
        return False, "backend base URL not configured (startup incomplete)"
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/ApplicationStatus", timeout=timeout)
    except (httpx.HTTPError, OSError) as exc:
        return False, f"backend unreachable at {base_url}: {exc}"
    if resp.status_code != 200:
        return False, f"backend at {base_url} returned HTTP {resp.status_code}"
    return True, ""


@lru_cache(maxsize=1)
def _get_agent_registry() -> AgentRegistry:
    """Build the system AgentRegistry once and reuse it across probes.

    A k8s probe loop hits /health every few seconds; rebuilding the config
    stack (re-parsing config.json) and a fresh AgentRegistry (which opens an
    httpx.AsyncClient) per probe wastes work and leaks a client each time.
    lru_cache does not cache exceptions, so a failed build still retries (and
    surfaces as 503) on the next probe.
    """
    config_manager = create_default_config_manager()
    return AgentRegistry(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)


def _resolve_agent_registry() -> AgentRegistry:
    """Prefer the registry main.py injected into the agents router — the one
    /agents/ serves — so /health reports the runtime's live agents instead of
    a parallel system-tenant build. Falls back to the cached build only when
    startup hasn't wired the injection yet."""
    from cogniverse_runtime.routers import agents as agents_router

    try:
        return agents_router.get_registry()
    except RuntimeError:
        return _get_agent_registry()


@router.get("/health")
async def health_check(request: Request) -> Any:
    """Health check endpoint with system status.

    Returns 503 (not 500) when the service cannot assemble its health view —
    e.g. a missing BACKEND_URL makes create_default_config_manager raise. A
    monitoring probe should read this as unhealthy, not as a server crash.
    Also 503 when the backend is registered but unreachable, so monitoring
    goes red during a backend outage instead of showing green.
    """
    try:
        # Reused across probes; backends/agents are still queried live below.
        agent_registry = _resolve_agent_registry()
        backend_registry = BackendRegistry.get_instance()
        backends = backend_registry.list_backends()
        agents = agent_registry.list_agents()
    except Exception as exc:
        logger.warning("Health check could not assemble system status: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "cogniverse-runtime",
                "reason": str(exc),
            },
        )

    base_url = getattr(request.app.state, "backend_base_url", None)
    reachable, reason = await _backend_reachable(base_url)
    if not reachable:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "cogniverse-runtime",
                "reason": reason,
            },
        )

    return {
        "status": "healthy",
        "service": "cogniverse-runtime",
        "backends": {
            "registered": len(backends),
            "backends": backends,
        },
        "agents": {
            "registered": len(agents),
            "agents": agents,
        },
    }


@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe - checks if service is running."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe(request: Request) -> Any:
    """Kubernetes readiness probe - checks if service is ready to accept traffic.

    Returns HTTP 503 when not ready so a k8s readinessProbe (which gates on
    the status code, not the body) actually keeps the pod out of the Service
    until the backend is registered AND reachable. Registration alone is not
    enough: the Vespa backend class self-registers at import, so a
    registration-only check reports ready with Vespa completely down.
    """
    backend_registry = BackendRegistry.get_instance()
    backends = backend_registry.list_backends()

    if not backends:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "No backends registered"},
        )

    base_url = getattr(request.app.state, "backend_base_url", None)
    reachable, reason = await _backend_reachable(base_url)
    if not reachable:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": reason},
        )

    return {
        "status": "ready",
        "backends": len(backends),
    }
