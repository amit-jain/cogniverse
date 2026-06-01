"""Health check endpoints for runtime monitoring."""

import logging
from functools import lru_cache
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.get("/health")
async def health_check() -> Any:
    """Health check endpoint with system status.

    Returns 503 (not 500) when the service cannot assemble its health view —
    e.g. a missing BACKEND_URL makes create_default_config_manager raise. A
    monitoring probe should read this as unhealthy, not as a server crash.
    """
    try:
        # Reused across probes; backends/agents are still queried live below.
        agent_registry = _get_agent_registry()
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
async def readiness_probe() -> Any:
    """Kubernetes readiness probe - checks if service is ready to accept traffic.

    Returns HTTP 503 when not ready so a k8s readinessProbe (which gates on
    the status code, not the body) actually keeps the pod out of the Service
    until backends are registered.
    """
    backend_registry = BackendRegistry.get_instance()
    backends = backend_registry.list_backends()

    if not backends:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "No backends registered"},
        )

    return {
        "status": "ready",
        "backends": len(backends),
    }
