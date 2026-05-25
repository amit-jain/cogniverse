"""Health check endpoints for runtime monitoring."""

import logging
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with system status."""
    config_manager = create_default_config_manager()
    backend_registry = BackendRegistry.get_instance()
    # Health probe is cluster-wide (no per-tenant filtering) — use
    # SYSTEM_TENANT_ID for the registry lookup.
    agent_registry = AgentRegistry(
        tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager
    )

    return {
        "status": "healthy",
        "service": "cogniverse-runtime",
        "backends": {
            "registered": len(backend_registry.list_backends()),
            "backends": backend_registry.list_backends(),
        },
        "agents": {
            "registered": len(agent_registry.list_agents()),
            "agents": agent_registry.list_agents(),
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
