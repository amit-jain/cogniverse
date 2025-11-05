"""Health check endpoints for runtime monitoring."""

import logging
from typing import Any, Dict

from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with system status."""
    config_manager = ConfigManager()
    backend_registry = BackendRegistry(config_manager=config_manager)
    agent_registry = AgentRegistry(config_manager=config_manager)

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
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe - checks if service is ready to accept traffic."""
    config_manager = ConfigManager()
    backend_registry = BackendRegistry(config_manager=config_manager)
    backends = backend_registry.list_backends()

    if not backends:
        return {
            "status": "not_ready",
            "reason": "No backends registered",
        }

    return {
        "status": "ready",
        "backends": len(backends),
    }
