"""Admin endpoints - system administration.

Note: Tenant management is available through the standalone tenant_manager app.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from cogniverse_core.registries.backend_registry import BackendRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


# Tenant management endpoints removed - use standalone tenant_manager app
# See: libs/runtime/cogniverse_runtime/admin/tenant_manager.py


@router.get("/system/stats")
async def get_system_stats(backend: str = "vespa") -> Dict[str, Any]:
    """Get system statistics."""
    try:
        backend_registry = BackendRegistry.get_instance()
        backend_instance = backend_registry.get_backend(backend)
        if not backend_instance:
            raise HTTPException(
                status_code=400, detail=f"Backend '{backend}' not found"
            )

        # Get basic stats from backend
        stats = {
            "backend": backend,
            "backend_type": backend.__class__.__name__,
        }

        # Add backend-specific stats if available
        if hasattr(backend_instance, "get_stats"):
            backend_stats = await backend_instance.get_stats()
            stats.update(backend_stats)

        return stats

    except Exception as e:
        logger.error(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
