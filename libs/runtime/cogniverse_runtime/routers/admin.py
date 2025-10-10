"""Admin endpoints - tenant management and system administration."""

import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin.tenant_manager import TenantManager

logger = logging.getLogger(__name__)

router = APIRouter()


class TenantCreateRequest(BaseModel):
    """Request to create a new tenant."""

    tenant_id: str
    org_id: str
    display_name: str
    backend: str = "vespa"
    config: Dict[str, Any] = {}


class TenantUpdateRequest(BaseModel):
    """Request to update tenant configuration."""

    display_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@router.post("/tenants")
async def create_tenant(request: TenantCreateRequest) -> Dict[str, Any]:
    """Create a new tenant with isolated namespace."""
    try:
        # Get backend
        backend_registry = BackendRegistry.get_instance()
        backend = backend_registry.get_backend(request.backend)
        if not backend:
            raise HTTPException(
                status_code=400, detail=f"Backend '{request.backend}' not found"
            )

        # Create tenant manager
        tenant_manager = TenantManager(backend=backend)

        # Create tenant
        result = await tenant_manager.create_tenant(
            tenant_id=request.tenant_id,
            org_id=request.org_id,
            display_name=request.display_name,
            config=request.config,
        )

        return {
            "status": "created",
            "tenant_id": request.tenant_id,
            "org_id": request.org_id,
            "namespace": result.get("namespace"),
        }

    except Exception as e:
        logger.error(f"Tenant creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tenants")
async def list_tenants(
    org_id: Optional[str] = None, backend: str = "vespa"
) -> Dict[str, Any]:
    """List all tenants, optionally filtered by org."""
    try:
        backend_registry = BackendRegistry.get_instance()
        backend_instance = backend_registry.get_backend(backend)
        if not backend_instance:
            raise HTTPException(
                status_code=400, detail=f"Backend '{backend}' not found"
            )

        tenant_manager = TenantManager(backend=backend_instance)
        tenants = await tenant_manager.list_tenants(org_id=org_id)

        return {"count": len(tenants), "tenants": tenants}

    except Exception as e:
        logger.error(f"List tenants error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tenants/{tenant_id}")
async def get_tenant(tenant_id: str, backend: str = "vespa") -> Dict[str, Any]:
    """Get tenant details."""
    try:
        backend_registry = BackendRegistry.get_instance()
        backend_instance = backend_registry.get_backend(backend)
        if not backend_instance:
            raise HTTPException(
                status_code=400, detail=f"Backend '{backend}' not found"
            )

        tenant_manager = TenantManager(backend=backend_instance)
        tenant = await tenant_manager.get_tenant(tenant_id=tenant_id)

        if not tenant:
            raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

        return tenant

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get tenant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/tenants/{tenant_id}")
async def update_tenant(
    tenant_id: str, request: TenantUpdateRequest, backend: str = "vespa"
) -> Dict[str, Any]:
    """Update tenant configuration."""
    try:
        backend_registry = BackendRegistry.get_instance()
        backend_instance = backend_registry.get_backend(backend)
        if not backend_instance:
            raise HTTPException(
                status_code=400, detail=f"Backend '{backend}' not found"
            )

        tenant_manager = TenantManager(backend=backend_instance)

        updates = {}
        if request.display_name:
            updates["display_name"] = request.display_name
        if request.config:
            updates["config"] = request.config

        result = await tenant_manager.update_tenant(tenant_id=tenant_id, updates=updates)

        return {
            "status": "updated",
            "tenant_id": tenant_id,
            "updates": updates,
        }

    except Exception as e:
        logger.error(f"Update tenant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tenants/{tenant_id}")
async def delete_tenant(tenant_id: str, backend: str = "vespa") -> Dict[str, Any]:
    """Delete a tenant and all its data."""
    try:
        backend_registry = BackendRegistry.get_instance()
        backend_instance = backend_registry.get_backend(backend)
        if not backend_instance:
            raise HTTPException(
                status_code=400, detail=f"Backend '{backend}' not found"
            )

        tenant_manager = TenantManager(backend=backend_instance)
        await tenant_manager.delete_tenant(tenant_id=tenant_id)

        return {
            "status": "deleted",
            "tenant_id": tenant_id,
        }

    except Exception as e:
        logger.error(f"Delete tenant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
