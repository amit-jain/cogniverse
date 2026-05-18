"""Admin endpoints - system administration and profile management."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.validation.profile_validator import ProfileValidator
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_runtime.admin.profile_models import (
    ProfileCreateRequest,
    ProfileCreateResponse,
    ProfileDeleteResponse,
    ProfileDetail,
    ProfileListResponse,
    ProfileSummary,
    ProfileUpdateRequest,
    ProfileUpdateResponse,
    SchemaDeploymentRequest,
    SchemaDeploymentResponse,
)
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()

_config_manager: ConfigManager | None = None
_schema_loader: SchemaLoader | None = None
_profile_validator_schema_dir = None  # Path or None


def set_config_manager(config_manager: ConfigManager) -> None:
    """Set ConfigManager for this module (for tests)."""
    global _config_manager
    _config_manager = config_manager


def set_schema_loader(schema_loader: SchemaLoader) -> None:
    """Set SchemaLoader for this module (for tests)."""
    global _schema_loader
    _schema_loader = schema_loader


def set_profile_validator_schema_dir(schema_dir) -> None:
    """Set ProfileValidator schema directory for this module (for tests)."""
    from pathlib import Path

    global _profile_validator_schema_dir
    _profile_validator_schema_dir = Path(schema_dir) if schema_dir is not None else None


def reset_dependencies() -> None:
    """Reset all module-level dependencies (for tests)."""
    global _config_manager, _schema_loader, _profile_validator_schema_dir
    _config_manager = None
    _schema_loader = None
    _profile_validator_schema_dir = None


def get_config_manager_dependency() -> ConfigManager:
    """
    FastAPI dependency for ConfigManager.

    This function should be overridden in main.py using app.dependency_overrides.
    For tests, use set_config_manager() to inject a test instance.

    Returns:
        ConfigManager instance

    Raises:
        RuntimeError: If not overridden via app.dependency_overrides or test injection
    """
    if _config_manager is not None:
        return _config_manager

    raise RuntimeError(
        "ConfigManager dependency not configured. "
        "Override this dependency in main.py using app.dependency_overrides "
        "or call set_config_manager() in tests."
    )


def get_schema_loader_dependency() -> SchemaLoader:
    """
    FastAPI dependency for SchemaLoader.

    This function should be overridden in main.py using app.dependency_overrides.
    For tests, use set_schema_loader() to inject a test instance.

    Returns:
        SchemaLoader instance

    Raises:
        RuntimeError: If not overridden via app.dependency_overrides or test injection
    """
    if _schema_loader is not None:
        return _schema_loader

    raise RuntimeError(
        "SchemaLoader dependency not configured. "
        "Override this dependency in main.py using app.dependency_overrides "
        "or call set_schema_loader() in tests."
    )


def get_profile_validator_dependency(
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> ProfileValidator:
    """
    FastAPI dependency for ProfileValidator.

    Args:
        config_manager: ConfigManager instance (injected)

    Returns:
        ProfileValidator instance

    Note:
        For test overrides, use set_profile_validator_schema_dir() to set schema directory,
        or override this dependency in app.dependency_overrides
    """
    return ProfileValidator(
        config_manager, schema_templates_dir=_profile_validator_schema_dir
    )


@router.get("/system/stats")
async def get_system_stats(
    tenant_id: Optional[str] = Query(None),
    backend: Optional[str] = Query(None),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Dict[str, Any]:
    """Get system statistics.

    Without parameters, returns general system stats.
    With tenant_id and backend, returns backend-specific stats.
    """
    try:
        backend_registry = BackendRegistry.get_instance()

        stats: Dict[str, Any] = {
            "registered_backends": list(backend_registry.list_backends()),
            "timestamp": datetime.now().isoformat(),
        }

        if backend and tenant_id:
            backend_instance = backend_registry.get_ingestion_backend(
                backend,
                tenant_id=tenant_id,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )
            if not backend_instance:
                raise HTTPException(
                    status_code=400, detail=f"Backend '{backend}' not found"
                )

            stats["backend"] = backend
            stats["tenant_id"] = tenant_id
            stats["backend_type"] = backend_instance.__class__.__name__

            if not hasattr(backend_instance, "get_stats"):
                raise HTTPException(
                    status_code=501,
                    detail=f"Backend '{backend}' does not implement get_stats()",
                )
            backend_stats = await backend_instance.get_stats()
            stats.update(backend_stats)

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles", response_model=ProfileCreateResponse, status_code=201)
async def create_profile(
    request: ProfileCreateRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
    validator: ProfileValidator = Depends(get_profile_validator_dependency),
) -> ProfileCreateResponse:
    """
    Create a new backend profile.

    Creates a profile configuration and optionally deploys the schema to Vespa.
    Profiles are tenant-scoped and versioned.

    Args:
        request: Profile creation request
        config_manager: ConfigManager instance (injected)
        schema_loader: SchemaLoader instance (injected)
        validator: ProfileValidator instance (injected)

    Returns:
        Profile creation response with deployment status

    Raises:
        HTTPException 400: Validation errors
        HTTPException 409: Profile already exists
        HTTPException 500: Creation or deployment failed
    """
    try:
        profile = BackendProfileConfig(
            profile_name=request.profile_name,
            type=request.type,
            description=request.description,
            schema_name=request.schema_name,
            embedding_model=request.embedding_model,
            pipeline_config=request.pipeline_config,
            strategies=request.strategies,
            embedding_type=request.embedding_type,
            schema_config=request.schema_config,
            model_specific=request.model_specific,
        )

        validation_errors = validator.validate_profile(
            profile, tenant_id=request.tenant_id, is_update=False
        )
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Profile validation failed",
                    "errors": validation_errors,
                },
            )

        config_manager.add_backend_profile(
            profile, tenant_id=request.tenant_id, service="backend"
        )

        schema_deployed = False
        tenant_schema_name = None

        if request.deploy_schema:
            backend_registry = BackendRegistry.get_instance()
            backend = backend_registry.get_ingestion_backend(
                "vespa",
                tenant_id=request.tenant_id,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )

            if not backend:
                raise HTTPException(
                    status_code=500,
                    detail="Backend not available for schema deployment",
                )

            backend.schema_registry.deploy_schema(
                tenant_id=request.tenant_id,
                base_schema_name=request.schema_name,
            )
            schema_deployed = True
            tenant_schema_name = backend.get_tenant_schema_name(
                request.tenant_id, request.schema_name
            )
            logger.info(
                f"Deployed schema '{tenant_schema_name}' for profile '{request.profile_name}'"
            )

        from cogniverse_sdk.interfaces.config_store import ConfigScope

        config_entry = config_manager.store.get_config(
            tenant_id=canonical_tenant_id(request.tenant_id),
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="backend_config",
        )
        actual_version = config_entry.version if config_entry else 1

        return ProfileCreateResponse(
            profile_name=request.profile_name,
            tenant_id=request.tenant_id,
            schema_deployed=schema_deployed,
            tenant_schema_name=tenant_schema_name,
            created_at=datetime.now().isoformat(),
            version=actual_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles", response_model=ProfileListResponse)
async def list_profiles(
    tenant_id: str,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> ProfileListResponse:
    """
    List all backend profiles for a tenant.

    Args:
        tenant_id: Tenant identifier (query parameter)
        config_manager: ConfigManager instance (injected)
        schema_loader: SchemaLoader instance (injected)

    Returns:
        List of profile summaries

    Raises:
        HTTPException 500: List operation failed
    """
    try:
        profiles = config_manager.list_backend_profiles(
            tenant_id=tenant_id, service="backend"
        )

        profile_summaries = []
        backend_registry = BackendRegistry.get_instance()

        for profile_name, profile in profiles.items():
            schema_deployed = False
            try:
                backend = backend_registry.get_ingestion_backend(
                    "vespa",
                    tenant_id=tenant_id,
                    config_manager=config_manager,
                    schema_loader=schema_loader,
                )
                if backend:
                    schema_deployed = backend.schema_exists(
                        schema_name=profile.schema_name, tenant_id=tenant_id
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to check schema deployment status for '{profile_name}': {e}"
                )

            profile_summaries.append(
                ProfileSummary(
                    profile_name=profile_name,
                    type=profile.type,
                    description=profile.description,
                    schema_name=profile.schema_name,
                    embedding_model=profile.embedding_model,
                    schema_deployed=schema_deployed,
                    created_at=datetime.now().isoformat(),  # config store does not persist creation time
                )
            )

        return ProfileListResponse(
            profiles=profile_summaries,
            total_count=len(profile_summaries),
            tenant_id=tenant_id,
        )

    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{profile_name}", response_model=ProfileDetail)
async def get_profile(
    profile_name: str,
    tenant_id: str,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> ProfileDetail:
    """
    Get a specific backend profile.

    Args:
        profile_name: Profile name (path parameter)
        tenant_id: Tenant identifier (query parameter)
        config_manager: ConfigManager instance (injected)
        schema_loader: SchemaLoader instance (injected)

    Returns:
        Detailed profile information

    Raises:
        HTTPException 404: Profile not found
        HTTPException 500: Get operation failed
    """
    try:
        profile = config_manager.get_backend_profile(
            profile_name=profile_name, tenant_id=tenant_id, service="backend"
        )

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found for tenant '{tenant_id}'",
            )

        schema_deployed = False
        tenant_schema_name = None

        try:
            backend_registry = BackendRegistry.get_instance()
            backend = backend_registry.get_ingestion_backend(
                "vespa",
                tenant_id=tenant_id,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )
            if backend:
                schema_deployed = backend.schema_exists(
                    schema_name=profile.schema_name, tenant_id=tenant_id
                )
                if schema_deployed:
                    tenant_schema_name = backend.get_tenant_schema_name(
                        tenant_id, profile.schema_name
                    )
        except Exception as e:
            logger.warning(f"Failed to check schema status: {e}")

        from cogniverse_sdk.interfaces.config_store import ConfigScope

        config_entry = config_manager.store.get_config(
            tenant_id=canonical_tenant_id(tenant_id),
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="backend_config",
        )
        config_version = config_entry.version if config_entry else 1

        return ProfileDetail(
            profile_name=profile.profile_name,
            tenant_id=tenant_id,
            type=profile.type,
            description=profile.description,
            schema_name=profile.schema_name,
            embedding_model=profile.embedding_model,
            pipeline_config=profile.pipeline_config,
            strategies=profile.strategies,
            embedding_type=profile.embedding_type,
            schema_config=profile.schema_config,
            model_specific=profile.model_specific,
            schema_deployed=schema_deployed,
            tenant_schema_name=tenant_schema_name,
            created_at=config_entry.created_at.isoformat()
            if config_entry
            else datetime.now().isoformat(),
            version=config_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profiles/{profile_name}", response_model=ProfileUpdateResponse)
async def update_profile(
    profile_name: str,
    request: ProfileUpdateRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    validator: ProfileValidator = Depends(get_profile_validator_dependency),
) -> ProfileUpdateResponse:
    """
    Update a backend profile.

    Only mutable fields can be updated (pipeline_config, strategies, description).
    Schema-related fields cannot be updated - create a new profile instead.

    Args:
        profile_name: Profile name (path parameter)
        request: Update request with fields to change
        config_manager: ConfigManager instance (injected)
        validator: ProfileValidator instance (injected)

    Returns:
        Update response with updated fields

    Raises:
        HTTPException 400: Invalid update (trying to update immutable fields)
        HTTPException 404: Profile not found
        HTTPException 500: Update operation failed
    """
    try:
        profile = config_manager.get_backend_profile(
            profile_name=profile_name,
            tenant_id=request.tenant_id,
            service="backend",
        )

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found for tenant '{request.tenant_id}'",
            )

        overrides = {}
        updated_fields = []

        if request.pipeline_config is not None:
            overrides["pipeline_config"] = request.pipeline_config
            updated_fields.append("pipeline_config")

        if request.strategies is not None:
            overrides["strategies"] = request.strategies
            updated_fields.append("strategies")

        if request.description is not None:
            overrides["description"] = request.description
            updated_fields.append("description")

        if request.model_specific is not None:
            overrides["model_specific"] = request.model_specific
            updated_fields.append("model_specific")

        if not overrides:
            raise HTTPException(status_code=400, detail="No fields to update provided")

        validation_errors = validator.validate_update_fields(overrides)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid update fields",
                    "errors": validation_errors,
                },
            )

        config_manager.update_backend_profile(
            profile_name=profile_name,
            overrides=overrides,
            base_tenant_id=request.tenant_id,
            target_tenant_id=request.tenant_id,
            service="backend",
        )

        from cogniverse_sdk.interfaces.config_store import ConfigScope

        config_entry = config_manager.store.get_config(
            tenant_id=canonical_tenant_id(request.tenant_id),
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="backend_config",
        )
        actual_version = config_entry.version if config_entry else 1

        return ProfileUpdateResponse(
            profile_name=profile_name,
            tenant_id=request.tenant_id,
            updated_fields=updated_fields,
            version=actual_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profiles/{profile_name}", response_model=ProfileDeleteResponse)
async def delete_profile(
    profile_name: str,
    tenant_id: str,
    delete_schema: bool = False,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> ProfileDeleteResponse:
    """
    Delete a backend profile.

    Optionally delete the associated schema (use with caution).

    Args:
        profile_name: Profile name (path parameter)
        tenant_id: Tenant identifier (query parameter)
        delete_schema: Whether to also delete the schema (query parameter, default: false)
        config_manager: ConfigManager instance (injected)
        schema_loader: SchemaLoader instance (injected)

    Returns:
        Deletion confirmation

    Raises:
        HTTPException 404: Profile not found
        HTTPException 409: Cannot delete schema (other profiles using it)
        HTTPException 500: Deletion failed
    """
    try:
        profile = config_manager.get_backend_profile(
            profile_name=profile_name, tenant_id=tenant_id, service="backend"
        )

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found for tenant '{tenant_id}'",
            )

        schema_deleted = False

        if delete_schema:
            all_profiles = config_manager.list_backend_profiles(
                tenant_id=tenant_id, service="backend"
            )

            other_profiles_using_schema = [
                p_name
                for p_name, p in all_profiles.items()
                if p_name != profile_name and p.schema_name == profile.schema_name
            ]

            if other_profiles_using_schema:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot delete schema '{profile.schema_name}': "
                    f"other profiles using it: {other_profiles_using_schema}",
                )

            backend_registry = BackendRegistry.get_instance()
            backend = backend_registry.get_ingestion_backend(
                "vespa",
                tenant_id=tenant_id,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )
            if not backend:
                raise HTTPException(
                    status_code=500,
                    detail="Backend not available for schema deletion",
                )
            deleted_schemas = backend.delete_schema(
                schema_name=profile.schema_name, tenant_id=tenant_id
            )
            schema_deleted = len(deleted_schemas) > 0

        success = config_manager.delete_backend_profile(
            profile_name=profile_name, tenant_id=tenant_id, service="backend"
        )

        if not success:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete profile '{profile_name}'"
            )

        return ProfileDeleteResponse(
            profile_name=profile_name,
            tenant_id=tenant_id,
            schema_deleted=schema_deleted,
            deleted_at=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles/{profile_name}/deploy", response_model=SchemaDeploymentResponse)
async def deploy_profile_schema(
    profile_name: str,
    request: SchemaDeploymentRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> SchemaDeploymentResponse:
    """
    Deploy schema for a backend profile.

    Deploys the Vespa schema associated with the profile to the tenant's namespace.

    Args:
        profile_name: Profile name (path parameter)
        request: Deployment request
        config_manager: ConfigManager instance (injected)
        schema_loader: SchemaLoader instance (injected)

    Returns:
        Deployment status

    Raises:
        HTTPException 404: Profile not found
        HTTPException 500: Deployment failed
    """
    try:
        profile = config_manager.get_backend_profile(
            profile_name=profile_name,
            tenant_id=request.tenant_id,
            service="backend",
        )

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found for tenant '{request.tenant_id}'",
            )

        backend_registry = BackendRegistry.get_instance()
        backend = backend_registry.get_ingestion_backend(
            "vespa",
            tenant_id=request.tenant_id,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        if not backend:
            raise HTTPException(
                status_code=500, detail="Backend not available for schema deployment"
            )

        schema_exists = backend.schema_exists(
            schema_name=profile.schema_name, tenant_id=request.tenant_id
        )

        if schema_exists and not request.force:
            tenant_schema_name = backend.get_tenant_schema_name(
                request.tenant_id, profile.schema_name
            )
            return SchemaDeploymentResponse(
                profile_name=profile_name,
                tenant_id=request.tenant_id,
                schema_name=profile.schema_name,
                tenant_schema_name=tenant_schema_name,
                deployment_status="already_deployed",
                deployed_at=datetime.now().isoformat(),
            )

        try:
            backend.schema_registry.deploy_schema(
                tenant_id=request.tenant_id,
                base_schema_name=profile.schema_name,
                force=request.force,
            )

            tenant_schema_name = backend.get_tenant_schema_name(
                request.tenant_id, profile.schema_name
            )

            return SchemaDeploymentResponse(
                profile_name=profile_name,
                tenant_id=request.tenant_id,
                schema_name=profile.schema_name,
                tenant_schema_name=tenant_schema_name,
                deployment_status="success",
                deployed_at=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Schema deployment failed: {e}")
            return SchemaDeploymentResponse(
                profile_name=profile_name,
                tenant_id=request.tenant_id,
                schema_name=profile.schema_name,
                tenant_schema_name="",
                deployment_status="failed",
                deployed_at=datetime.now().isoformat(),
                error_message=str(e),
            )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Failed to deploy schema: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class InviteRequest(BaseModel):
    tenant_id: str
    expires_in_hours: int = 24


@router.post("/messaging/invite")
async def create_messaging_invite(
    request: InviteRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Dict[str, str]:
    """Generate an invite token for messaging gateway registration.

    Returns a token that a user sends to the Telegram bot via /start <token>
    to link their Telegram account to the specified tenant.
    """
    import uuid

    token = uuid.uuid4().hex
    expiry = (datetime.utcnow() + timedelta(hours=request.expires_in_hours)).isoformat()

    from cogniverse_sdk.interfaces.config_store import ConfigScope

    config_manager.set_config_value(
        tenant_id="_system",
        scope=ConfigScope.SYSTEM,
        service="messaging_gateway",
        config_key=f"invite_token_{token}",
        config_value={
            "tenant_id": request.tenant_id,
            "token": token,
            "expires_at": expiry,
            "used": False,
        },
    )

    return {"token": token, "tenant_id": request.tenant_id}


_ADMIN_TYPE_TO_NAMESPACE: Dict[str, str] = {
    "preference": "_user_memories",
    "strategy": "_strategy_store",
}

_ADMIN_ALL_NAMESPACES = ["_user_memories", "_strategy_store"]


@router.delete("/memories/{tenant_id}/{memory_id}")
async def admin_delete_memory(tenant_id: str, memory_id: str):
    """Admin: delete any memory by ID, regardless of namespace."""
    from cogniverse_core.memory.manager import Mem0MemoryManager

    mgr = Mem0MemoryManager(tenant_id)
    if not mgr.memory:
        raise HTTPException(status_code=503, detail="Memory backend not initialised")

    for ns in _ADMIN_ALL_NAMESPACES:
        if mgr.delete_memory(memory_id=memory_id, tenant_id=tenant_id, agent_name=ns):
            logger.info(
                "Admin deleted memory %s (ns=%s) for tenant %s",
                memory_id,
                ns,
                tenant_id,
            )
            return {"status": "deleted", "memory_id": memory_id}

    raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")


@router.delete("/memories/{tenant_id}")
async def admin_clear_memories(
    tenant_id: str,
    type: Optional[str] = Query(
        default=None, description="Memory type to clear: preference, strategy, or all"
    ),
):
    """Admin: clear memories by type. Can clear system memories (strategies)."""
    from cogniverse_core.memory.manager import Mem0MemoryManager

    mgr = Mem0MemoryManager(tenant_id)
    if not mgr.memory:
        raise HTTPException(status_code=503, detail="Memory backend not initialised")

    if type and type != "all":
        ns = _ADMIN_TYPE_TO_NAMESPACE.get(type)
        if ns is None:
            raise HTTPException(status_code=400, detail=f"Unknown memory type: {type}")
        mgr.clear_agent_memory(tenant_id=tenant_id, agent_name=ns)
        logger.info("Admin cleared '%s' memories for tenant %s", type, tenant_id)
        return {"status": "cleared", "type": type}

    for ns in _ADMIN_ALL_NAMESPACES:
        mgr.clear_agent_memory(tenant_id=tenant_id, agent_name=ns)
    logger.info("Admin cleared all memories for tenant %s", tenant_id)
    return {"status": "cleared", "type": "all"}


@router.delete("/tenants/{tenant_id}/sessions/{session_id}")
async def admin_drop_session(tenant_id: str, session_id: str):
    """End a session: hard-delete every EPHEMERAL_SESSION memory tagged with it.

    Schema-driven — only memories whose kind is registered with
    ``retention=EPHEMERAL_SESSION`` are eligible. Other kinds tagged with
    the same session_id are untouched. Returns per-kind deletion counts.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.memory.schema import build_default_registry

    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must be non-empty")

    mgr = Mem0MemoryManager(tenant_id)
    if not mgr.memory:
        raise HTTPException(status_code=503, detail="Memory backend not initialised")

    registry = build_default_registry()
    deleted_by_kind = mgr.drop_session(session_id, registry)
    total = sum(deleted_by_kind.values())
    logger.info(
        "Admin drop_session(%s) for tenant %s: deleted %d memories %s",
        session_id,
        tenant_id,
        total,
        deleted_by_kind,
    )
    return {
        "status": "dropped",
        "tenant_id": tenant_id,
        "session_id": session_id,
        "deleted_by_kind": deleted_by_kind,
        "total_deleted": total,
    }


@router.post("/sessions/{session_id}/close")
async def admin_close_session(session_id: str):
    """Fan-out session close: drop the session across every warm tenant.

    A user session can write EPHEMERAL_SESSION memories under any tenant
    the request touched. On session close (logout, ws-disconnect, idle
    timeout) the gateway POSTs here once and the runtime sweeps every
    warm ``Mem0MemoryManager`` calling ``drop_session(session_id)``.

    Tenants that have already been evicted from the warm LRU are skipped
    — their next access will deserialise from Vespa and the
    EPHEMERAL_SESSION rows still exist there. The next request that warms
    the manager and triggers a session-close webhook will sweep them.
    Operators who need a guaranteed sweep can call the per-tenant DELETE
    endpoint with the known tenant id; this fan-out is best-effort over
    the warm set.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.memory.schema import build_default_registry

    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must be non-empty")

    registry = build_default_registry()
    per_tenant: Dict[str, Dict[str, int]] = {}
    total = 0
    skipped: List[str] = []
    for mgr in list(Mem0MemoryManager._instances.values()):
        tenant_id = getattr(mgr, "tenant_id", None) or "unknown"
        if not getattr(mgr, "memory", None):
            skipped.append(tenant_id)
            continue
        try:
            deleted = mgr.drop_session(session_id, registry)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "drop_session failed for tenant %s session %s: %s",
                tenant_id,
                session_id,
                exc,
            )
            skipped.append(tenant_id)
            continue
        if deleted:
            per_tenant[tenant_id] = deleted
            total += sum(deleted.values())

    logger.info(
        "Admin close_session(%s) swept %d warm tenants, deleted %d memories",
        session_id,
        len(per_tenant),
        total,
    )
    return {
        "status": "closed",
        "session_id": session_id,
        "per_tenant": per_tenant,
        "total_deleted": total,
        "skipped_tenants": skipped,
    }


# ---------------------------------------------------------------------------
# Operability admin endpoints (pin quota / variant select / canary)
#
# These three groups of endpoints close the audit's "operator can't reach
# this" gap for pinning quotas, signature variants, and canary promotion.
# Pin quotas + variant selections live in a process-local override dict
# (good enough for the admin loop until a TenantConfig persistence layer
# for these specific keys exists). Canary actions go straight to
# ArtifactManager which persists them to Phoenix.
# ---------------------------------------------------------------------------


class PinQuotasUpdateRequest(BaseModel):
    user: Optional[int] = None
    tenant_admin: Optional[int] = None
    org_admin: Optional[int] = None


class PinQuotasResponse(BaseModel):
    tenant_id: str
    quotas: Dict[str, int]


def _default_pin_quotas() -> Dict[str, int]:
    from cogniverse_core.memory.pinning import PinQuotas

    d = PinQuotas()
    return {
        "user": d.user,
        "tenant_admin": d.tenant_admin,
        "org_admin": -1 if d.org_admin is None else d.org_admin,
    }


_pin_quota_overrides: Dict[str, Dict[str, int]] = {}
_signature_variant_overrides: Dict[str, Dict[str, str]] = {}


@router.get("/tenants/{tenant_id}/pin_quotas", response_model=PinQuotasResponse)
async def get_pin_quotas(tenant_id: str) -> PinQuotasResponse:
    """return effective pin quotas for a tenant."""
    blob = _pin_quota_overrides.get(tenant_id) or _default_pin_quotas()
    return PinQuotasResponse(tenant_id=tenant_id, quotas=blob)


@router.put("/tenants/{tenant_id}/pin_quotas", response_model=PinQuotasResponse)
async def set_pin_quotas(
    tenant_id: str, body: PinQuotasUpdateRequest
) -> PinQuotasResponse:
    """set per-role pin quotas for a tenant.

    Only non-None fields are updated. Negative values (other than
    org_admin's unlimited sentinel of -1) are rejected.
    """
    current = dict(_pin_quota_overrides.get(tenant_id) or _default_pin_quotas())
    if body.user is not None:
        if body.user < 0:
            raise HTTPException(400, "user quota must be >= 0")
        current["user"] = body.user
    if body.tenant_admin is not None:
        if body.tenant_admin < 0:
            raise HTTPException(400, "tenant_admin quota must be >= 0")
        current["tenant_admin"] = body.tenant_admin
    if body.org_admin is not None:
        current["org_admin"] = body.org_admin
    _pin_quota_overrides[tenant_id] = current
    logger.info("Updated pin quotas for tenant=%s: %s", tenant_id, current)
    return PinQuotasResponse(tenant_id=tenant_id, quotas=current)


# Memory pin / unpin / list endpoints. PinService was already wired
# into the lifecycle scheduler (so pinned memories survive cleanup), but the
# only way to actually pin a memory used to be the in-process Python API —
# meaning a tenant admin had no operational path to pin anything. These
# endpoints close that gap. The requester's role + actor_id ride in the
# request body; auth middleware (out of scope here) is responsible for
# rejecting requests whose claimed role does not match the caller's
# authenticated identity.


class PinCreateRequest(BaseModel):
    target_kind: str
    pinned_by: str  # Pinnable enum value: user / tenant_admin / org_admin
    actor_id: str


class PinUnpinRequest(BaseModel):
    requester_role: str  # Pinnable enum value
    actor_id: str


class PinRecordResponse(BaseModel):
    memory_id: str
    target_memory_id: str
    target_kind: str
    pinned_by: str
    pinned_by_actor: str


class PinListResponse(BaseModel):
    tenant_id: str
    pins: list[PinRecordResponse]


class PinUnpinResponse(BaseModel):
    tenant_id: str
    target_memory_id: str
    removed: int


def _get_pin_service(tenant_id: str):
    """Build a PinService bound to the tenant's Mem0 manager + registry.

    Constructs the registry from the default schema set so authority +
    quota checks fire correctly even when the underlying memory manager
    was lazily initialised without one.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.memory.pinning import PinQuotas, PinService
    from cogniverse_core.memory.schema import build_default_registry

    mgr = Mem0MemoryManager(tenant_id)
    if not mgr.memory:
        # Defer to the same lazy-init path the tenant router uses — keeps
        # admin pinning workable on k3d setups where memory isn't pre-wired.
        from cogniverse_runtime.routers import tenant as _tenant_router

        _tenant_router._lazy_init_memory(mgr, tenant_id)
    if not mgr.memory:
        raise HTTPException(
            status_code=503,
            detail=f"Memory backend not initialised for tenant {tenant_id}",
        )
    return PinService(
        mgr,
        build_default_registry(),
        quotas=PinQuotas.for_tenant(tenant_id),
    )


def _parse_pinnable(value: str) -> "object":
    from cogniverse_core.memory.pinning import Pinnable

    try:
        return Pinnable(value)
    except ValueError as exc:
        valid = ", ".join(p.value for p in Pinnable)
        raise HTTPException(
            400, f"invalid role {value!r}; expected one of: {valid}"
        ) from exc


@router.post(
    "/tenants/{tenant_id}/memories/{memory_id}/pin",
    response_model=PinRecordResponse,
)
async def pin_memory(
    tenant_id: str, memory_id: str, body: PinCreateRequest
) -> PinRecordResponse:
    """pin a memory so the lifecycle scheduler skips it.

    Authority and quota are enforced via the schema registry + PinQuotas.
    Returns the persisted PinRecord on success.
    """
    from cogniverse_core.memory.pinning import (
        PinAuthorityError,
        PinQuotaExceededError,
    )

    pinned_by = _parse_pinnable(body.pinned_by)
    if not body.actor_id.strip():
        raise HTTPException(400, "actor_id must be non-empty")
    svc = _get_pin_service(tenant_id)
    try:
        record = svc.pin(
            target_memory_id=memory_id,
            target_kind=body.target_kind,
            pinned_by=pinned_by,
            actor_id=body.actor_id,
            tenant_id=tenant_id,
        )
    except PinAuthorityError as exc:
        raise HTTPException(403, str(exc)) from exc
    except PinQuotaExceededError as exc:
        raise HTTPException(429, str(exc)) from exc
    logger.info(
        "Pinned memory tenant=%s memory_id=%s by=%s/%s",
        tenant_id,
        memory_id,
        body.pinned_by,
        body.actor_id,
    )
    return PinRecordResponse(
        memory_id=record.memory_id,
        target_memory_id=record.target_memory_id,
        target_kind=record.target_kind,
        pinned_by=record.pinned_by.value,
        pinned_by_actor=record.pinned_by_actor,
    )


@router.delete(
    "/tenants/{tenant_id}/memories/{memory_id}/pin",
    response_model=PinUnpinResponse,
)
async def unpin_memory(
    tenant_id: str, memory_id: str, body: PinUnpinRequest
) -> PinUnpinResponse:
    """remove pin records for a memory.

    Org admin can unpin anything; tenant admin can unpin tenant_admin+user
    pins; users can only unpin their own. Authority violations return 403.
    """
    from cogniverse_core.memory.pinning import PinAuthorityError, PinNotFoundError

    requester = _parse_pinnable(body.requester_role)
    if not body.actor_id.strip():
        raise HTTPException(400, "actor_id must be non-empty")
    svc = _get_pin_service(tenant_id)
    try:
        removed = svc.unpin(
            target_memory_id=memory_id,
            requester=requester,
            actor_id=body.actor_id,
            tenant_id=tenant_id,
        )
    except PinNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PinAuthorityError as exc:
        raise HTTPException(403, str(exc)) from exc
    logger.info(
        "Unpinned memory tenant=%s memory_id=%s requester=%s removed=%d",
        tenant_id,
        memory_id,
        body.requester_role,
        removed,
    )
    return PinUnpinResponse(
        tenant_id=tenant_id, target_memory_id=memory_id, removed=removed
    )


@router.get("/tenants/{tenant_id}/pins", response_model=PinListResponse)
async def list_pins(tenant_id: str) -> PinListResponse:
    """list all pin records for a tenant (audit + UI)."""
    svc = _get_pin_service(tenant_id)
    records = svc.list_pins(tenant_id)
    return PinListResponse(
        tenant_id=tenant_id,
        pins=[
            PinRecordResponse(
                memory_id=r.memory_id,
                target_memory_id=r.target_memory_id,
                target_kind=r.target_kind,
                pinned_by=r.pinned_by.value,
                pinned_by_actor=r.pinned_by_actor,
            )
            for r in records
        ],
    )


# Promote a tenant memory to the org trunk so every tenant in the
# org sees it. Schema sensitivity gates which kinds are promotable;
# Pinnable role gates which actors may promote. Org-shared by design;
# tenant_private memories are forbidden from promotion regardless of
# actor authority.


class PromoteToOrgTrunkRequest(BaseModel):
    actor_role: str  # Pinnable enum value: tenant_admin / org_admin
    actor_id: str


class PromoteToOrgTrunkResponse(BaseModel):
    source_tenant_id: str
    source_memory_id: str
    promoted_memory_id: str
    org_trunk_tenant_id: str


@router.post(
    "/tenants/{tenant_id}/memories/{memory_id}/promote_to_org_trunk",
    response_model=PromoteToOrgTrunkResponse,
)
async def promote_to_org_trunk(
    tenant_id: str, memory_id: str, body: PromoteToOrgTrunkRequest
) -> PromoteToOrgTrunkResponse:
    """copy a memory into the org trunk (admin-gated)."""
    from cogniverse_core.memory.federation import (
        FederationDeniedError,
        FederationService,
    )
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.memory.pinning import Pinnable
    from cogniverse_core.memory.schema import build_default_registry

    if not body.actor_id.strip():
        raise HTTPException(400, "actor_id must be non-empty")
    try:
        actor_role = Pinnable(body.actor_role)
    except ValueError as exc:
        valid = ", ".join(p.value for p in Pinnable)
        raise HTTPException(
            400, f"invalid actor_role {body.actor_role!r}; expected one of: {valid}"
        ) from exc

    # Locate the source memory in the tenant's store. We don't know
    # which agent_name owns it, so go through the tenant-wide get_all
    # (Mem0 doesn't require agent_id when user_id is given).
    source_mm = _get_pin_service(tenant_id)._mm  # reuse the lazy-init path
    try:
        rows_blob = source_mm.memory.get_all(user_id=tenant_id)
    except Exception as exc:
        raise HTTPException(503, f"could not list tenant memories: {exc}") from exc
    rows = (
        rows_blob.get("results", [])
        if isinstance(rows_blob, dict)
        else (rows_blob or [])
    )
    src = next((r for r in rows if str(r.get("id")) == memory_id), None)
    if src is None:
        raise HTTPException(404, f"memory {memory_id} not found in tenant {tenant_id}")

    svc = FederationService(
        memory_manager_factory=lambda tid: Mem0MemoryManager(tid),
        registry=build_default_registry(),
    )
    try:
        result = svc.promote_to_org_trunk(
            source_tenant_id=tenant_id,
            source_memory=src,
            actor_role=actor_role,
            actor_id=body.actor_id,
        )
    except FederationDeniedError as exc:
        raise HTTPException(403, str(exc)) from exc

    logger.info(
        "Promoted memory tenant=%s memory_id=%s by=%s/%s -> trunk=%s/%s",
        tenant_id,
        memory_id,
        body.actor_role,
        body.actor_id,
        result.org_trunk_tenant_id,
        result.promoted_memory_id,
    )
    return PromoteToOrgTrunkResponse(
        source_tenant_id=tenant_id,
        source_memory_id=result.source_memory_id,
        promoted_memory_id=result.promoted_memory_id,
        org_trunk_tenant_id=result.org_trunk_tenant_id,
    )


# Endorse a memory: bumps its trust score by a role-specific
# delta (user +0.05, tenant_admin +0.10, org_admin +0.20) and persists
# the new TrustRecord back to the memory's metadata. The audit endpoint
# reads endorsement counts off these records — without this
# write path they would always read zero.


class EndorseRequest(BaseModel):
    endorser_role: str  # user / tenant_admin / org_admin
    actor_id: str


class EndorseResponse(BaseModel):
    memory_id: str
    new_score: float
    endorsements: int


@router.post(
    "/tenants/{tenant_id}/memories/{memory_id}/endorse",
    response_model=EndorseResponse,
)
async def endorse_memory(
    tenant_id: str, memory_id: str, body: EndorseRequest
) -> EndorseResponse:
    """record an endorsement on a memory's trust record."""
    from cogniverse_core.memory.trust import (
        _ENDORSEMENT_DELTA,
        apply_endorsement,
        attach_trust_to_metadata,
        extract_trust,
    )

    if not body.actor_id.strip():
        raise HTTPException(400, "actor_id must be non-empty")
    # Role validation up-front so a bad input fails fast before we hit
    # the memory backend.
    if body.endorser_role not in _ENDORSEMENT_DELTA:
        valid = ", ".join(sorted(_ENDORSEMENT_DELTA))
        raise HTTPException(
            400,
            f"unknown endorser_role={body.endorser_role!r}; valid: {valid}",
        )

    source_mm = _get_pin_service(tenant_id)._mm  # reuse the lazy-init path
    try:
        rows_blob = source_mm.memory.get_all(user_id=tenant_id)
    except Exception as exc:
        raise HTTPException(503, f"could not list tenant memories: {exc}") from exc
    rows = (
        rows_blob.get("results", [])
        if isinstance(rows_blob, dict)
        else (rows_blob or [])
    )
    src = next((r for r in rows if str(r.get("id")) == memory_id), None)
    if src is None:
        raise HTTPException(404, f"memory {memory_id} not found in tenant {tenant_id}")

    trust = extract_trust(src)
    if trust is None:
        raise HTTPException(
            422,
            f"memory {memory_id} has no trust record; the schema enforcement "
            "path must run on the original write to attach one before "
            "endorsement is meaningful",
        )

    new_trust = apply_endorsement(trust, body.endorser_role)

    new_metadata = attach_trust_to_metadata(src.get("metadata") or {}, new_trust)
    try:
        source_mm.memory.update(
            memory_id=memory_id,
            data=src.get("memory") or src.get("text") or "",
            metadata=new_metadata,
        )
    except Exception as exc:
        raise HTTPException(503, f"trust update failed: {exc}") from exc

    logger.info(
        "Endorsed memory tenant=%s memory_id=%s by=%s/%s -> score=%.3f n=%d",
        tenant_id,
        memory_id,
        body.endorser_role,
        body.actor_id,
        new_trust.score,
        new_trust.endorsements,
    )
    return EndorseResponse(
        memory_id=memory_id,
        new_score=new_trust.score,
        endorsements=new_trust.endorsements,
    )


# restore a soft-deleted memory. The lifecycle scheduler flips
# `metadata.archived=true` when a kind hits its TTL but not yet 2*TTL,
# giving operators a window to pull a record back. After 2*TTL the
# scheduler hard-deletes — restore is no-op then.


class RestoreMemoryResponse(BaseModel):
    tenant_id: str
    memory_id: str
    restored: bool


@router.post(
    "/tenants/{tenant_id}/memories/{memory_id}/restore",
    response_model=RestoreMemoryResponse,
)
async def restore_memory(tenant_id: str, memory_id: str) -> RestoreMemoryResponse:
    """clear the archived flag on a soft-deleted memory."""
    source_mm = _get_pin_service(tenant_id)._mm  # reuse the lazy-init path
    ok = source_mm.restore_archived_memory(memory_id)
    if not ok:
        raise HTTPException(
            404,
            f"memory {memory_id} not found or not in archived state",
        )
    logger.info("Restored archived memory tenant=%s memory_id=%s", tenant_id, memory_id)
    return RestoreMemoryResponse(
        tenant_id=tenant_id, memory_id=memory_id, restored=True
    )


class SignatureVariantSelectRequest(BaseModel):
    variant_id: str


class SignatureVariantResponse(BaseModel):
    tenant_id: str
    selections: Dict[str, str]


@router.get(
    "/tenants/{tenant_id}/signature_variants",
    response_model=SignatureVariantResponse,
)
async def get_signature_variants(tenant_id: str) -> SignatureVariantResponse:
    """list per-agent variant selections for a tenant."""
    return SignatureVariantResponse(
        tenant_id=tenant_id,
        selections=dict(_signature_variant_overrides.get(tenant_id, {})),
    )


@router.put(
    "/tenants/{tenant_id}/signature_variants/{agent_type}",
    response_model=SignatureVariantResponse,
)
async def set_signature_variant(
    tenant_id: str,
    agent_type: str,
    body: SignatureVariantSelectRequest,
) -> SignatureVariantResponse:
    """pick the variant id this tenant uses for an agent."""
    if not body.variant_id.strip():
        raise HTTPException(400, "variant_id must be non-empty")
    selections = dict(_signature_variant_overrides.get(tenant_id, {}))
    selections[agent_type] = body.variant_id
    _signature_variant_overrides[tenant_id] = selections
    logger.info(
        "Tenant=%s now using variant=%r for agent=%s",
        tenant_id,
        body.variant_id,
        agent_type,
    )
    return SignatureVariantResponse(tenant_id=tenant_id, selections=selections)


class CanaryPromoteRequest(BaseModel):
    version: int
    traffic_pct: int = 10


class CanaryActionResponse(BaseModel):
    tenant_id: str
    agent_type: str
    state: Dict[str, Any]


def _build_artifact_manager(tenant_id: str):
    """Construct an ArtifactManager for an admin canary action.

    Uses ``PHOENIX_HTTP_ENDPOINT`` / ``PHOENIX_GRPC_ENDPOINT`` env vars
    so admin endpoints work the same way the rollback CLI does:
    cluster-default Phoenix in production, docker-managed Phoenix in
    integration tests.
    """
    import os as _os

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": _os.environ.get(
                "PHOENIX_HTTP_ENDPOINT", "http://localhost:6006"
            ),
            "grpc_endpoint": _os.environ.get("PHOENIX_GRPC_ENDPOINT", "localhost:4317"),
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@router.post(
    "/tenants/{tenant_id}/canary/{agent_type}/promote",
    response_model=CanaryActionResponse,
)
async def promote_canary(
    tenant_id: str, agent_type: str, body: CanaryPromoteRequest
) -> CanaryActionResponse:
    """promote a versioned artefact to canary at a traffic_pct."""
    am = _build_artifact_manager(tenant_id)
    try:
        state = await am.promote_to_canary(
            agent_type, version=body.version, traffic_pct=body.traffic_pct
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    logger.info(
        "Promoted canary tenant=%s agent=%s v%d at %d%%",
        tenant_id,
        agent_type,
        body.version,
        body.traffic_pct,
    )
    return CanaryActionResponse(tenant_id=tenant_id, agent_type=agent_type, state=state)


@router.post(
    "/tenants/{tenant_id}/canary/{agent_type}/retire",
    response_model=CanaryActionResponse,
)
async def retire_canary(
    tenant_id: str,
    agent_type: str,
    reason: str = Query("admin_retire"),
) -> CanaryActionResponse:
    """retire the active canary, returning to active-only routing."""
    am = _build_artifact_manager(tenant_id)
    state = await am.retire_canary(agent_type, reason=reason)
    logger.info(
        "Retired canary tenant=%s agent=%s reason=%s",
        tenant_id,
        agent_type,
        reason,
    )
    return CanaryActionResponse(tenant_id=tenant_id, agent_type=agent_type, state=state)


def _reset_admin_overrides_for_tests() -> None:
    """Reset the in-memory override dicts. Called by integration tests."""
    _pin_quota_overrides.clear()
    _signature_variant_overrides.clear()
