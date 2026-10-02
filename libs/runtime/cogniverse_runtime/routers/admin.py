"""Admin endpoints - system administration and profile management."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator

from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
    ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KEY,
    ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KIND,
    canonicalize_entity_extraction_ground_truth_rows,
    serialize_entity_extraction_ground_truth_rows,
)
from cogniverse_agents.optimizer.golden_set_ground_truth import (
    GOLDEN_SET_GROUND_TRUTH_BLOB_KEY,
    GOLDEN_SET_GROUND_TRUTH_BLOB_KIND,
    canonicalize_golden_set_ground_truth_rows,
    serialize_golden_set_ground_truth_rows,
)
from cogniverse_agents.optimizer.profile_selection_ground_truth import (
    PROFILE_SELECTION_GROUND_TRUTH_BLOB_KEY,
    PROFILE_SELECTION_GROUND_TRUTH_BLOB_KIND,
    canonicalize_profile_selection_ground_truth_rows,
    serialize_profile_selection_ground_truth_rows,
)
from cogniverse_core.common.tenant_utils import (
    TenantDeletedError,
    canonical_tenant_id,
    raise_if_tenant_deleted,
)
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.validation.profile_validator import ProfileValidator
from cogniverse_foundation.caching.refreshing_cache import RefreshingCache
from cogniverse_foundation.config.manager import (
    BackendProfileExistsError,
    ConfigManager,
)
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_foundation.config.utils import get_config
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
from cogniverse_runtime.harness_keys import HarnessKeyStore
from cogniverse_runtime.http_errors import failure_response
from cogniverse_sdk.interfaces.config_store import (
    ConfigScope,
    ConfigStoreUnavailableError,
    ConfigWriteConflictError,
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
    _signature_variant_cache.invalidate(lambda key: True)


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
    _signature_variant_cache.invalidate(lambda key: True)


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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

            if not hasattr(backend_instance, "get_statistics"):
                raise HTTPException(
                    status_code=501,
                    detail=(f"Backend '{backend}' does not implement get_statistics()"),
                )
            # get_statistics is the sync Backend-interface method — off the
            # loop; it can call into the live cluster.
            backend_stats = await asyncio.to_thread(backend_instance.get_statistics)
            stats.update(backend_stats)

        return stats

    except HTTPException:
        raise
    except Exception as e:
        raise failure_response(
            500,
            "stats_unavailable",
            "System stats could not be read; the runtime log names the cause.",
            e,
            tenant_id=tenant_id,
            backend=backend,
        )


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
        HTTPException 409: Concurrent writes to the tenant's backend config
            outlasted every compare-and-set attempt
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

        def _validate_and_add() -> int:
            """Validate and store the profile; the backend config version the
            write produced. Config-store reads and a compare-and-set write
            that retries with backoff under contention: off the serving
            loop."""
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

            try:
                return config_manager.add_backend_profile(
                    profile,
                    tenant_id=request.tenant_id,
                    service="backend",
                    replace=False,
                ).version
            except BackendProfileExistsError as exc:
                # Another create of this name landed after the validation read.
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Profile validation failed",
                        "errors": [str(exc)],
                    },
                ) from exc

        version = await asyncio.to_thread(_validate_and_add)

        schema_deployed = False
        tenant_schema_name = None

        if request.deploy_schema:

            def _deploy() -> str:
                """Build the tenant's ingestion backend (Vespa connections on
                a cold cache) and deploy the schema, which blocks through
                prepareandactivate and convergence sleeps: off the loop."""
                backend = BackendRegistry.get_instance().get_ingestion_backend(
                    "vespa",
                    tenant_id=request.tenant_id,
                    config_manager=config_manager,
                    schema_loader=schema_loader,
                )
                backend.schema_registry.deploy_schema(
                    tenant_id=request.tenant_id,
                    base_schema_name=request.schema_name,
                )
                return backend.get_tenant_schema_name(
                    request.tenant_id, request.schema_name
                )

            tenant_schema_name = await asyncio.to_thread(_deploy)
            schema_deployed = True
            logger.info(
                f"Deployed schema '{tenant_schema_name}' for profile '{request.profile_name}'"
            )

        return ProfileCreateResponse(
            profile_name=request.profile_name,
            tenant_id=request.tenant_id,
            schema_deployed=schema_deployed,
            tenant_schema_name=tenant_schema_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            version=version,
        )

    except HTTPException:
        raise
    except ConfigWriteConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise failure_response(
            500,
            "profile_create_failed",
            f"Creating profile '{request.profile_name}' failed; the runtime log "
            "names the cause.",
            e,
            profile_name=request.profile_name,
            tenant_id=request.tenant_id,
        )


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

        backend_registry = BackendRegistry.get_instance()
        backend = backend_registry.get_ingestion_backend(
            "vespa",
            tenant_id=tenant_id,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        profile_summaries = []

        for profile_name, profile in profiles.items():
            schema_deployed = backend.schema_exists(
                schema_name=profile.schema_name, tenant_id=tenant_id
            )
            profile_summaries.append(
                ProfileSummary(
                    profile_name=profile_name,
                    type=profile.type,
                    description=profile.description,
                    schema_name=profile.schema_name,
                    embedding_model=profile.embedding_model,
                    schema_deployed=schema_deployed,
                    created_at=datetime.now(
                        timezone.utc
                    ).isoformat(),  # config store does not persist creation time
                )
            )

        return ProfileListResponse(
            profiles=profile_summaries,
            total_count=len(profile_summaries),
            tenant_id=tenant_id,
        )

    except Exception as e:
        raise failure_response(
            500,
            "profile_list_failed",
            f"Listing profiles for tenant '{tenant_id}' failed; the runtime log "
            "names the cause.",
            e,
            tenant_id=tenant_id,
        )


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

        backend_registry = BackendRegistry.get_instance()
        backend = backend_registry.get_ingestion_backend(
            "vespa",
            tenant_id=tenant_id,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        schema_deployed = backend.schema_exists(
            schema_name=profile.schema_name, tenant_id=tenant_id
        )
        tenant_schema_name = (
            backend.get_tenant_schema_name(tenant_id, profile.schema_name)
            if schema_deployed
            else None
        )

        from cogniverse_sdk.interfaces.config_store import ConfigScope

        config_entry = await asyncio.to_thread(
            config_manager.store.get_config,
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
            else datetime.now(timezone.utc).isoformat(),
            version=config_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise failure_response(
            500,
            "profile_read_failed",
            f"Reading profile '{profile_name}' failed; the runtime log names "
            "the cause.",
            e,
            profile_name=profile_name,
            tenant_id=tenant_id,
        )


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
        HTTPException 409: Concurrent writes to the tenant's backend config
            outlasted every compare-and-set attempt
        HTTPException 500: Update operation failed
    """

    def _update() -> tuple[List[str], int]:
        """The updated fields and the backend config version the update
        produced. Its config-store reads and compare-and-set write, which
        retries with backoff under contention, run off the serving loop."""
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

        written = config_manager.update_backend_profile(
            profile_name=profile_name,
            overrides=overrides,
            base_tenant_id=request.tenant_id,
            target_tenant_id=request.tenant_id,
            service="backend",
        )
        return updated_fields, written.version

    try:
        updated_fields, version = await asyncio.to_thread(_update)

        return ProfileUpdateResponse(
            profile_name=profile_name,
            tenant_id=request.tenant_id,
            updated_fields=updated_fields,
            version=version,
        )

    except HTTPException:
        raise
    except ConfigWriteConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise failure_response(
            500,
            "profile_update_failed",
            f"Updating profile '{profile_name}' failed; the runtime log names "
            "the cause.",
            e,
            profile_name=profile_name,
            tenant_id=request.tenant_id,
        )


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
        HTTPException 409: Cannot delete schema (other profiles using it), or
            concurrent writes to the tenant's backend config outlasted every
            compare-and-set attempt
        HTTPException 500: Deletion failed
    """

    def _delete() -> bool:
        """The profile delete's blocking work, run off the serving loop.

        Every step is synchronous Vespa traffic: the config reads, and — when
        the schema goes too — a full application redeploy that retries on 409
        with sleeps and a 300s read timeout. Inline, one profile delete freezes
        every request, stream and probe on this replica for its duration.
        """
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
        return schema_deleted

    try:
        schema_deleted = await asyncio.to_thread(_delete)

        return ProfileDeleteResponse(
            profile_name=profile_name,
            tenant_id=tenant_id,
            schema_deleted=schema_deleted,
            deleted_at=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except ConfigWriteConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise failure_response(
            500,
            "profile_delete_failed",
            f"Deleting profile '{profile_name}' failed; the runtime log names "
            "the cause.",
            e,
            profile_name=profile_name,
            tenant_id=tenant_id,
        )


def _catalog_profile(
    config_manager: ConfigManager, profile_name: str, tenant_id: str
) -> Optional[BackendProfileConfig]:
    """``profile_name`` from the tenant's merged catalog: the shipped profiles
    with the tenant's stored overrides on top, as ingestion resolves it."""
    profiles = get_config(tenant_id, config_manager).get("backend", {}).get("profiles")
    raw = (profiles or {}).get(profile_name)
    return None if raw is None else BackendProfileConfig.from_dict(profile_name, raw)


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
    The profile resolves from the tenant's stored profiles first, otherwise from
    the tenant's merged catalog of shipped profiles, which the list/get routes
    do not show.

    Args:
        profile_name: Profile name (path parameter)
        request: Deployment request
        config_manager: ConfigManager instance (injected)
        schema_loader: SchemaLoader instance (injected)

    Returns:
        Deployment status

    Raises:
        HTTPException 404: Profile not found
        HTTPException 410: The tenant has been deleted
        HTTPException 500: Deployment failed
    """

    def _resolve_target():
        """The profile, the tenant's ingestion backend and whether the schema
        is deployed: config store and Vespa reads, run off the serving loop."""
        profile = config_manager.get_backend_profile(
            profile_name=profile_name,
            tenant_id=request.tenant_id,
            service="backend",
        ) or _catalog_profile(config_manager, profile_name, request.tenant_id)
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found for tenant '{request.tenant_id}'",
            )
        raise_if_tenant_deleted(config_manager.store, request.tenant_id)
        backend = BackendRegistry.get_instance().get_ingestion_backend(
            "vespa",
            tenant_id=request.tenant_id,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        return (
            profile,
            backend,
            backend.schema_exists(
                schema_name=profile.schema_name, tenant_id=request.tenant_id
            ),
        )

    try:
        profile, backend, schema_exists = await asyncio.to_thread(_resolve_target)

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
                deployed_at=datetime.now(timezone.utc).isoformat(),
            )

        try:
            # Blocking deploy + convergence sleeps — off the loop.
            await asyncio.to_thread(
                backend.schema_registry.deploy_schema,
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
                deployed_at=datetime.now(timezone.utc).isoformat(),
            )

        except TenantDeletedError as e:
            raise _tenant_deleted(e, profile_name) from e
        except Exception as e:
            logger.error(f"Schema deployment failed: {e}")
            return SchemaDeploymentResponse(
                profile_name=profile_name,
                tenant_id=request.tenant_id,
                schema_name=profile.schema_name,
                tenant_schema_name="",
                deployment_status="failed",
                deployed_at=datetime.now(timezone.utc).isoformat(),
                error_message=str(e),
            )

    except HTTPException:
        raise
    except TenantDeletedError as e:
        raise _tenant_deleted(e, profile_name) from e
    except Exception as e:
        raise failure_response(
            500,
            "schema_deploy_failed",
            f"Deploying the schema of profile '{profile_name}' failed; the "
            "runtime log names the cause.",
            e,
            profile_name=profile_name,
            tenant_id=request.tenant_id,
        )


class SchemaRefusalInfo(BaseModel):
    error: str
    refused_at: str


class DriftedSchemaInfo(BaseModel):
    tenant_id: str
    base_schema_name: str
    schema_name: str
    refusal: Optional[SchemaRefusalInfo]


class SchemaDriftResponse(BaseModel):
    drifted: List[DriftedSchemaInfo]


@router.get("/schemas/drift", response_model=SchemaDriftResponse)
async def list_schema_drift(
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> SchemaDriftResponse:
    """Tenant schemas registered with a definition other than the shipped one.

    The startup migration redeploys each of them. One whose redeploy the
    backend refused carries that refusal; one without a refusal has not been
    migrated yet. 503 when the schema registry or the recorded refusals
    cannot be read.
    """
    from cogniverse_core.registries.exceptions import (
        RegistryStorageError,
        SchemaRegistryInitializationError,
    )
    from cogniverse_core.registries.schema_registry import drifted_schemas

    try:
        drifted = await asyncio.to_thread(
            drifted_schemas, config_manager, schema_loader
        )
    except (RegistryStorageError, SchemaRegistryInitializationError) as exc:
        raise failure_response(
            503,
            "schema_drift_unavailable",
            "The schema registry or the recorded migration refusals could not "
            "be read; retry.",
            exc,
        ) from exc
    return SchemaDriftResponse(
        drifted=[
            DriftedSchemaInfo(
                tenant_id=entry.tenant_id,
                base_schema_name=entry.base_schema_name,
                schema_name=entry.schema_name,
                refusal=None
                if entry.refusal is None
                else SchemaRefusalInfo(
                    error=entry.refusal.error, refused_at=entry.refusal.refused_at
                ),
            )
            for entry in drifted
        ]
    )


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
    expiry = (
        datetime.now(timezone.utc) + timedelta(hours=request.expires_in_hours)
    ).isoformat()

    from cogniverse_sdk.interfaces.config_store import ConfigScope

    await asyncio.to_thread(
        config_manager.set_config_value,
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


_MESSAGING_GATEWAY_AGENT = "_messaging_gateway"

_system_memory_factory = None


def set_system_memory_factory(factory) -> None:
    """Test seam: override how the SYSTEM-partition Mem0 manager is built."""
    global _system_memory_factory
    _system_memory_factory = factory


def _system_memory_manager():
    """Mem0 manager for the SYSTEM partition (user-tenant mappings)."""
    if _system_memory_factory is not None:
        return _system_memory_factory()
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.memory.manager import Mem0MemoryManager

    mgr = Mem0MemoryManager(SYSTEM_TENANT_ID)
    if not mgr.memory:
        from cogniverse_runtime.memory_init import lazy_init_memory
        from cogniverse_runtime.routers.tenant import _require_config_manager

        lazy_init_memory(
            mgr, SYSTEM_TENANT_ID, _require_config_manager(), auto_create_schema=False
        )
    return mgr


class RegisterRequest(BaseModel):
    platform: str
    external_user_id: str
    token: str


@router.post("/messaging/register")
async def register_messaging_user(
    request: RegisterRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Dict[str, Any]:
    """Claim an invite token for this user, store the user-tenant mapping,
    consume the token — in that order.

    The claim is a compare-and-set on the token's config record, so of any
    number of concurrent registers on any process or replica exactly one
    user gets the token; every other user gets 404 from then on. 404 =
    invalid/expired/used token, or one another user claimed. 503 = config
    store or Mem0 outage; the token stays claimed for this user, who can
    retry.
    """
    from cogniverse_core.messaging_auth import InviteTokenManager, UserTenantMapper

    token_manager = InviteTokenManager(config_manager)
    try:
        tenant_id = await asyncio.to_thread(
            token_manager.claim_token,
            request.token,
            request.platform,
            request.external_user_id,
        )
    except Exception as exc:
        raise failure_response(
            503,
            "registration_unavailable",
            "The invite token store did not answer; retry the registration.",
            exc,
        ) from exc
    if not tenant_id:
        raise HTTPException(status_code=404, detail="invalid_token")

    try:
        mapper = UserTenantMapper(_system_memory_manager())
    except Exception as exc:
        raise failure_response(
            503,
            "registration_unavailable",
            "The user mapping store did not answer; the invite token is held "
            "for this user, retry the registration.",
            exc,
        ) from exc
    registered = await asyncio.to_thread(
        mapper.register_user,
        request.platform,
        request.external_user_id,
        tenant_id,
    )
    if not registered:
        raise HTTPException(
            status_code=503,
            detail=(
                "registration unavailable: mapping store failed; "
                "token held for this user"
            ),
        )

    consumed = await asyncio.to_thread(
        token_manager.mark_token_used,
        request.token,
        request.platform,
        request.external_user_id,
    )
    if not consumed:
        logger.error(
            "User %s registered but token %s... not consumed; "
            "it stays claimed by that user",
            request.external_user_id,
            request.token[:8],
        )
    return {"tenant_id": tenant_id}


@router.get("/messaging/resolve")
async def resolve_messaging_user(
    platform: str = Query(...),
    external_user_id: str = Query(...),
) -> Dict[str, Any]:
    """Resolve a messaging user to their tenant.

    ``{"tenant_id": null}`` = genuinely unregistered. A Mem0 outage is 503,
    never null — the gateway must not read an outage as "please register".
    """
    from cogniverse_core.messaging_auth import UserTenantMapper

    try:
        mapper = UserTenantMapper(_system_memory_manager())
        tenant_id = await asyncio.to_thread(
            mapper.get_tenant_id, platform, external_user_id
        )
    except Exception as exc:
        raise failure_response(
            503,
            "resolve_unavailable",
            "The user mapping store did not answer; retry the lookup.",
            exc,
            platform=platform,
        ) from exc
    return {"tenant_id": tenant_id}


class SendMessageRequest(BaseModel):
    tenant_id: str
    message: str


def _resolve_telegram_chat_ids(tenant_id: str) -> List[str]:
    """Return the telegram chat ids linked to ``tenant_id``.

    Reverses the user↔tenant mapping the gateway wrote into the SYSTEM mem0
    partition (agent_name ``_messaging_gateway``). ``get_all_memories`` raises
    on a backend outage, so an outage propagates (the route turns it into 503)
    rather than being read as "no linked chats". A never-initialised store (the
    gateway never ran) returns [] — a genuine no-mappings state.
    """
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.memory.manager import Mem0MemoryManager

    target = canonical_tenant_id(tenant_id)
    mgr = Mem0MemoryManager(SYSTEM_TENANT_ID)
    if not mgr.memory:
        from cogniverse_runtime.memory_init import lazy_init_memory
        from cogniverse_runtime.routers.tenant import _require_config_manager

        lazy_init_memory(
            mgr, SYSTEM_TENANT_ID, _require_config_manager(), auto_create_schema=False
        )
    if not mgr.memory:
        return []

    # Walk every page (limit=None): the target tenant's mappings can sit
    # past the store's 100-row page when other tenants share the SYSTEM
    # partition, and a capped read would silently drop their linked chats.
    rows = mgr.get_all_memories(
        tenant_id=SYSTEM_TENANT_ID,
        agent_name=_MESSAGING_GATEWAY_AGENT,
        limit=None,
    )
    chat_ids: List[str] = []
    seen: set = set()
    for row in rows:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                continue
        if (
            meta.get("type") == "user_mapping"
            and meta.get("platform") == "telegram"
            and canonical_tenant_id(str(meta.get("tenant_id") or "")) == target
        ):
            chat_id = str(meta.get("external_user_id") or "")
            if chat_id and chat_id not in seen:
                seen.add(chat_id)
                chat_ids.append(chat_id)
    return chat_ids


async def _resolve_outbound_queue():
    """Pick the in-pod or Redis-backed outbound queue from config.

    Non-empty ``SystemConfig.redis_url`` → cross-pod Redis backend so the
    gateway drains messages the runtime enqueued on any pod. Empty (or config
    unwired, e.g. a router-only test harness) → the in-pod singleton. Both
    expose ``enqueue``/``drain``. Env reads for ``REDIS_URL`` happen at the
    runtime startup boundary (main.py); this never touches env directly.
    """
    from cogniverse_runtime.messaging import get_outbound_queue

    redis_url = ""
    if _config_manager is not None:
        redis_url = _config_manager.get_system_config().redis_url
    if redis_url:
        from cogniverse_runtime.messaging_redis import get_redis_outbound_queue

        return await get_redis_outbound_queue(redis_url)
    return get_outbound_queue()


@router.post("/messaging/send")
async def send_message(request: SendMessageRequest) -> Dict[str, int]:
    """Enqueue a message for delivery to the tenant's linked messaging chats.

    Resolves the tenant's linked chats and enqueues one message per chat for
    the gateway to drain and deliver. Returns the number enqueued (0 when the
    tenant has no linked chats). A backend outage while resolving surfaces as
    503 so callers can retry — it is never read as "no linked chats".
    """
    from cogniverse_runtime.messaging import OutboundMessage

    try:
        chat_ids = await asyncio.to_thread(
            _resolve_telegram_chat_ids, request.tenant_id
        )
    except Exception as exc:  # noqa: BLE001 — surface, never mask as enqueued: 0
        logger.error("Could not resolve linked chats for send: %r", exc)
        raise HTTPException(
            status_code=503,
            detail={"message": "could not resolve linked chats; retry"},
        )

    if not chat_ids:
        return {"enqueued": 0}

    queue = await _resolve_outbound_queue()
    now = datetime.now(timezone.utc).isoformat()
    for chat_id in chat_ids:
        await queue.enqueue(
            OutboundMessage(
                tenant_id=request.tenant_id,
                chat_id=chat_id,
                text=request.message,
                created_at=now,
            )
        )
    return {"enqueued": len(chat_ids)}


@router.get("/messaging/outbound/drain")
async def drain_outbound_messages() -> Dict[str, Any]:
    """Return and clear the pending outbound messages (the gateway polls this)."""
    queue = await _resolve_outbound_queue()
    batch = await queue.drain()
    return {
        "messages": [
            {
                "tenant_id": m.tenant_id,
                "chat_id": m.chat_id,
                "text": m.text,
                "platform": m.platform,
                "created_at": m.created_at,
            }
            for m in batch
        ]
    }


_ADMIN_TYPE_TO_NAMESPACE: Dict[str, str] = {
    "preference": "_user_memories",
    "strategy": "_strategy_store",
}

_ADMIN_ALL_NAMESPACES = ["_user_memories", "_strategy_store"]


@router.delete("/memories/{tenant_id}/{memory_id}")
async def admin_delete_memory(tenant_id: str, memory_id: str):
    """Admin: delete any memory by ID, regardless of namespace."""
    from cogniverse_core.memory.manager import Mem0MemoryManager

    tenant_id = canonical_tenant_id(tenant_id)

    def _delete() -> bool:
        mgr = Mem0MemoryManager(tenant_id)
        if not mgr.memory:
            raise HTTPException(
                status_code=503, detail="Memory backend not initialised"
            )
        # delete_memory addresses the memory by id alone; the namespace
        # argument is signature compatibility.
        return mgr.delete_memory(
            memory_id=memory_id,
            tenant_id=tenant_id,
            agent_name="_user_memories",
        )

    if await asyncio.to_thread(_delete):
        logger.info("Admin deleted memory %s for tenant %s", memory_id, tenant_id)
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

    tenant_id = canonical_tenant_id(tenant_id)

    if type and type != "all":
        ns = _ADMIN_TYPE_TO_NAMESPACE.get(type)
        if ns is None:
            raise HTTPException(status_code=400, detail=f"Unknown memory type: {type}")
        namespaces = [ns]
        cleared_type = type
    else:
        namespaces = list(_ADMIN_ALL_NAMESPACES)
        cleared_type = "all"

    def _clear() -> None:
        mgr = Mem0MemoryManager(tenant_id)
        if not mgr.memory:
            raise HTTPException(
                status_code=503, detail="Memory backend not initialised"
            )
        for namespace in namespaces:
            mgr.clear_agent_memory(tenant_id=tenant_id, agent_name=namespace)

    await asyncio.to_thread(_clear)
    logger.info("Admin cleared '%s' memories for tenant %s", cleared_type, tenant_id)
    return {"status": "cleared", "type": cleared_type}


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

    tenant_id = canonical_tenant_id(tenant_id)
    registry = build_default_registry()

    def _drop() -> Dict[str, int]:
        mgr = Mem0MemoryManager(tenant_id)
        if not mgr.memory:
            raise HTTPException(
                status_code=503, detail="Memory backend not initialised"
            )
        return mgr.drop_session(session_id, registry)

    deleted_by_kind = await asyncio.to_thread(_drop)
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


# Delivers a session close to every runtime worker process; wired at startup.
_cluster_events = None

# How long a session close waits for every worker to sweep its warm tenants.
SESSION_CLOSE_ACK_TIMEOUT_S = 60.0


def set_cluster_events(cluster_events) -> None:
    """Wire the channel a session close reaches every worker process through."""
    global _cluster_events
    _cluster_events = cluster_events


def sweep_closed_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop a closed session's memories from this process's warm tenants.

    The ``session_closed`` cluster-event handler, run on every worker.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.memory.schema import build_default_registry

    session_id = payload["session_id"]
    registry = build_default_registry()
    per_tenant: Dict[str, Dict[str, int]] = {}
    skipped: List[str] = []
    for mgr in list(Mem0MemoryManager._instances.values()):
        tenant_id = getattr(mgr, "tenant_id", None) or "unknown"
        if not getattr(mgr, "memory", None):
            skipped.append(tenant_id)
            continue
        try:
            deleted = mgr.drop_session(session_id, registry)
        except Exception as exc:
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
    return {"per_tenant": per_tenant, "skipped_tenants": skipped}


@router.post("/sessions/{session_id}/close")
async def admin_close_session(session_id: str):
    """Fan-out session close: drop the session across every warm tenant of
    every runtime worker process.

    A user session can write EPHEMERAL_SESSION memories under any tenant
    the request touched, on whichever worker served it. On session close
    (logout, ws-disconnect, idle timeout) the gateway POSTs here once; the
    close reaches every worker, and each sweeps its warm
    ``Mem0MemoryManager`` instances calling ``drop_session(session_id)``.
    The answer waits for every worker's sweep and sums them; a worker that
    does not confirm within ``SESSION_CLOSE_ACK_TIMEOUT_S``, or a channel
    that is down, answers 503 and the close can be retried.

    Tenants that are warm on no worker are skipped — their next access will
    deserialise from Vespa and the EPHEMERAL_SESSION rows still exist there.
    The next request that warms the manager and triggers a session-close
    webhook will sweep them. Operators who need a guaranteed sweep can call
    the per-tenant DELETE endpoint with the known tenant id.
    """
    from cogniverse_runtime.cluster_events import ClusterEventError

    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must be non-empty")
    if _cluster_events is None:
        raise HTTPException(
            status_code=503, detail="session close: cluster events are not wired"
        )
    try:
        swept = await _cluster_events.publish(
            "session_closed",
            {"session_id": session_id},
            timeout_s=SESSION_CLOSE_ACK_TIMEOUT_S,
        )
    except ClusterEventError as exc:
        raise failure_response(
            503,
            "session_close_incomplete",
            f"Session {session_id} was not closed on every runtime worker; retry "
            "the close.",
            exc,
            session_id=session_id,
        ) from exc

    per_tenant: Dict[str, Dict[str, int]] = {}
    skipped: set = set()
    for result in swept.values():
        for tenant_id, deleted in result["per_tenant"].items():
            merged = per_tenant.setdefault(tenant_id, {})
            for kind, count in deleted.items():
                merged[kind] = merged.get(kind, 0) + count
        skipped.update(result["skipped_tenants"])
    total = sum(sum(deleted.values()) for deleted in per_tenant.values())

    logger.info(
        "Admin close_session(%s) swept %d workers, deleted %d memories in %d tenants",
        session_id,
        len(swept),
        total,
        len(per_tenant),
    )
    return {
        "status": "closed",
        "session_id": session_id,
        "per_tenant": per_tenant,
        "total_deleted": total,
        "skipped_tenants": sorted(skipped),
        "workers": sorted(swept),
    }


# ---------------------------------------------------------------------------
# Operability admin endpoints (pin quota / variant select / canary)
#
# Pin quotas and variant selections are per-tenant config-store records.
# Canary actions go straight to ArtifactManager which persists them to Phoenix.
# ---------------------------------------------------------------------------


class PinQuotasUpdateRequest(BaseModel):
    user: Optional[int] = None
    tenant_admin: Optional[int] = None
    org_admin: Optional[int] = None


class PinQuotasResponse(BaseModel):
    tenant_id: str
    quotas: Dict[str, int]


class ProfileSelectionGroundTruthResponse(BaseModel):
    tenant_id: str
    row_count: int
    version: int
    active: Dict[str, Any]


class GoldenSetGroundTruthResponse(BaseModel):
    tenant_id: str
    row_count: int
    version: int
    active: Dict[str, Any]


class EntityExtractionGroundTruthResponse(BaseModel):
    tenant_id: str
    row_count: int
    version: int
    active: Dict[str, Any]


def _default_pin_quotas() -> Dict[str, int]:
    from cogniverse_core.memory.pinning import PinQuotas

    d = PinQuotas()
    return {
        "user": d.user,
        "tenant_admin": d.tenant_admin,
        "org_admin": -1 if d.org_admin is None else d.org_admin,
    }


# Pin quotas and signature-variant selections are per-tenant config records.
# Every PUT is a compare-and-set read-modify-write of the record, so concurrent
# PUTs on any process or replica each keep the fields they changed, and a PUT
# is in the store before it is answered.
_ADMIN_OVERRIDES_SERVICE = "admin_overrides"
_PIN_QUOTA_KEY = "pin_quotas"
_SIGNATURE_VARIANT_KEY = "signature_variants"

# The dispatcher resolves a tenant's variant selection on every request, so
# each process serves it from memory: re-read off the request path once it is
# SIGNATURE_VARIANT_REFRESH_S old, never served once it is
# SIGNATURE_VARIANT_MAX_STALENESS_S old. A PUT on this process is served at
# once; one on another process or replica within those bounds.
SIGNATURE_VARIANT_REFRESH_S = 5.0
SIGNATURE_VARIANT_MAX_STALENESS_S = 30.0
_signature_variant_cache: RefreshingCache[str, Dict[str, str]] = RefreshingCache(
    name="signature-variants",
    refresh_after_s=SIGNATURE_VARIANT_REFRESH_S,
    max_staleness_s=SIGNATURE_VARIANT_MAX_STALENESS_S,
    max_entries=1024,
)


def _admin_config_store():
    if _config_manager is None:
        raise RuntimeError(
            "admin config store is not wired: set_config_manager was never called"
        )
    return _config_manager.store


def _stored_override(tenant_key: str, config_key: str) -> Optional[Dict[str, Any]]:
    """A tenant's override record as the store holds it now, or None."""
    entry = _admin_config_store().get_config(
        tenant_key, ConfigScope.SYSTEM, _ADMIN_OVERRIDES_SERVICE, config_key
    )
    return None if entry is None else dict(entry.config_value)


def _update_override(
    tenant_key: str,
    config_key: str,
    change: Callable[[Dict[str, Any]], None],
    empty: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply ``change`` to a tenant's override record with compare-and-set.

    ``change`` edits the record as stored (``empty()`` when there is none) in
    place, and runs again on the newer record whenever a concurrent PUT
    lands first. Returns the record as stored afterwards; an unchanged record
    is not rewritten.
    """

    def update(entry):
        current = dict(entry.config_value) if entry is not None else empty()
        changed = dict(current)
        change(changed)
        return None if entry is not None and changed == current else changed

    entry = _admin_config_store().update_config(
        tenant_key, ConfigScope.SYSTEM, _ADMIN_OVERRIDES_SERVICE, config_key, update
    )
    return dict(entry.config_value)


async def _load_pin_quotas(tenant_id: str) -> Dict[str, int]:
    """Return a tenant's stored pin quotas, or the defaults if none are stored.

    Reads the store on every call, so enforcement on every process and
    replica applies the last answered PUT. A store outage propagates rather
    than masquerading as "unset" — an admin must not silently see defaults
    when the real values are merely unreachable.
    """
    stored = await asyncio.to_thread(
        _stored_override, canonical_tenant_id(tenant_id), _PIN_QUOTA_KEY
    )
    return _default_pin_quotas() if stored is None else stored


@router.get("/tenants/{tenant_id}/pin_quotas", response_model=PinQuotasResponse)
async def get_pin_quotas(tenant_id: str) -> PinQuotasResponse:
    """return effective pin quotas for a tenant."""
    try:
        quotas = await _load_pin_quotas(tenant_id)
    except Exception as exc:
        # The store read raises on an outage (never masquerades as "unset");
        # map it to 503 rather than an opaque 500.
        raise _store_unavailable("pin-quota", exc, tenant_id) from exc
    return PinQuotasResponse(tenant_id=tenant_id, quotas=quotas)


@router.put("/tenants/{tenant_id}/pin_quotas", response_model=PinQuotasResponse)
async def set_pin_quotas(
    tenant_id: str, body: PinQuotasUpdateRequest
) -> PinQuotasResponse:
    """set per-role pin quotas for a tenant.

    Only non-None fields are updated, merged onto the stored record. Negative
    values (other than org_admin's unlimited sentinel of -1) are rejected.
    The result is in the store before the response is sent.
    """
    # Validate before touching the store so a bad request never depends on it.
    if body.user is not None and body.user < 0:
        raise HTTPException(400, "user quota must be >= 0")
    if body.tenant_admin is not None and body.tenant_admin < 0:
        raise HTTPException(400, "tenant_admin quota must be >= 0")
    if body.org_admin is not None and body.org_admin < -1:
        # -1 is the unlimited sentinel; any other negative persists as a
        # literal limit that used >= comparisons always exceed, silently
        # rejecting every org_admin pin for the tenant.
        raise HTTPException(400, "org_admin quota must be >= 0, or -1 for unlimited")

    key = canonical_tenant_id(tenant_id)
    requested = {
        role: value
        for role, value in (
            ("user", body.user),
            ("tenant_admin", body.tenant_admin),
            ("org_admin", body.org_admin),
        )
        if value is not None
    }
    try:
        quotas = await asyncio.to_thread(
            _update_override,
            key,
            _PIN_QUOTA_KEY,
            lambda stored: stored.update(requested),
            _default_pin_quotas,
        )
    except ConfigWriteConflictError as exc:
        raise HTTPException(409, f"pin-quota update conflicted: {exc}") from exc
    except Exception as exc:
        raise _store_unavailable("pin-quota", exc, tenant_id) from exc
    logger.info("Updated pin quotas for tenant=%s: %s", key, quotas)
    return PinQuotasResponse(tenant_id=tenant_id, quotas=quotas)


@router.put(
    "/tenants/{tenant_id}/profile_selection_ground_truth",
    response_model=ProfileSelectionGroundTruthResponse,
)
async def set_profile_selection_ground_truth(
    tenant_id: str, rows: List[Dict[str, Any]]
) -> ProfileSelectionGroundTruthResponse:
    """Persist tenant-owned profile-selection ground truth as the active blob."""

    try:
        canonical_rows = canonicalize_profile_selection_ground_truth_rows(rows)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    key = canonical_tenant_id(tenant_id)
    content = serialize_profile_selection_ground_truth_rows(canonical_rows)
    try:
        am = _build_artifact_manager(key)
        _, version = await am.save_blob_versioned(
            PROFILE_SELECTION_GROUND_TRUTH_BLOB_KIND,
            PROFILE_SELECTION_GROUND_TRUTH_BLOB_KEY,
            content,
            consumed_example_ids=["admin_upload:profile_selection_ground_truth"],
            decision="promote",
            scored=False,
            score=None,
            base_score=None,
            candidate_score=None,
        )
        state = await am.activate_version_guarded(
            PROFILE_SELECTION_GROUND_TRUTH_BLOB_KIND,
            PROFILE_SELECTION_GROUND_TRUTH_BLOB_KEY,
            version,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _store_unavailable(
            "profile_selection_ground_truth", exc, tenant_id
        ) from exc

    logger.info(
        "Updated + persisted profile_selection_ground_truth for tenant=%s with %d rows",
        key,
        len(canonical_rows),
    )
    return ProfileSelectionGroundTruthResponse(
        tenant_id=key,
        row_count=len(canonical_rows),
        version=version,
        active=state["active"],
    )


@router.put(
    "/tenants/{tenant_id}/golden_set_ground_truth",
    response_model=GoldenSetGroundTruthResponse,
)
async def set_golden_set_ground_truth(
    tenant_id: str, rows: List[Dict[str, Any]]
) -> GoldenSetGroundTruthResponse:
    """Persist tenant-owned golden-set ground truth as the active blob."""

    try:
        canonical_rows = canonicalize_golden_set_ground_truth_rows(rows)
    except ValueError as exc:
        message = str(exc).replace(
            "profile_selection_ground_truth", "golden_set_ground_truth"
        )
        raise HTTPException(400, message) from exc

    key = canonical_tenant_id(tenant_id)
    content = serialize_golden_set_ground_truth_rows(canonical_rows)
    try:
        am = _build_artifact_manager(key)
        _, version = await am.save_blob_versioned(
            GOLDEN_SET_GROUND_TRUTH_BLOB_KIND,
            GOLDEN_SET_GROUND_TRUTH_BLOB_KEY,
            content,
            consumed_example_ids=["admin_upload:golden_set_ground_truth"],
            decision="promote",
            scored=False,
            score=None,
            base_score=None,
            candidate_score=None,
        )
        state = await am.activate_version_guarded(
            GOLDEN_SET_GROUND_TRUTH_BLOB_KIND,
            GOLDEN_SET_GROUND_TRUTH_BLOB_KEY,
            version,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _store_unavailable("golden_set_ground_truth", exc, tenant_id) from exc

    logger.info(
        "Updated + persisted golden_set_ground_truth for tenant=%s with %d rows",
        key,
        len(canonical_rows),
    )
    return GoldenSetGroundTruthResponse(
        tenant_id=key,
        row_count=len(canonical_rows),
        version=version,
        active=state["active"],
    )


@router.put(
    "/tenants/{tenant_id}/entity_extraction_ground_truth",
    response_model=EntityExtractionGroundTruthResponse,
)
async def set_entity_extraction_ground_truth(
    tenant_id: str, rows: List[Dict[str, Any]]
) -> EntityExtractionGroundTruthResponse:
    """Persist tenant-owned entity-extraction ground truth as the active blob."""

    try:
        canonical_rows = canonicalize_entity_extraction_ground_truth_rows(rows)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    key = canonical_tenant_id(tenant_id)
    content = serialize_entity_extraction_ground_truth_rows(canonical_rows)
    try:
        am = _build_artifact_manager(key)
        _, version = await am.save_blob_versioned(
            ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KIND,
            ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KEY,
            content,
            consumed_example_ids=["admin_upload:entity_extraction_ground_truth"],
            decision="promote",
            scored=False,
            score=None,
            base_score=None,
            candidate_score=None,
        )
        state = await am.activate_version_guarded(
            ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KIND,
            ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KEY,
            version,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _store_unavailable(
            "entity_extraction_ground_truth", exc, tenant_id
        ) from exc

    logger.info(
        "Updated + persisted entity_extraction_ground_truth for tenant=%s with %d rows",
        key,
        len(canonical_rows),
    )
    return EntityExtractionGroundTruthResponse(
        tenant_id=key,
        row_count=len(canonical_rows),
        version=version,
        active=state["active"],
    )


# Memory pin / unpin / list endpoints. Pinned memories survive lifecycle
# cleanup via PinService (wired into the scheduler). The requester's role +
# actor_id ride in the request body.


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


async def _pin_service_for(tenant_id: str):
    """Build the tenant's PinService with quotas read from the store now.

    Enforcement resolves quotas through PinQuotas.for_tenant from the stored
    record read here, so every process and replica enforces the last
    answered PUT.
    """
    admin_overrides = await _load_pin_quotas(tenant_id)
    return await asyncio.to_thread(_get_pin_service, tenant_id, admin_overrides)


def _get_pin_service(tenant_id: str, admin_overrides: Mapping[str, int] | None):
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
        from cogniverse_runtime.memory_init import lazy_init_memory
        from cogniverse_runtime.routers.tenant import _require_config_manager

        # Pin/promote/restore operate on memories that already exist, so
        # the schema is already deployed. Don't auto-create here: a deploy
        # triggers a Vespa redeploy that can drop rows another process
        # just fed mid-operation (the source memory then reads as 404).
        try:
            lazy_init_memory(
                mgr, tenant_id, _require_config_manager(), auto_create_schema=False
            )
        except Exception as exc:
            raise failure_response(
                503,
                "memory_unavailable",
                f"Memory backend not initialised for tenant {tenant_id}.",
                exc,
                tenant_id=tenant_id,
            ) from exc
    if not mgr.memory:
        raise HTTPException(
            status_code=503,
            detail=f"Memory backend not initialised for tenant {tenant_id}",
        )
    return PinService(
        mgr,
        build_default_registry(),
        quotas=PinQuotas.for_tenant(tenant_id, admin_overrides=admin_overrides),
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
    tenant_id = canonical_tenant_id(tenant_id)
    svc = await _pin_service_for(tenant_id)
    try:
        record = await asyncio.to_thread(
            svc.pin,
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
    tenant_id = canonical_tenant_id(tenant_id)
    svc = await _pin_service_for(tenant_id)
    try:
        removed = await asyncio.to_thread(
            svc.unpin,
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
    tenant_id = canonical_tenant_id(tenant_id)
    svc = await _pin_service_for(tenant_id)
    records = await asyncio.to_thread(svc.list_pins, tenant_id)
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
    from cogniverse_core.memory.schema import build_promotable_registry

    if not body.actor_id.strip():
        raise HTTPException(400, "actor_id must be non-empty")
    try:
        actor_role = Pinnable(body.actor_role)
    except ValueError as exc:
        valid = ", ".join(p.value for p in Pinnable)
        raise HTTPException(
            400, f"invalid actor_role {body.actor_role!r}; expected one of: {valid}"
        ) from exc

    tenant_id = canonical_tenant_id(tenant_id)
    source_mm = (await _pin_service_for(tenant_id))._mm  # reuse the lazy-init path
    # Document point-GET: read-your-writes. The search-backed get_all lags
    # index visibility, so a freshly written memory would 404 here.
    try:
        src = await asyncio.to_thread(source_mm.memory.get, memory_id)
    except Exception as exc:
        raise failure_response(
            503,
            "memory_unavailable",
            f"Memory {memory_id} could not be read; retry.",
            exc,
            tenant_id=tenant_id,
            memory_id=memory_id,
        ) from exc
    row_tenant = (src or {}).get("user_id")
    if src is None or (row_tenant is not None and row_tenant != tenant_id):
        raise HTTPException(404, f"memory {memory_id} not found in tenant {tenant_id}")

    svc = FederationService(
        memory_manager_factory=lambda tid: Mem0MemoryManager(tid),
        registry=build_promotable_registry(),
    )
    try:
        result = await asyncio.to_thread(
            svc.promote_to_org_trunk,
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

    tenant_id = canonical_tenant_id(tenant_id)
    source_mm = (await _pin_service_for(tenant_id))._mm  # reuse the lazy-init path
    # Document point-GET: read-your-writes. The search-backed get_all lags
    # index visibility, so a freshly written memory would 404 here.
    try:
        src = await asyncio.to_thread(source_mm.memory.get, memory_id)
    except Exception as exc:
        raise failure_response(
            503,
            "memory_unavailable",
            f"Memory {memory_id} could not be read; retry.",
            exc,
            tenant_id=tenant_id,
            memory_id=memory_id,
        ) from exc
    row_tenant = (src or {}).get("user_id")
    if src is None or (row_tenant is not None and row_tenant != tenant_id):
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
        await asyncio.to_thread(
            source_mm.memory.update,
            memory_id=memory_id,
            data=src.get("memory") or src.get("text") or "",
            metadata=new_metadata,
        )
    except Exception as exc:
        raise failure_response(
            503,
            "memory_unavailable",
            f"The trust update of memory {memory_id} failed; retry.",
            exc,
            tenant_id=tenant_id,
            memory_id=memory_id,
        ) from exc

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
    tenant_id = canonical_tenant_id(tenant_id)
    source_mm = (await _pin_service_for(tenant_id))._mm  # reuse the lazy-init path
    ok = await asyncio.to_thread(source_mm.restore_archived_memory, memory_id)
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


async def load_signature_variants(tenant_id: str) -> Dict[str, str]:
    """Return a tenant's stored signature-variant selections, read now.

    A store outage propagates rather than masquerading as "no selection".
    """
    stored = await asyncio.to_thread(
        _stored_override, canonical_tenant_id(tenant_id), _SIGNATURE_VARIANT_KEY
    )
    return {} if stored is None else stored


async def cached_signature_variants(tenant_id: str) -> Dict[str, str]:
    """A tenant's selections as this process serves them to the dispatcher.

    Served from memory within ``SIGNATURE_VARIANT_REFRESH_S`` /
    ``SIGNATURE_VARIANT_MAX_STALENESS_S``; a read the store cannot answer
    raises once nothing younger than the staleness bound is held.
    """
    key = canonical_tenant_id(tenant_id)
    selections = await asyncio.to_thread(
        _signature_variant_cache.get,
        key,
        lambda: _stored_override(key, _SIGNATURE_VARIANT_KEY) or {},
    )
    return dict(selections)


@router.get(
    "/tenants/{tenant_id}/signature_variants",
    response_model=SignatureVariantResponse,
)
async def get_signature_variants(tenant_id: str) -> SignatureVariantResponse:
    """list per-agent variant selections for a tenant."""
    try:
        selections = await load_signature_variants(tenant_id)
    except Exception as exc:
        # The store read raises on an outage (never masquerades as "no
        # selection"); map it to 503 rather than an opaque 500.
        raise _store_unavailable("signature-variant", exc, tenant_id) from exc
    return SignatureVariantResponse(tenant_id=tenant_id, selections=selections)


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
    # Store under the canonical key so the dispatcher finds it whether the
    # tenant arrives as simple or colon form.
    key = canonical_tenant_id(tenant_id)
    try:
        selections = await asyncio.to_thread(
            _update_override,
            key,
            _SIGNATURE_VARIANT_KEY,
            lambda stored: stored.update({agent_type: body.variant_id}),
            dict,
        )
    except ConfigWriteConflictError as exc:
        raise HTTPException(409, f"signature-variant update conflicted: {exc}") from exc
    except Exception as exc:
        raise _store_unavailable("signature-variant", exc, tenant_id) from exc
    _signature_variant_cache.invalidate(lambda cached: cached == key)
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


# Phoenix endpoints the admin artifact actions build ArtifactManagers against,
# wired once at startup from the entrypoint (set_phoenix_endpoints) so this
# router reads no environment at request time. Empty until wired; the pair is
# replaced as one object so a reader never sees one endpoint from each wiring.
_phoenix_endpoints: Dict[str, str] = {}


def set_phoenix_endpoints(http_endpoint: str, grpc_endpoint: str) -> None:
    """Wire the Phoenix endpoints used by the admin artifact actions.

    Called at startup from the entrypoint with the deployment's
    ``telemetry_url`` and ``telemetry_collector_endpoint``.
    """
    missing = [
        name
        for name, value in (
            ("http_endpoint", http_endpoint),
            ("grpc_endpoint", grpc_endpoint),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"admin Phoenix wiring needs {' and '.join(missing)}; got "
            f"http_endpoint={http_endpoint!r}, grpc_endpoint={grpc_endpoint!r}"
        )
    global _phoenix_endpoints
    _phoenix_endpoints = {
        "http_endpoint": http_endpoint,
        "grpc_endpoint": grpc_endpoint,
    }


def _build_artifact_manager(tenant_id: str):
    """Construct an ArtifactManager for an admin artifact action, targeting the
    Phoenix endpoints wired at startup (see ``set_phoenix_endpoints``)."""
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    endpoints = _phoenix_endpoints
    if not endpoints:
        raise RuntimeError(
            f"admin artifact store for tenant {tenant_id!r} has no Phoenix "
            "endpoints: set_phoenix_endpoints was never called"
        )
    provider = PhoenixProvider()
    provider.initialize({"tenant_id": tenant_id, **endpoints})
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
    """Drop every signature-variant selection this process holds."""
    _signature_variant_cache.invalidate(lambda key: True)


class HarnessKeyCreateRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)

    @field_validator("tenant_id", "name")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant(cls, value: str) -> str:
        from cogniverse_core.common.tenant_utils import validate_tenant_id

        validate_tenant_id(value)
        return value


def _tenant_deleted(exc: TenantDeletedError, profile_name: str) -> HTTPException:
    """410 for a schema deploy of a tenant marked deleted."""
    return failure_response(
        410,
        "tenant_deleted",
        f"Tenant '{exc.tenant_id}' has been deleted; its schemas and memories "
        "are not written until the tenant is created again.",
        exc,
        tenant_id=exc.tenant_id,
        profile_name=profile_name,
    )


def _store_unavailable(store: str, exc: Exception, tenant_id: str):
    """503 for a tenant artifact store that did not answer."""
    return failure_response(
        503,
        "store_unavailable",
        f"The {store} store did not answer; retry.",
        exc,
        store=store,
        tenant_id=tenant_id,
    )


def _harness_key_store_unavailable(exc: ConfigStoreUnavailableError):
    return failure_response(
        503,
        "harness_key_store_unavailable",
        "The harness key store did not answer; retry.",
        exc,
    )


@router.post("/harness/keys")
async def create_harness_key(
    request: HarnessKeyCreateRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
):
    tenant_id = canonical_tenant_id(request.tenant_id)
    try:
        return await asyncio.to_thread(
            HarnessKeyStore(config_manager.store).create,
            tenant_id,
            request.name,
        )
    except ConfigStoreUnavailableError as exc:
        raise _harness_key_store_unavailable(exc) from exc


@router.get("/harness/keys")
async def list_harness_keys(
    tenant_id: str | None = Query(None, min_length=1),
    page_size: int = Query(100, ge=1, le=1000),
    continuation: str | None = Query(None, max_length=8192),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
):
    try:
        if tenant_id is not None:
            from cogniverse_core.common.tenant_utils import validate_tenant_id

            validate_tenant_id(tenant_id)
            tenant_id = canonical_tenant_id(tenant_id)
        return await asyncio.to_thread(
            HarnessKeyStore(config_manager.store).list,
            tenant_id,
            page_size=page_size,
            continuation=continuation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConfigStoreUnavailableError as exc:
        raise _harness_key_store_unavailable(exc) from exc


@router.delete("/harness/keys/{key_hash}")
async def revoke_harness_key(
    key_hash: str = Path(pattern=r"^[0-9a-f]{64}$"),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
):
    try:
        revoked = await asyncio.to_thread(
            HarnessKeyStore(config_manager.store).revoke,
            key_hash,
        )
        return {"revoked": revoked, "key_hash": key_hash}
    except ConfigStoreUnavailableError as exc:
        raise _harness_key_store_unavailable(exc) from exc


@router.post("/graph/merge-article-nodes")
async def merge_article_nodes(
    dry_run: bool = Query(True, description="Report the merges without writing."),
    tenant_id: Optional[str] = Query(
        None, min_length=1, description="Scope to one tenant; default every tenant."
    ),
    exclude: Optional[List[str]] = Query(
        None,
        description="Article node ids never to merge; repeatable or comma-separated.",
    ),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Dict[str, Any]:
    """Merge knowledge-graph nodes whose id is ``<the|a|an>_<id>`` into the
    tenant's ``<id>`` node, re-pointing their edges and content back-refs.

    See ``cogniverse_agents.graph.article_node_migration`` for the rule and
    the merge semantics. 400 when an ``exclude`` entry is not an article node
    id; 404 when ``tenant_id`` has no deployed graph.
    """
    from cogniverse_agents.graph import article_node_migration
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

    tenant_ids = None
    if tenant_id is not None:
        try:
            tenant_ids = [canonical_tenant_id(tenant_id)]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    excluded: List[str] = []
    for entry in exclude or []:
        for article_id in (part.strip() for part in entry.split(",")):
            if not article_id or article_id in excluded:
                continue
            try:
                excluded.append(article_node_migration.validate_article_id(article_id))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    def resolve_graph_backend():
        return BackendRegistry.get_instance().get_ingestion_backend(
            "vespa",
            tenant_id=SYSTEM_TENANT_ID,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

    try:
        reports = await asyncio.to_thread(
            article_node_migration.merge_article_nodes,
            resolve_graph_backend,
            tenant_ids=tenant_ids,
            apply=not dry_run,
            exclude=excluded,
        )
    except article_node_migration.UnknownGraphTenantError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"dry_run": dry_run, "tenants": [r.to_dict() for r in reports]}
