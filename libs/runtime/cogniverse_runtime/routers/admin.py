"""Admin endpoints - system administration.

Note: Tenant management is available through the standalone tenant_manager app.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from cogniverse_core.config.config_manager import get_config_manager
from cogniverse_core.config.unified_config import BackendProfileConfig
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.validation.profile_validator import ProfileValidator
from fastapi import APIRouter, HTTPException

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

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize ConfigManager and ProfileValidator globally
_config_manager = None
_profile_validator = None


def get_config_manager_instance():
    """Get or create ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = get_config_manager()
    return _config_manager


def get_profile_validator_instance():
    """Get or create ProfileValidator instance."""
    global _profile_validator
    if _profile_validator is None:
        config_manager = get_config_manager_instance()
        _profile_validator = ProfileValidator(config_manager)
    return _profile_validator


# Tenant management endpoints removed - use standalone tenant_manager app
# See: libs/runtime/cogniverse_runtime/admin/tenant_manager.py


@router.get("/system/stats")
async def get_system_stats(tenant_id: str, backend: str) -> Dict[str, Any]:
    """Get system statistics."""
    try:
        backend_registry = BackendRegistry.get_instance()
        backend_instance = backend_registry.get_ingestion_backend(backend, tenant_id=tenant_id)
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


# ============================================================================
# Profile Management Endpoints
# ============================================================================


@router.post("/profiles", response_model=ProfileCreateResponse, status_code=201)
async def create_profile(request: ProfileCreateRequest) -> ProfileCreateResponse:
    """
    Create a new backend profile.

    Creates a profile configuration and optionally deploys the schema to Vespa.
    Profiles are tenant-scoped and versioned.

    Args:
        request: Profile creation request

    Returns:
        Profile creation response with deployment status

    Raises:
        HTTPException 400: Validation errors
        HTTPException 409: Profile already exists
        HTTPException 500: Creation or deployment failed
    """
    try:
        config_manager = get_config_manager_instance()
        validator = get_profile_validator_instance()

        # Create BackendProfileConfig from request
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

        # Validate profile
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

        # Add profile to ConfigManager
        config_manager.add_backend_profile(
            profile, tenant_id=request.tenant_id, service="backend"
        )

        # Optionally deploy schema
        schema_deployed = False
        tenant_schema_name = None

        if request.deploy_schema:
            try:
                backend_registry = BackendRegistry.get_instance()
                backend = backend_registry.get_ingestion_backend(
                    "vespa", tenant_id=request.tenant_id
                )

                if backend:
                    success = backend.deploy_schema(
                        schema_name=request.schema_name, tenant_id=request.tenant_id
                    )
                    schema_deployed = success

                    if success:
                        tenant_schema_name = backend.get_tenant_schema_name(
                            request.tenant_id, request.schema_name
                        )
                        logger.info(
                            f"Deployed schema '{tenant_schema_name}' for profile '{request.profile_name}'"
                        )
                else:
                    logger.warning("Backend not available for schema deployment")

            except Exception as e:
                logger.error(f"Schema deployment failed: {e}")
                # Don't fail profile creation if deployment fails
                # User can deploy later via /deploy endpoint

        return ProfileCreateResponse(
            profile_name=request.profile_name,
            tenant_id=request.tenant_id,
            schema_deployed=schema_deployed,
            tenant_schema_name=tenant_schema_name,
            created_at=datetime.now().isoformat(),
            version=1,  # First version
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles", response_model=ProfileListResponse)
async def list_profiles(tenant_id: str) -> ProfileListResponse:
    """
    List all backend profiles for a tenant.

    Args:
        tenant_id: Tenant identifier (query parameter)

    Returns:
        List of profile summaries

    Raises:
        HTTPException 500: List operation failed
    """
    try:
        config_manager = get_config_manager_instance()

        # Get all profiles for tenant
        profiles = config_manager.list_backend_profiles(
            tenant_id=tenant_id, service="backend"
        )

        # Convert to summary format
        profile_summaries = []
        backend_registry = BackendRegistry.get_instance()

        for profile_name, profile in profiles.items():
            # Check if schema is deployed
            schema_deployed = False
            try:
                backend = backend_registry.get_ingestion_backend("vespa", tenant_id=tenant_id)
                if backend:
                    schema_deployed = backend.schema_exists(
                        schema_name=profile.schema_name, tenant_id=tenant_id
                    )
            except Exception:
                pass  # If check fails, assume not deployed

            profile_summaries.append(
                ProfileSummary(
                    profile_name=profile_name,
                    type=profile.type,
                    description=profile.description,
                    schema_name=profile.schema_name,
                    embedding_model=profile.embedding_model,
                    schema_deployed=schema_deployed,
                    created_at=datetime.now().isoformat(),  # TODO: Get actual creation time
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
async def get_profile(profile_name: str, tenant_id: str) -> ProfileDetail:
    """
    Get a specific backend profile.

    Args:
        profile_name: Profile name (path parameter)
        tenant_id: Tenant identifier (query parameter)

    Returns:
        Detailed profile information

    Raises:
        HTTPException 404: Profile not found
        HTTPException 500: Get operation failed
    """
    try:
        config_manager = get_config_manager_instance()

        # Get profile
        profile = config_manager.get_backend_profile(
            profile_name=profile_name, tenant_id=tenant_id, service="backend"
        )

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found for tenant '{tenant_id}'",
            )

        # Check if schema is deployed
        schema_deployed = False
        tenant_schema_name = None

        try:
            backend_registry = BackendRegistry.get_instance()
            backend = backend_registry.get_ingestion_backend("vespa", tenant_id=tenant_id)
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
            created_at=datetime.now().isoformat(),  # TODO: Get actual creation time
            version=1,  # TODO: Get actual version from config store
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profiles/{profile_name}", response_model=ProfileUpdateResponse)
async def update_profile(
    profile_name: str, request: ProfileUpdateRequest
) -> ProfileUpdateResponse:
    """
    Update a backend profile.

    Only mutable fields can be updated (pipeline_config, strategies, description).
    Schema-related fields cannot be updated - create a new profile instead.

    Args:
        profile_name: Profile name (path parameter)
        request: Update request with fields to change

    Returns:
        Update response with updated fields

    Raises:
        HTTPException 400: Invalid update (trying to update immutable fields)
        HTTPException 404: Profile not found
        HTTPException 500: Update operation failed
    """
    try:
        config_manager = get_config_manager_instance()
        validator = get_profile_validator_instance()

        # Get existing profile
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

        # Build overrides dictionary (only include non-None fields)
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
            raise HTTPException(
                status_code=400, detail="No fields to update provided"
            )

        # Validate update fields
        validation_errors = validator.validate_update_fields(overrides)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid update fields",
                    "errors": validation_errors,
                },
            )

        # Update profile
        config_manager.update_backend_profile(
            profile_name=profile_name,
            overrides=overrides,
            base_tenant_id=request.tenant_id,
            target_tenant_id=request.tenant_id,
            service="backend",
        )

        return ProfileUpdateResponse(
            profile_name=profile_name,
            tenant_id=request.tenant_id,
            updated_fields=updated_fields,
            version=2,  # TODO: Get actual incremented version
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profiles/{profile_name}", response_model=ProfileDeleteResponse)
async def delete_profile(
    profile_name: str, tenant_id: str, delete_schema: bool = False
) -> ProfileDeleteResponse:
    """
    Delete a backend profile.

    Optionally delete the associated schema (use with caution).

    Args:
        profile_name: Profile name (path parameter)
        tenant_id: Tenant identifier (query parameter)
        delete_schema: Whether to also delete the schema (query parameter, default: false)

    Returns:
        Deletion confirmation

    Raises:
        HTTPException 404: Profile not found
        HTTPException 409: Cannot delete schema (other profiles using it)
        HTTPException 500: Deletion failed
    """
    try:
        config_manager = get_config_manager_instance()

        # Check if profile exists
        profile = config_manager.get_backend_profile(
            profile_name=profile_name, tenant_id=tenant_id, service="backend"
        )

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found for tenant '{tenant_id}'",
            )

        schema_deleted = False

        # Delete schema if requested
        if delete_schema:
            # Check if other profiles use this schema
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

            # Delete schema
            try:
                backend_registry = BackendRegistry.get_instance()
                backend = backend_registry.get_ingestion_backend("vespa", tenant_id=tenant_id)
                if backend:
                    deleted_schemas = backend.delete_schema(
                        schema_name=profile.schema_name, tenant_id=tenant_id
                    )
                    schema_deleted = len(deleted_schemas) > 0
            except Exception as e:
                logger.error(f"Failed to delete schema: {e}")
                # Continue with profile deletion even if schema deletion fails

        # Delete profile
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


@router.post(
    "/profiles/{profile_name}/deploy", response_model=SchemaDeploymentResponse
)
async def deploy_profile_schema(
    profile_name: str, request: SchemaDeploymentRequest
) -> SchemaDeploymentResponse:
    """
    Deploy schema for a backend profile.

    Deploys the Vespa schema associated with the profile to the tenant's namespace.

    Args:
        profile_name: Profile name (path parameter)
        request: Deployment request

    Returns:
        Deployment status

    Raises:
        HTTPException 404: Profile not found
        HTTPException 500: Deployment failed
    """
    try:
        config_manager = get_config_manager_instance()

        # Get profile
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

        # Check if schema already deployed (unless force=True)
        backend_registry = BackendRegistry.get_instance()
        backend = backend_registry.get_ingestion_backend("vespa", tenant_id=request.tenant_id)

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

        # Deploy schema
        try:
            success = backend.deploy_schema(
                schema_name=profile.schema_name, tenant_id=request.tenant_id
            )

            tenant_schema_name = backend.get_tenant_schema_name(
                request.tenant_id, profile.schema_name
            )

            if success:
                return SchemaDeploymentResponse(
                    profile_name=profile_name,
                    tenant_id=request.tenant_id,
                    schema_name=profile.schema_name,
                    tenant_schema_name=tenant_schema_name,
                    deployment_status="success",
                    deployed_at=datetime.now().isoformat(),
                )
            else:
                return SchemaDeploymentResponse(
                    profile_name=profile_name,
                    tenant_id=request.tenant_id,
                    schema_name=profile.schema_name,
                    tenant_schema_name=tenant_schema_name,
                    deployment_status="failed",
                    deployed_at=datetime.now().isoformat(),
                    error_message="Schema deployment returned false",
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
