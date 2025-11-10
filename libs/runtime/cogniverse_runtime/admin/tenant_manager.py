"""
Tenant Management API

Provides CRUD operations for organizations and tenants.
NO auth required in Phase 7.8 - auth integration deferred to Phase 8.

Architecture:
- org:tenant format ("acme:production")
- Auto-create org when creating first tenant
- Vespa for metadata storage
- Per-tenant schema deployment
- Same API for all user types (billing limits differentiate tiers)

Example Usage:
    # Create organization
    POST /admin/organizations
    {"org_id": "acme", "org_name": "Acme Corp", "created_by": "admin"}

    # Create tenant (auto-creates org if needed)
    POST /admin/tenants
    {"tenant_id": "acme:production", "created_by": "admin"}

    # List tenants
    GET /admin/organizations/acme/tenants

    # Delete tenant
    DELETE /admin/tenants/acme:production
"""

import logging
import time
from typing import Dict, List, Optional

import uvicorn
from cogniverse_core.common.tenant_utils import parse_tenant_id
from cogniverse_foundation.config.utils import get_config
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader
from fastapi import FastAPI, HTTPException

from cogniverse_runtime.admin.models import (
    CreateOrganizationRequest,
    CreateTenantRequest,
    Organization,
    OrganizationListResponse,
    Tenant,
    TenantListResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tenant Management API",
    description="Organization and tenant CRUD operations (Phase 7.8 - no auth)",
    version="1.0.0",
)

# Backend for metadata storage and schema management
backend: Optional[Backend] = None
_config_manager = None  # For test injection
_schema_loader: SchemaLoader = None  # For dependency injection


def set_config_manager(config_manager):
    """Set ConfigManager for this module (for tests)"""
    global _config_manager
    _config_manager = config_manager


def set_schema_loader(schema_loader: SchemaLoader) -> None:
    """
    Set the SchemaLoader instance for this module.

    Must be called during application startup before handling requests.

    Args:
        schema_loader: SchemaLoader instance to use
    """
    global _schema_loader
    _schema_loader = schema_loader


def get_backend() -> Backend:
    """Get or create backend for metadata operations"""
    global backend
    if backend is None:
        # Use injected ConfigManager (from tests) or create new one
        from cogniverse_foundation.config.utils import create_default_config_manager
        config_manager = _config_manager if _config_manager is not None else create_default_config_manager()

        config = get_config(tenant_id="system", config_manager=config_manager)
        backend_type = config.get("backend_type", "vespa")

        from cogniverse_core.registries.backend_registry import BackendRegistry
        registry = BackendRegistry.get_instance()

        # Get backend instance with configuration
        backend_config = {
            "url": config.get("backend_url"),
            "port": config.get("backend_port"),
        }

        # Get backend WITHOUT tenant_id (this is for metadata operations across all tenants)
        # We'll pass tenant_id explicitly when needed for schema operations
        try:
            # Require injected SchemaLoader
            if _schema_loader is None:
                raise RuntimeError(
                    "SchemaLoader not initialized. Call set_schema_loader() during app startup."
                )
            schema_loader = _schema_loader

            backend = registry.get_ingestion_backend(
                backend_type, tenant_id="system", config=backend_config, config_manager=config_manager, schema_loader=schema_loader
            )
        except Exception as e:
            logger.error(f"Failed to get backend: {e}")
            raise

        logger.info(f"Initialized {backend_type} backend for tenant management")

    return backend


def validate_org_id(org_id: str) -> None:
    """Validate organization ID format"""
    if not org_id:
        raise ValueError("org_id cannot be empty")

    if not isinstance(org_id, str):
        raise ValueError(f"org_id must be string, got {type(org_id)}")

    if not org_id.replace("_", "").isalnum():
        raise ValueError(
            f"Invalid org_id '{org_id}': only alphanumeric and underscore allowed"
        )


def validate_tenant_name(tenant_name: str) -> None:
    """Validate tenant name format"""
    if not tenant_name:
        raise ValueError("tenant_name cannot be empty")

    if not isinstance(tenant_name, str):
        raise ValueError(f"tenant_name must be string, got {type(tenant_name)}")

    if not tenant_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"Invalid tenant_name '{tenant_name}': only alphanumeric, underscore, and hyphen allowed"
        )


# ============================================================================
# Organization Endpoints
# ============================================================================


@app.post("/admin/organizations", response_model=Organization)
async def create_organization(request: CreateOrganizationRequest) -> Organization:
    """
    Create a new organization with default tenant.

    Auto-creates a "default" tenant for the organization.

    Args:
        request: Organization creation request

    Returns:
        Created organization

    Raises:
        HTTPException 400: Invalid org_id format
        HTTPException 409: Organization already exists
        HTTPException 500: Creation failed
    """
    try:
        validate_org_id(request.org_id)

        backend = get_backend()

        # Check if org already exists
        existing = await get_organization_internal(request.org_id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Organization {request.org_id} already exists",
            )

        # Create organization
        org = Organization(
            org_id=request.org_id,
            org_name=request.org_name,
            created_at=int(time.time() * 1000),
            created_by=request.created_by,
            status="active",
            tenant_count=0,
        )

        # Store via Backend
        success = backend.create_metadata_document(
            schema="organization_metadata",
            doc_id=org.org_id,
            fields={
                "org_id": org.org_id,
                "org_name": org.org_name,
                "created_at": org.created_at,
                "created_by": org.created_by,
                "status": org.status,
                "tenant_count": org.tenant_count,
            },
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create organization {org.org_id} in backend"
            )

        logger.info(f"Created organization: {org.org_id}")
        return org

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create organization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/organizations", response_model=OrganizationListResponse)
async def list_organizations() -> OrganizationListResponse:
    """
    List all organizations.

    Returns:
        List of all organizations with count
    """
    try:
        backend = get_backend()

        # Query all organizations
        documents = backend.query_metadata_documents(
            schema="organization_metadata",
            yql="select * from organization_metadata where true",
            hits=400,
        )

        organizations = []
        for fields in documents:
            org_id = fields.get("org_id")

            # Compute tenant_count dynamically
            tenants = await list_tenants_for_org_internal(org_id)

            org = Organization(
                org_id=org_id,
                org_name=fields.get("org_name"),
                created_at=fields.get("created_at"),
                created_by=fields.get("created_by"),
                status=fields.get("status", "active"),
                tenant_count=len(tenants),
            )
            organizations.append(org)

        return OrganizationListResponse(
            organizations=organizations, total_count=len(organizations)
        )

    except Exception as e:
        logger.error(f"Failed to list organizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/organizations/{org_id}", response_model=Organization)
async def get_organization(org_id: str) -> Organization:
    """
    Get single organization by ID.

    Args:
        org_id: Organization identifier

    Returns:
        Organization details

    Raises:
        HTTPException 404: Organization not found
    """
    org = await get_organization_internal(org_id)
    if not org:
        raise HTTPException(status_code=404, detail=f"Organization {org_id} not found")
    return org


async def get_organization_internal(org_id: str) -> Optional[Organization]:
    """Internal helper to get organization"""
    try:
        backend = get_backend()

        fields = backend.get_metadata_document(
            schema="organization_metadata", doc_id=org_id
        )

        if not fields:
            return None

        # Compute tenant_count dynamically by querying tenants
        tenants = await list_tenants_for_org_internal(org_id)

        return Organization(
            org_id=fields.get("org_id"),
            org_name=fields.get("org_name"),
            created_at=fields.get("created_at"),
            created_by=fields.get("created_by"),
            status=fields.get("status", "active"),
            tenant_count=len(tenants),
        )

    except Exception as e:
        logger.warning(f"Error getting organization {org_id}: {e}")
        return None


@app.delete("/admin/organizations/{org_id}")
async def delete_organization(org_id: str) -> Dict:
    """
    Delete organization and all its tenants.

    WARNING: This removes all data for the organization!

    Args:
        org_id: Organization identifier

    Returns:
        Deletion summary

    Raises:
        HTTPException 404: Organization not found
        HTTPException 500: Deletion failed
    """
    try:
        validate_org_id(org_id)

        # Check org exists
        org = await get_organization_internal(org_id)
        if not org:
            raise HTTPException(
                status_code=404, detail=f"Organization {org_id} not found"
            )

        backend = get_backend()

        # Delete all tenants for this org
        tenants = await list_tenants_for_org_internal(org_id)
        deleted_tenants = []

        for tenant in tenants:
            try:
                await delete_tenant_internal(tenant.tenant_full_id)
                deleted_tenants.append(tenant.tenant_full_id)
            except Exception as e:
                logger.error(f"Failed to delete tenant {tenant.tenant_full_id}: {e}")

        # Delete organization
        backend.delete_metadata_document(schema="organization_metadata", doc_id=org_id)

        logger.info(
            f"Deleted organization {org_id} with {len(deleted_tenants)} tenants"
        )

        return {
            "status": "deleted",
            "org_id": org_id,
            "tenants_deleted": len(deleted_tenants),
            "deleted_tenant_ids": deleted_tenants,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete organization {org_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tenant Endpoints
# ============================================================================


@app.post("/admin/tenants", response_model=Tenant)
async def create_tenant(request: CreateTenantRequest) -> Tenant:
    """
    Create a new tenant (auto-creates org if doesn't exist).

    Args:
        request: Tenant creation request

    Returns:
        Created tenant

    Raises:
        HTTPException 400: Invalid tenant_id format
        HTTPException 409: Tenant already exists
        HTTPException 500: Creation failed

    Example:
        POST /admin/tenants
        {"tenant_id": "acme:production", "created_by": "admin"}
    """
    try:
        # Parse tenant_id
        # If org_id provided separately and tenant_id is simple format, construct full ID
        tenant_id_to_parse = request.tenant_id
        if request.org_id and ":" not in request.tenant_id:
            tenant_id_to_parse = f"{request.org_id}:{request.tenant_id}"

        org_id, tenant_name = parse_tenant_id(tenant_id_to_parse)

        validate_org_id(org_id)
        validate_tenant_name(tenant_name)

        tenant_full_id = f"{org_id}:{tenant_name}"

        backend = get_backend()

        # Check if tenant already exists
        existing = await get_tenant_internal(tenant_full_id)
        if existing:
            raise HTTPException(
                status_code=409, detail=f"Tenant {tenant_full_id} already exists"
            )

        # Auto-create org if doesn't exist
        org_created = False
        org = await get_organization_internal(org_id)
        if not org:
            logger.info(f"Auto-creating organization {org_id} for tenant {tenant_full_id}")
            org = Organization(
                org_id=org_id,
                org_name=org_id.title(),  # Use org_id as name
                created_at=int(time.time() * 1000),
                created_by=request.created_by,
                status="active",
                tenant_count=0,  # Not used, computed dynamically
            )

            success = backend.create_metadata_document(
                schema="organization_metadata",
                doc_id=org.org_id,
                fields={
                    "org_id": org.org_id,
                    "org_name": org.org_name,
                    "created_at": org.created_at,
                    "created_by": org.created_by,
                    "status": org.status,
                    "tenant_count": org.tenant_count,
                },
            )
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to auto-create organization {org.org_id} in backend"
                )
            org_created = True

        # Deploy schemas for tenant via Backend
        base_schemas = request.base_schemas or [
            "video_colpali_smol500_mv_frame",
        ]

        deployed_schemas = []
        for base_schema in base_schemas:
            try:
                backend.schema_registry.deploy_schema(
                    tenant_id=tenant_full_id,
                    base_schema_name=base_schema
                )
                deployed_schemas.append(base_schema)
            except Exception as e:
                logger.error(f"Failed to deploy schema {base_schema} for {tenant_full_id}: {e}")

        # Create tenant
        tenant = Tenant(
            tenant_full_id=tenant_full_id,
            org_id=org_id,
            tenant_name=tenant_name,
            created_at=int(time.time() * 1000),
            created_by=request.created_by,
            status="active",
            schemas_deployed=deployed_schemas,
        )

        # Store via Backend
        success = backend.create_metadata_document(
            schema="tenant_metadata",
            doc_id=tenant_full_id,
            fields={
                "tenant_full_id": tenant.tenant_full_id,
                "org_id": tenant.org_id,
                "tenant_name": tenant.tenant_name,
                "created_at": tenant.created_at,
                "created_by": tenant.created_by,
                "status": tenant.status,
                "schemas_deployed": tenant.schemas_deployed,
            },
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create tenant {tenant_full_id} in backend"
            )

        logger.info(
            f"Created tenant: {tenant_full_id} (org_created: {org_created}, schemas: {len(deployed_schemas)})"
        )

        return tenant

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create tenant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/admin/organizations/{org_id}/tenants", response_model=TenantListResponse
)
async def list_tenants_for_org(org_id: str) -> TenantListResponse:
    """
    List all tenants for an organization.

    Args:
        org_id: Organization identifier

    Returns:
        List of tenants in the organization

    Raises:
        HTTPException 404: Organization not found
    """
    try:
        validate_org_id(org_id)

        # Verify org exists
        org = await get_organization_internal(org_id)
        if not org:
            raise HTTPException(
                status_code=404, detail=f"Organization {org_id} not found"
            )

        tenants = await list_tenants_for_org_internal(org_id)

        return TenantListResponse(
            tenants=tenants, total_count=len(tenants), org_id=org_id
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to list tenants for {org_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def list_tenants_for_org_internal(org_id: str) -> List[Tenant]:
    """Internal helper to list tenants"""
    try:
        backend = get_backend()

        # Query tenants for this org using term matching in userQuery
        documents = backend.query_metadata_documents(
            schema='tenant_metadata',
            yql='select * from tenant_metadata where userQuery()',
            query=f'org_id:{org_id}',
            hits=400,
        )

        tenants = []
        for fields in documents:
            tenant = Tenant(
                tenant_full_id=fields.get("tenant_full_id"),
                org_id=fields.get("org_id"),
                tenant_name=fields.get("tenant_name"),
                created_at=fields.get("created_at"),
                created_by=fields.get("created_by"),
                status=fields.get("status", "active"),
                schemas_deployed=fields.get("schemas_deployed", []),
            )
            tenants.append(tenant)

        return tenants

    except Exception as e:
        logger.error(f"Failed to list tenants for {org_id}: {e}")
        return []


@app.get("/admin/tenants/{tenant_full_id}", response_model=Tenant)
async def get_tenant(tenant_full_id: str) -> Tenant:
    """
    Get single tenant by full ID.

    Args:
        tenant_full_id: Full tenant ID (org:tenant)

    Returns:
        Tenant details

    Raises:
        HTTPException 404: Tenant not found
    """
    tenant = await get_tenant_internal(tenant_full_id)
    if not tenant:
        raise HTTPException(
            status_code=404, detail=f"Tenant {tenant_full_id} not found"
        )
    return tenant


async def get_tenant_internal(tenant_full_id: str) -> Optional[Tenant]:
    """Internal helper to get tenant"""
    try:
        backend = get_backend()

        fields = backend.get_metadata_document(
            schema="tenant_metadata", doc_id=tenant_full_id
        )

        if not fields:
            return None
        return Tenant(
            tenant_full_id=fields.get("tenant_full_id"),
            org_id=fields.get("org_id"),
            tenant_name=fields.get("tenant_name"),
            created_at=fields.get("created_at"),
            created_by=fields.get("created_by"),
            status=fields.get("status", "active"),
            schemas_deployed=fields.get("schemas_deployed", []),
        )

    except Exception as e:
        logger.warning(f"Error getting tenant {tenant_full_id}: {e}")
        return None


@app.delete("/admin/tenants/{tenant_full_id}")
async def delete_tenant(tenant_full_id: str) -> Dict:
    """
    Delete tenant and its schemas.

    WARNING: This removes all data for the tenant!

    Args:
        tenant_full_id: Full tenant ID (org:tenant)

    Returns:
        Deletion summary

    Raises:
        HTTPException 404: Tenant not found
        HTTPException 500: Deletion failed
    """
    try:
        result = await delete_tenant_internal(tenant_full_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete tenant {tenant_full_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_tenant_internal(tenant_full_id: str) -> Dict:
    """Internal helper to delete tenant"""
    # Get tenant
    tenant = await get_tenant_internal(tenant_full_id)
    if not tenant:
        raise HTTPException(
            status_code=404, detail=f"Tenant {tenant_full_id} not found"
        )

    backend = get_backend()

    # Delete tenant schemas
    deleted_schemas = []
    try:
        schemas = backend.delete_schema(schema_name=None, tenant_id=tenant_full_id)
        deleted_schemas.extend(schemas)
    except Exception as e:
        logger.error(f"Failed to delete schemas for {tenant_full_id}: {e}")

    # Delete tenant metadata
    backend.delete_metadata_document(schema="tenant_metadata", doc_id=tenant_full_id)

    logger.info(f"Deleted tenant {tenant_full_id} with {len(deleted_schemas)} schemas")

    return {
        "status": "deleted",
        "tenant_full_id": tenant_full_id,
        "schemas_deleted": len(deleted_schemas),
        "deleted_schemas": deleted_schemas,
    }


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tenant_manager",
        "version": "1.0.0",
        "features": [
            "organization_management",
            "tenant_management",
            "auto_org_creation",
            "schema_deployment",
        ],
    }


if __name__ == "__main__":
    from cogniverse_foundation.config.utils import create_default_config_manager
    config_manager = create_default_config_manager()
    config = get_config(tenant_id="default", config_manager=config_manager)
    port = config.get("tenant_manager_port", 9000)

    logger.info(f"Starting Tenant Management API on port {port}")
    logger.info(f"Organization API: http://localhost:{port}/admin/organizations")
    logger.info(f"Tenant API: http://localhost:{port}/admin/tenants")

    uvicorn.run(app, host="0.0.0.0", port=port)
