"""
Tenant Management API

Provides CRUD operations for organizations and tenants. Auth
integration is not yet wired; the runtime currently trusts callers
to pass tenant_id and actor identity in request bodies.

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

import asyncio
import logging
import time
from typing import Dict, List, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID, parse_tenant_id
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.admin.models import (
    CreateOrganizationRequest,
    CreateTenantRequest,
    Organization,
    OrganizationListResponse,
    Tenant,
    TenantListResponse,
)
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

# Router for tenant management endpoints (mountable by Runtime)
router = APIRouter()

# Standalone app (for running tenant_manager independently)
app = FastAPI(
    title="Tenant Management API",
    description="Organization and tenant CRUD operations (auth not yet integrated)",
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

        config_manager = (
            _config_manager
            if _config_manager is not None
            else create_default_config_manager()
        )

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
                backend_type,
                tenant_id="system",
                config=backend_config,
                config_manager=config_manager,
                schema_loader=schema_loader,
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

    # No hyphens: the tenant name becomes part of the Vespa schema name
    # (which allows only [a-zA-Z0-9_]), and sanitizing "-"→"_" would collide
    # distinct tenants (acme-corp vs acme_corp → same schema). Matches the
    # org_id rule above.
    if not tenant_name.replace("_", "").isalnum():
        raise ValueError(
            f"Invalid tenant_name '{tenant_name}': only alphanumeric and underscore allowed"
        )


# ============================================================================
# Organization Endpoints
# ============================================================================


@router.post("/organizations", response_model=Organization)
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
                detail=f"Failed to create organization {org.org_id} in backend",
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


@router.get("/organizations", response_model=OrganizationListResponse)
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


@router.get("/organizations/{org_id}", response_model=Organization)
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
    backend = get_backend()

    try:
        # Blocking Vespa GET — off the event loop.
        fields = await asyncio.to_thread(
            backend.get_metadata_document,
            schema="organization_metadata",
            doc_id=org_id,
        )
    except Exception as e:
        # Outage is not "org not found" — surface 503 so a create/read during a
        # backend blip doesn't 404 (or, for create, clobber a live org read as
        # missing).
        logger.error(f"Organization registry read failed for {org_id}: {e}")
        raise HTTPException(
            status_code=503, detail="Organization registry temporarily unavailable"
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


@router.delete("/organizations/{org_id}")
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


@router.post("/tenants", response_model=Tenant)
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
        HTTPException 502: A requested schema failed to deploy (no tenant created)
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
            logger.info(
                f"Auto-creating organization {org_id} for tenant {tenant_full_id}"
            )
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
                    detail=f"Failed to auto-create organization {org.org_id} in backend",
                )
            org_created = True

        # Deploy schemas for tenant via Backend
        base_schemas = request.base_schemas or [
            "video_colpali_smol500_mv_frame",
        ]

        deployed_schemas = []
        failed_schemas = []
        for base_schema in base_schemas:
            try:
                await asyncio.to_thread(
                    backend.schema_registry.deploy_schema,
                    tenant_id=tenant_full_id,
                    base_schema_name=base_schema,
                )
                deployed_schemas.append(base_schema)
            except Exception as e:
                logger.error(
                    f"Failed to deploy schema {base_schema} for {tenant_full_id}: {e}"
                )
                failed_schemas.append(base_schema)

        if failed_schemas:
            # Do not create an active tenant whose schemas didn't deploy. Such a
            # tenant silently accepts writes that then hit an undeployed doc type
            # — every ingest/search fails or grinds in the feed-retry loop, and
            # graph upsert reports success having persisted nothing. Fail loud so
            # the operator can fix the Vespa config server and retry.
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Tenant {tenant_full_id} not created: schema deploy failed for "
                    f"{failed_schemas}. Check the Vespa config server and retry."
                ),
            )

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
                detail=f"Failed to create tenant {tenant_full_id} in backend",
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


@router.get("/organizations/{org_id}/tenants", response_model=TenantListResponse)
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


async def list_organizations_internal() -> List[str]:
    """Internal helper to list every org_id known to the backend.

    Lives next to ``list_tenants_for_org_internal`` so callers that need
    a global tenant sweep (e.g. the daily-cleanup CronWorkflow) can
    enumerate without going through the FastAPI HTTPException-raising
    route. Returns an empty list rather than raising on backend error
    so a single bad organization document does not crash the sweep.
    """
    try:
        backend = get_backend()
        documents = backend.query_metadata_documents(
            schema="organization_metadata",
            yql="select * from organization_metadata where true",
            hits=400,
        )
        return [fields["org_id"] for fields in documents if fields.get("org_id")]
    except Exception as e:
        logger.error(f"Failed to list organizations: {e}")
        return []


async def list_tenants_for_org_internal(org_id: str) -> List[Tenant]:
    """Internal helper to list tenants"""
    try:
        backend = get_backend()

        # Query tenants for this org using term matching in userQuery
        documents = backend.query_metadata_documents(
            schema="tenant_metadata",
            yql="select * from tenant_metadata where userQuery()",
            query=f"org_id:{org_id}",
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


@router.get("/tenants/{tenant_full_id}", response_model=Tenant)
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
    """Internal helper to get tenant.

    Normalizes ``tenant_full_id`` via ``canonical_tenant_id`` so simple-form
    inputs (``acme``) resolve to the same doc_id POST stored under
    (``acme:acme``). Without this, GET /admin/tenants/{tid} returns 404 even
    immediately after a successful POST that used the simple form.
    """
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    backend = get_backend()
    canonical = canonical_tenant_id(tenant_full_id)

    try:
        # Blocking Vespa GET — run off the event loop; this sits under
        # assert_tenant_exists on every search/ingestion/graph request.
        fields = await asyncio.to_thread(
            backend.get_metadata_document, schema="tenant_metadata", doc_id=canonical
        )
    except Exception as e:
        # A backend outage is NOT "tenant not found". Surface 503 so callers
        # retry, instead of a permanent-looking 404 on every tenant-scoped
        # request during a Vespa blip (which reads as "the tenant was deleted").
        logger.error(f"Tenant registry read failed for {tenant_full_id}: {e}")
        raise HTTPException(
            status_code=503, detail="Tenant registry temporarily unavailable"
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


@router.delete("/tenants/{tenant_full_id}")
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


def _discover_orphan_schema_targets(
    deployed_full_names: List[str], canonical_tid: str
) -> set:
    """Attribute deployed Vespa schemas to (canonical_tid, base) for deletion.

    Matches the canonical (doubled) tenant suffix only. Every deploy path
    canonicalizes the tenant id (schema_registry.deploy_schema /
    backend.get_tenant_schema_name), so a tenant's live schemas always carry
    the canonical suffix. The bare tenant suffix is deliberately NOT matched:
    it is short enough to be a false suffix of another tenant's schema
    (``base_a_pr`` ends in ``_pr``), which would over-delete across tenants.
    Registered bare-form schemas are still covered by the registry
    lookup above; unregistered orphans are handled by /admin/reconcile-orphans.
    """
    canonical_suffix = "_" + canonical_tid.replace(":", "_")
    targets: set[tuple[str, str]] = set()
    for full_name in deployed_full_names:
        if full_name.endswith(canonical_suffix):
            targets.add((canonical_tid, full_name[: -len(canonical_suffix)]))
    return targets


async def delete_tenant_internal(tenant_full_id: str) -> Dict:
    """Delete a tenant's schemas and metadata.

    Looks up tenant_metadata under the canonical form (POST stored it that
    way) and discovers schemas from the registry (both id forms) plus a
    canonical-suffix match against Vespa-deployed names. Every deploy path
    canonicalizes the tenant id, so live schemas carry the canonical suffix;
    the bare suffix is intentionally not matched because it over-deletes
    across tenants (``base_a_pr`` ends in ``_pr``).
    """
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    original_tid = tenant_full_id
    canonical_tid = canonical_tenant_id(tenant_full_id)
    tenant = await get_tenant_internal(canonical_tid)

    backend = get_backend()
    schema_manager = backend.schema_manager
    schema_registry = schema_manager._schema_registry

    # (tenant_id, base_name) pairs — preserves which form to pass to
    # delete_schema so it computes the correct full schema name.
    targets: set[tuple[str, str]] = set()

    if schema_registry is not None:
        for tid in {original_tid, canonical_tid}:
            for info in schema_registry.get_tenant_schemas(tid):
                targets.add((tid, info.base_schema_name))

    try:
        deployed_full_names = schema_manager.list_deployed_document_types()
    except Exception as e:
        deployed_full_names = []
        logger.warning(
            f"Vespa-side schema discovery failed for tenant "
            f"'{canonical_tid}' (continuing with registry-only set): {e}"
        )

    targets |= _discover_orphan_schema_targets(deployed_full_names, canonical_tid)

    # Allow schema-only orphans (no tenant_metadata record) to be cleaned
    # up — they're created by /ingestion/upload auto-deploy bypassing
    # tenant create, and accumulate every test run without this branch.
    if not tenant and not targets:
        raise HTTPException(status_code=404, detail=f"Tenant {canonical_tid} not found")

    deleted_schemas: list = []
    for tid, base_name in sorted(targets):
        try:
            full_name = schema_manager.delete_schema(tid, base_name)
            deleted_schemas.append(full_name)
        except Exception as e:
            logger.error(
                f"Failed to delete schema '{base_name}' for tenant '{tid}': {e}"
            )

    if tenant:
        backend.delete_metadata_document(schema="tenant_metadata", doc_id=canonical_tid)

    from cogniverse_core.common.tenant_utils import invalidate_tenant_exists

    invalidate_tenant_exists(canonical_tid)
    tenant_full_id = canonical_tid  # for the logger.info + return below

    logger.info(f"Deleted tenant {tenant_full_id} with {len(deleted_schemas)} schemas")

    return {
        "status": "deleted",
        "tenant_full_id": tenant_full_id,
        "schemas_deleted": len(deleted_schemas),
        "deleted_schemas": deleted_schemas,
    }


# ============================================================================
# Reconciliation
# ============================================================================


def _list_orphan_schemas() -> Dict[str, list]:
    """Diff Vespa-deployed schemas against the registry's active set.

    Returns a dict with two lists: ``orphan_schemas`` (Vespa-only full
    schema names) and ``orphan_tenants`` (tenants implied by stripping
    known base prefixes from those names). Bases unknown to the
    well-known list are reported in ``unrecovered_schemas``.
    """
    backend = get_backend()
    schema_manager = backend.schema_manager
    schema_registry = schema_manager._schema_registry

    deployed = set(schema_manager.list_deployed_document_types())
    registered = {
        info.full_schema_name for info in (schema_registry._get_all_schemas() or [])
    }
    orphans = sorted(deployed - registered - schema_manager._PROTECTED_SCHEMAS)

    # Recover tenant_id from the orphan name by stripping a known base
    # schema prefix. Bases that ship with the platform; if a deployment
    # adds new base schema names, extend this list.
    KNOWN_BASES = (
        "video_colpali_smol500_mv_frame",
        "video_videoprism_base_mv_chunk_30s",
        "video_videoprism_large_mv_chunk_30s",
        "video_colqwen_omni_mv_chunk_30s",
        "image_colpali_mv",
        "audio_clap_semantic",
        "audio_content",
        "document_text",
        "document_text_semantic",
        "knowledge_graph",
        "agent_memories",
        "wiki_pages",
        "code_lateon_mv",
        # Knowledge-system per-tenant provenance schema. Without this
        # entry the orphan reconciler couldn't strip provenance_<tid>
        # back to <tid>, so every Knowledge System e2e test left a
        # provenance schema behind that the next sweep tripped over
        # ("Refusing to deploy: Vespa has schemas X not in registry").
        "provenance",
    )
    orphan_tenants: set = set()
    unrecovered: list = []
    for orphan in orphans:
        for base in KNOWN_BASES:
            prefix = f"{base}_"
            if orphan.startswith(prefix):
                orphan_tenants.add(orphan[len(prefix) :])
                break
        else:
            unrecovered.append(orphan)
    return {
        "orphan_schemas": orphans,
        "orphan_tenants": sorted(orphan_tenants),
        "unrecovered_schemas": unrecovered,
    }


@router.post("/reconcile-orphans")
async def reconcile_orphans(
    dry_run: bool = Query(
        default=True,
        description=(
            "When true (default), report orphans without modifying state. "
            "Pass dry_run=false to actually drop the orphan schemas."
        ),
    ),
) -> Dict:
    """List Vespa-only orphan schemas, optionally drop them in one redeploy.

    Diffs Vespa's deployed schemas against the SchemaRegistry's active
    set. Orphans (Vespa has, registry doesn't) are grouped by the
    implied tenant_id by stripping known base prefixes.

    With ``dry_run=true`` returns the diff for operator review. With
    ``dry_run=false`` calls ``delete_tenant_schemas_bulk`` so all
    orphan tenants are dropped atomically (single redeploy) — required
    because individual tenant deletes refuse when a peer-tenant
    unreconstructable orphan exists.
    """
    diff = _list_orphan_schemas()
    if dry_run or not diff["orphan_tenants"]:
        return {
            "dry_run": dry_run,
            "deleted": [],
            **diff,
        }

    backend = get_backend()
    deleted = backend.schema_manager.delete_tenant_schemas_bulk(diff["orphan_tenants"])
    return {
        "dry_run": False,
        "deleted": deleted,
        **diff,
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


# Mount router on standalone app (after all endpoints are defined)
app.include_router(router, prefix="/admin")


if __name__ == "__main__":
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)
    port = config.get("tenant_manager_port", 9000)

    logger.info(f"Starting Tenant Management API on port {port}")
    logger.info(f"Organization API: http://localhost:{port}/admin/organizations")
    logger.info(f"Tenant API: http://localhost:{port}/admin/tenants")

    uvicorn.run(app, host="0.0.0.0", port=port)
