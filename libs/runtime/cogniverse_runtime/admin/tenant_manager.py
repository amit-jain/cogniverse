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
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, AsyncIterator, Callable, Dict, Iterator, List, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query
from requests import exceptions as requests_exceptions

from cogniverse_core.common.tenant_utils import (
    SYSTEM_TENANT_ID,
    canonical_tenant_id,
    clear_tenant_deleted,
    complete_tenant_delete,
    mark_tenant_deleted,
    parse_tenant_id,
    tenant_delete_pending,
)
from cogniverse_core.memory.manager import (
    MEMORY_BASE_SCHEMA,
    PROVENANCE_BASE_SCHEMA,
)
from cogniverse_core.registries.exceptions import RegistryStorageError
from cogniverse_core.registries.schema_deploy_lease import (
    LeaseWaitTimeout,
    SchemaDeployLease,
)
from cogniverse_core.registries.schema_registry import delete_tenant_refusals
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.admin.models import (
    CreateOrganizationRequest,
    CreateTenantRequest,
    Organization,
    OrganizationListResponse,
    SetTenantTierRequest,
    Tenant,
    TenantListResponse,
    TenantTier,
)
from cogniverse_runtime.cluster_events import ClusterEventError, ClusterEvents
from cogniverse_runtime.harness_keys import HarnessKeyStore
from cogniverse_runtime.http_errors import failure_response
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
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

_config_manager = None  # For test injection
_schema_loader: SchemaLoader = None  # For dependency injection
_backend: Backend | None = None  # Injected metadata backend; bypasses the registry
# Delivers a tenant delete to every runtime worker process; wired at startup.
_cluster_events: ClusterEvents | None = None

# How long a tenant delete waits for every worker to release the tenant.
TENANT_DELETE_ACK_TIMEOUT_S = 15.0

# How long a tenant create or delete waits for another create or delete of the
# same tenant, on any process, to finish.
TENANT_OPERATION_WAIT_S = 600.0

# Built once for processes that never inject one (standalone CLIs). The
# registry refuses a cached backend to a requester carrying a different
# ConfigManager, so resolving per call must not build a new one per call.
_fallback_config_manager = None
_fallback_config_manager_lock = threading.Lock()


def set_config_manager(config_manager):
    """Set ConfigManager for this module (for tests)"""
    global _config_manager
    _config_manager = config_manager


def set_cluster_events(cluster_events: ClusterEvents | None) -> None:
    """Wire the channel a tenant delete reaches every worker process through."""
    global _cluster_events
    _cluster_events = cluster_events


def release_deleted_tenant(payload: Dict) -> Dict:
    """Drop everything this process holds for a deleted tenant.

    The ``tenant_deleted`` cluster-event handler, run on every worker: the
    existence cache entry, every registered per-tenant cache (agents, graph
    and artifact managers), the tenant's warm memory manager, and its queued
    background memory writes. A write already running finishes against the
    deletion marker and is refused there.
    """
    from cogniverse_agents.background_memory_writes import (
        get_background_memory_writer,
    )
    from cogniverse_core.common.tenant_utils import invalidate_tenant_exists
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_foundation.caching import evict_tenant_from_registered_caches

    tenant_id = canonical_tenant_id(payload["tenant_id"])
    invalidate_tenant_exists(tenant_id)
    cache_entries = evict_tenant_from_registered_caches(tenant_id)
    org_id, tenant_name = tenant_id.split(":", 1)
    # Managers are keyed by the id their callers passed; "acme" names "acme:acme".
    keys = {tenant_id, payload["tenant_id"]} | (
        {org_id} if org_id == tenant_name else set()
    )
    memory_managers = sum(
        Mem0MemoryManager._instances.pop(key) is not None for key in keys
    )
    cancelled = get_background_memory_writer().cancel_tenant(tenant_id)
    return {
        "cache_entries": cache_entries,
        "memory_managers": memory_managers,
        "queued_memory_writes_cancelled": cancelled,
    }


def set_schema_loader(schema_loader: SchemaLoader) -> None:
    """
    Set the SchemaLoader instance for this module.

    Must be called during application startup before handling requests.

    Args:
        schema_loader: SchemaLoader instance to use
    """
    global _schema_loader
    _schema_loader = schema_loader


def set_backend(backend: Backend | None) -> None:
    """Inject the metadata backend, or ``None`` to resolve it from the registry.

    An injected backend belongs to its owner: it is never checked out of
    the registry and never closed by this module.
    """
    global _backend
    _backend = backend


def _default_config_manager():
    """The process's own ConfigManager, for callers that injected none."""
    global _fallback_config_manager

    from cogniverse_foundation.config.utils import create_default_config_manager

    with _fallback_config_manager_lock:
        if _fallback_config_manager is None:
            _fallback_config_manager = create_default_config_manager()
        return _fallback_config_manager


def get_backend() -> Backend:
    """Resolve the metadata backend from the registry.

    Resolved on every call. The registry owns the instance's lifetime and
    closes it on eviction, overwrite or clear, so a handle kept across
    requests goes dead and every tenant read then fails until the process
    restarts. The registry's own LRU is the cache: a hit is a dict lookup.

    Callers that use the backend for a whole operation take
    ``metadata_backend()`` instead, which also holds it against eviction.
    """
    if _backend is not None:
        return _backend
    config_manager = _config_manager
    if config_manager is None:
        config_manager = _default_config_manager()

    config = get_config(tenant_id="system", config_manager=config_manager)
    backend_type = config.get("backend_type", "vespa")

    from cogniverse_core.registries.backend_registry import BackendRegistry

    if _schema_loader is None:
        raise RuntimeError(
            "SchemaLoader not initialized. Call set_schema_loader() during app startup."
        )

    # No tenant_id in the backend config: metadata operations span every
    # tenant, and tenant_id is passed explicitly to the schema operations
    # that need it.
    return BackendRegistry.get_instance().get_ingestion_backend(
        backend_type,
        tenant_id="system",
        config={
            "url": config.get("backend_url"),
            "port": config.get("backend_port"),
        },
        config_manager=config_manager,
        schema_loader=_schema_loader,
    )


@contextmanager
def metadata_backend() -> Iterator[Backend]:
    """Yield the metadata backend, held against eviction for the block.

    Eviction, an overwriting ``set`` and ``clear`` all close what they drop,
    and a backend closed mid-operation loses the connection pool the
    operation is running on. A checkout blocks that close until the block
    exits, so a multi-step write cannot tear on a released instance.

    A backend the registry does not hold (injected in tests, built directly)
    checks out nothing and is yielded as-is: it belongs to whoever built it.
    """
    from cogniverse_core.registries.backend_registry import leased_backend

    with leased_backend(get_backend) as instance:
        yield instance


def _tenant_operation_lease(store, tenant_id: str) -> SchemaDeployLease:
    """The lease one create or delete of ``tenant_id`` holds at a time, on
    every process and replica sharing the config store."""
    return SchemaDeployLease(
        store,
        wait_seconds=TENANT_OPERATION_WAIT_S,
        service="tenant_operation_lease",
        config_key=tenant_id,
        purpose=f"Tenant {tenant_id} create or delete",
        heartbeat=True,
    )


@asynccontextmanager
async def _tenant_operation(store, tenant_id: str, done: str) -> AsyncIterator[None]:
    """Hold ``tenant_id`` for one create or delete, so creates and deletes of
    one tenant run one at a time across every process.

    A holder that keeps it for ``TENANT_OPERATION_WAIT_S`` answers 503
    ``tenant_operation_in_progress``; a store that cannot take it answers 503
    ``tenant_operation_unavailable``. ``done`` completes "Tenant X was not ..."
    in those answers.
    """
    lease = _tenant_operation_lease(store, tenant_id)
    try:
        await asyncio.to_thread(lease.acquire)
    except LeaseWaitTimeout as exc:
        raise failure_response(
            503,
            "tenant_operation_in_progress",
            f"Tenant {tenant_id} was not {done}: another create or delete of it "
            "is still running; retry.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    except Exception as exc:
        raise failure_response(
            503,
            "tenant_operation_unavailable",
            f"Tenant {tenant_id} was not {done}: the config store did not "
            "answer; retry.",
            exc,
            tenant_id=tenant_id,
        ) from exc
    try:
        yield
    finally:
        await asyncio.to_thread(lease.release)


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


TENANT_BASE_SCHEMAS: tuple[str, ...] = (
    "video_colpali_smol500_mv_frame",
    MEMORY_BASE_SCHEMA,
    PROVENANCE_BASE_SCHEMA,
)
"""Schemas every tenant gets at registration.

The memory-aware agents ensure ``agent_memories`` and ``provenance`` on the
first request that touches a tenant. Deploying them here means that ensure
finds them and answers from the tenant's own row, instead of the request
building and activating a Vespa application package and reading every
tenant's registry rows and the whole deployment journal to do it — work that
holds the GIL and stops the replica answering anything while it runs.
"""


_TENANT_SCHEMA_DEPLOY_MAX_ATTEMPTS = 5
_TENANT_SCHEMA_DEPLOY_INITIAL_BACKOFF_S = 0.5
_TENANT_SCHEMA_DEPLOY_MAX_BACKOFF_S = 4.0


def _tenant_schema_deploy_retryable(exc: Exception) -> bool:
    """Return True when a tenant schema deploy can safely be retried.

    Transport-layer failures from the HTTP client are retryable, and so is a
    registry revision conflict the error marks ``retryable``: the peer's
    registration it names is authoritative and the next attempt deploys from
    it. The schema registry flattens
    config-server failures before they reach this helper, so message sniffing
    would just paper over a backend contract gap.
    """
    from cogniverse_core.registries.exceptions import (
        RegistryStorageError,
        SchemaConvergenceError,
        SchemaRevisionConflictError,
    )

    node: BaseException | None = exc
    for _ in range(4):
        if node is None:
            break
        if isinstance(node, SchemaRevisionConflictError):
            return node.retryable
        if isinstance(node, (RegistryStorageError, SchemaConvergenceError)):
            return False
        if isinstance(
            node,
            (
                requests_exceptions.ConnectionError,
                requests_exceptions.Timeout,
            ),
        ):
            return True
        node = node.__cause__ or node.__context__
    return False


async def _deploy_tenant_schemas_with_retry(
    backend: Backend, tenant_full_id: str, base_schema_names: list[str]
) -> None:
    """Deploy one tenant package with a short retry for transport failures."""
    backoff_s = _TENANT_SCHEMA_DEPLOY_INITIAL_BACKOFF_S
    for attempt in range(1, _TENANT_SCHEMA_DEPLOY_MAX_ATTEMPTS + 1):
        try:
            await asyncio.to_thread(
                backend.schema_registry.deploy_schemas,
                tenant_id=tenant_full_id,
                base_schema_names=base_schema_names,
            )
            return
        except Exception as exc:
            if (
                attempt == _TENANT_SCHEMA_DEPLOY_MAX_ATTEMPTS
                or not _tenant_schema_deploy_retryable(exc)
            ):
                raise
            logger.warning(
                "Retrying schema deploy for tenant %s base_schemas %s after %s: %s "
                "(attempt %d/%d)",
                tenant_full_id,
                base_schema_names,
                type(exc).__name__,
                exc,
                attempt,
                _TENANT_SCHEMA_DEPLOY_MAX_ATTEMPTS,
            )
            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, _TENANT_SCHEMA_DEPLOY_MAX_BACKOFF_S)


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

        with metadata_backend() as backend:
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
        raise failure_response(
            500,
            "organization_create_failed",
            f"Creating organization {request.org_id} failed; the runtime log "
            "names the cause.",
            e,
            org_id=request.org_id,
        )


@router.get("/organizations", response_model=OrganizationListResponse)
async def list_organizations() -> OrganizationListResponse:
    """
    List all organizations.

    Returns:
        List of all organizations with count
    """
    try:
        with metadata_backend() as backend:
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
        raise failure_response(
            500,
            "organization_list_failed",
            "Listing organizations failed; the runtime log names the cause.",
            e,
        )


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
    with metadata_backend() as backend:
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
        HTTPException 503: A child tenant delete failed; the organization and
            every tenant that is still present are retained for a retry
        HTTPException 502: The organization record's delete did not confirm
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

        with metadata_backend() as backend:
            # Delete all tenants for this org
            tenants = await list_tenants_for_org_internal(org_id)
            deleted_tenants = []
            failed_tenants = []

            for tenant in tenants:
                try:
                    await delete_tenant_internal(tenant.tenant_full_id)
                    deleted_tenants.append(tenant.tenant_full_id)
                except Exception as e:
                    failed_tenants.append(tenant.tenant_full_id)
                    logger.error(
                        f"Failed to delete tenant {tenant.tenant_full_id}: {e}"
                    )

            if failed_tenants:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": (
                            f"Organization {org_id} deletion incomplete; retry the delete"
                        ),
                        "org_id": org_id,
                        "deleted_tenant_ids": deleted_tenants,
                        "failed_tenant_ids": failed_tenants,
                    },
                )

            # delete_metadata_document reports any non-200 as False, including
            # a failed response for a record that is nevertheless gone. Re-read
            # before failing: an absent record means the delete is durable, so
            # only a record still present is unconfirmed. Reporting "deleted"
            # for a surviving record leaves an organization no retry reaches.
            if not bool(
                backend.delete_metadata_document(
                    schema="organization_metadata", doc_id=org_id
                )
            ) and backend.get_metadata_document(
                schema="organization_metadata", doc_id=org_id
            ):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"organization_metadata delete for {org_id} did not "
                        "confirm; retry the delete"
                    ),
                )

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
        raise failure_response(
            500,
            "organization_delete_failed",
            f"Deleting organization {org_id} failed; the runtime log names the cause.",
            e,
            org_id=org_id,
        )


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
        HTTPException 503: The tenant's earlier delete is incomplete and could
            not be finished, or another create or delete of it held it too
            long; retry
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

        config_manager = _config_manager or _default_config_manager()

        async with _tenant_operation(config_manager.store, tenant_full_id, "created"):
            # A delete of this tenant that did not complete is finished first,
            # releasing it on every worker and dropping what it left.
            await _finish_incomplete_delete(config_manager, tenant_full_id)
            with metadata_backend() as backend:
                # Check if tenant already exists
                existing = await get_tenant_internal(tenant_full_id)
                if existing:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Tenant {tenant_full_id} already exists",
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

                # A tenant id deleted earlier is free again once created: its
                # deletion marker would refuse the schema deploys below.
                await asyncio.to_thread(
                    clear_tenant_deleted, config_manager.store, tenant_full_id
                )

                # Deploy schemas for tenant via Backend.
                base_schemas = request.base_schemas or list(TENANT_BASE_SCHEMAS)

                deployed_schemas: list[str] = []
                try:
                    await _deploy_tenant_schemas_with_retry(
                        backend, tenant_full_id, base_schemas
                    )
                    deployed_schemas.extend(base_schemas)

                    # Create tenant only after the schemas are live.
                    tenant = Tenant(
                        tenant_full_id=tenant_full_id,
                        org_id=org_id,
                        tenant_name=tenant_name,
                        created_at=int(time.time() * 1000),
                        created_by=request.created_by,
                        status="active",
                        schemas_deployed=deployed_schemas,
                    )

                    # Store via Backend.
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
                except Exception:
                    # Best-effort rollback keeps the create path from leaving a tenant
                    # with schemas but no metadata, or an auto-created org with no tenant.
                    schema_manager = backend.schema_manager
                    if deployed_schemas and schema_manager is None:
                        logger.error(
                            "Cannot roll back tenant schemas for %s: backend.schema_manager "
                            "is unavailable after deploying %d schema(s)",
                            tenant_full_id,
                            len(deployed_schemas),
                        )
                    elif deployed_schemas:
                        try:
                            await asyncio.to_thread(
                                schema_manager.delete_tenant_schemas, tenant_full_id
                            )
                        except Exception as rollback_exc:
                            logger.error(
                                f"Failed to roll back tenant schemas for {tenant_full_id}: "
                                f"{rollback_exc}"
                            )

                    if org_created:
                        try:
                            await asyncio.to_thread(
                                backend.delete_metadata_document,
                                schema="organization_metadata",
                                doc_id=org_id,
                            )
                        except Exception as rollback_exc:
                            logger.error(
                                f"Failed to roll back organization {org_id} for "
                                f"{tenant_full_id}: {rollback_exc}"
                            )
                    raise

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise failure_response(
            500,
            "tenant_create_failed",
            f"Creating tenant {request.tenant_id} failed; the runtime log names "
            "the cause.",
            e,
            tenant_id=request.tenant_id,
        )


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
        raise failure_response(
            500,
            "tenant_list_failed",
            f"Listing the tenants of organization {org_id} failed; the runtime "
            "log names the cause.",
            e,
            org_id=org_id,
        )


async def list_organizations_internal() -> List[str]:
    """Internal helper to list every org_id known to the backend.

    Lives next to ``list_tenants_for_org_internal`` so callers that need
    a global tenant sweep (e.g. the daily-cleanup CronWorkflow) can
    enumerate without going through the FastAPI HTTPException-raising
    route. Raises 503 on a backend outage rather than returning [] — an
    empty list read as "no orgs" let the cleanup cron report success while
    doing nothing (memories never expire). A malformed org document is
    skipped by the ``org_id`` filter, not swallowed as a whole-sweep empty.
    """
    with metadata_backend() as backend:
        try:
            documents = backend.query_metadata_documents(
                schema="organization_metadata",
                yql="select * from organization_metadata where true",
                hits=400,
            )
        except Exception as e:
            logger.error(f"Failed to list organizations: {e}")
            raise HTTPException(
                status_code=503, detail="Organization registry temporarily unavailable"
            )
        return [fields["org_id"] for fields in documents if fields.get("org_id")]


async def list_tenants_for_org_internal(org_id: str) -> List[Tenant]:
    """Internal helper to list tenants.

    Raises 503 on a backend outage rather than returning [] — an empty list
    read as "no tenants" let the per-org cleanup cron report success while
    processing nothing.
    """
    with metadata_backend() as backend:
        try:
            # Query tenants for this org using term matching in userQuery
            documents = backend.query_metadata_documents(
                schema="tenant_metadata",
                yql="select * from tenant_metadata where userQuery()",
                query=f"org_id:{org_id}",
                hits=400,
            )
        except Exception as e:
            logger.error(f"Failed to list tenants for {org_id}: {e}")
            raise HTTPException(
                status_code=503, detail="Tenant registry temporarily unavailable"
            )

        tenants = []
        for fields in documents:
            tenants.append(
                Tenant(
                    tenant_full_id=fields.get("tenant_full_id"),
                    org_id=fields.get("org_id"),
                    tenant_name=fields.get("tenant_name"),
                    created_at=fields.get("created_at"),
                    created_by=fields.get("created_by"),
                    status=fields.get("status", "active"),
                    schemas_deployed=fields.get("schemas_deployed", []),
                )
            )

        return tenants


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

    canonical = canonical_tenant_id(tenant_full_id)
    try:
        # Resolving the backend reads the system config, so an outage surfaces
        # here too — inside the same contract as the read it serves.
        with metadata_backend() as backend:
            # Blocking Vespa GET — run off the event loop; this sits under
            # assert_tenant_exists on every search/ingestion/graph request.
            fields = await asyncio.to_thread(
                backend.get_metadata_document,
                schema="tenant_metadata",
                doc_id=canonical,
            )
    except HTTPException:
        raise
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


def _tier_config_manager():
    """The ConfigManager the tier routes read and write through."""
    return _config_manager if _config_manager is not None else _default_config_manager()


async def _assert_tenant_exists(canonical: str) -> None:
    if await get_tenant_internal(canonical) is None:
        raise HTTPException(status_code=404, detail=f"Tenant {canonical} not found")


@router.get("/tenants/{tenant_full_id}/tier", response_model=TenantTier)
async def get_tenant_tier(tenant_full_id: str) -> TenantTier:
    """The tenant's semantic-router tier.

    A tenant that has never been given one reads as ``DEFAULT_ROUTER_TIER``:
    absence is the default, not an error.

    Raises:
        HTTPException 404: Tenant not found
        HTTPException 503: Tenant registry or config store unavailable
    """
    from cogniverse_foundation.config.tenant_tiers import read_tenant_tier

    canonical = canonical_tenant_id(tenant_full_id)
    await _assert_tenant_exists(canonical)
    try:
        tier = await asyncio.to_thread(
            read_tenant_tier, _tier_config_manager(), canonical
        )
    except Exception as e:
        logger.error(f"Tier read failed for {canonical}: {e}")
        raise HTTPException(
            status_code=503, detail="Tenant tier store temporarily unavailable"
        )
    return TenantTier(tenant_id=canonical, tier=tier)


@router.put("/tenants/{tenant_full_id}/tier", response_model=TenantTier)
async def set_tenant_tier_route(
    tenant_full_id: str, request: SetTenantTierRequest
) -> TenantTier:
    """Set the tenant's semantic-router tier.

    The tier selects which routing decisions the request can match, so a value
    outside ``ROUTER_TIERS`` is refused rather than stored: it would match no
    decision and fall through to the default model.

    Raises:
        HTTPException 404: Tenant not found
        HTTPException 422: Tier outside ROUTER_TIERS
        HTTPException 503: Tenant registry or config store unavailable
    """
    from cogniverse_foundation.config.tenant_tiers import (
        set_tenant_tier,
        validate_router_tier,
    )

    canonical = canonical_tenant_id(tenant_full_id)
    try:
        validate_router_tier(request.tier)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    await _assert_tenant_exists(canonical)
    try:
        tier = await asyncio.to_thread(
            set_tenant_tier, _tier_config_manager(), canonical, request.tier
        )
    except Exception as e:
        logger.error(f"Tier write failed for {canonical}: {e}")
        raise HTTPException(
            status_code=503, detail="Tenant tier store temporarily unavailable"
        )
    logger.info(f"Set router tier for {canonical} to {tier}")
    return TenantTier(tenant_id=canonical, tier=tier)


@router.delete("/tenants/{tenant_full_id}")
async def delete_tenant(tenant_full_id: str) -> Dict:
    """
    Delete tenant and its schemas.

    WARNING: This removes all data for the tenant!

    The tenant is marked deleted before anything is dropped, so from the
    first step on every runtime process refuses its memory writes and schema
    deploys, and every worker process releases what it holds for it before
    the schemas go. The marker stays until the tenant is created again.
    Creates and deletes of one tenant run one at a time, and a delete removes
    the tenant it found when it arrived.

    Args:
        tenant_full_id: Full tenant ID (org:tenant)

    Returns:
        Deletion summary, with the workers that released the tenant

    Raises:
        HTTPException 404: Tenant not found
        HTTPException 503: The marker could not be written, not every worker
            confirmed it released the tenant, or another create or delete of
            it held it too long; retry the delete
        HTTPException 500: Deletion failed
    """
    try:
        result = await delete_tenant_internal(tenant_full_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise failure_response(
            500,
            "tenant_delete_failed",
            f"Deleting tenant {tenant_full_id} failed; the runtime log names the "
            "cause.",
            e,
            tenant_id=tenant_full_id,
        )


def _delete_tenant_state(config_manager, tenant_id: str) -> tuple[list[str], bool]:
    """Delete every config-store row the tenant left once its schemas are
    dropped: its own rows in every scope (registry tombstones, backend
    profiles, pin quotas, signature variants and other overrides), its
    schema deployment intents, its provenance write lease and the drift
    migration's refusals of its schemas.

    The deletion marker, its pending record and the tenant's operation lease
    are kept: the marker until the tenant is created again, the lease because
    a create or delete of the tenant contends through it. Revoked harness keys
    are immutable and stay revoked. Returns the rows found, as
    ``service:key``, and whether every one was deleted. Nothing raises: a row
    that cannot be read or deleted is logged at ERROR naming the tenant and
    the row.
    """
    from cogniverse_core.memory.manager import PROVENANCE_WRITE_LEASE_SERVICE
    from cogniverse_core.registries.schema_deployment_intents import (
        SchemaDeploymentIntents,
    )
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    store = config_manager.store
    found: list[str] = []
    deleted = True

    def failed(row: str, exc: Exception) -> None:
        nonlocal deleted
        deleted = False
        logger.error(
            f"Cannot delete {row} of deleted tenant {tenant_id} "
            f"({type(exc).__name__}: {exc}); the delete stays pending and its "
            "retry, or the next create of the tenant, deletes it"
        )

    def delete(row: str, delete_row) -> None:
        found.append(row)
        try:
            delete_row()
        except Exception as exc:
            failed(row, exc)

    try:
        own = store.list_configs(tenant_id=tenant_id)
    except Exception as exc:
        own = []
        failed("the config rows", exc)
    for entry in own:
        delete(
            f"{entry.service}:{entry.config_key}",
            lambda entry=entry: store.delete_config(
                tenant_id, entry.scope, entry.service, entry.config_key
            ),
        )
    intents = SchemaDeploymentIntents(store)
    try:
        names = intents.tenant_names(tenant_id)
    except Exception as exc:
        names = []
        failed("the schema deployment intents", exc)
    for name in names:
        delete(
            f"schema_deployment_intents:{name}",
            lambda name=name: intents.delete(name),
        )
    try:
        lease = store.get_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SCHEMA,
            PROVENANCE_WRITE_LEASE_SERVICE,
            tenant_id,
        )
    except Exception as exc:
        lease = None
        failed("the provenance write lease", exc)
    if lease is not None:
        delete(
            f"{PROVENANCE_WRITE_LEASE_SERVICE}:{tenant_id}",
            lambda: store.delete_config(
                SYSTEM_TENANT_ID,
                ConfigScope.SCHEMA,
                PROVENANCE_WRITE_LEASE_SERVICE,
                tenant_id,
            ),
        )
    if not delete_tenant_refusals(config_manager, tenant_id):
        deleted = False
    return found, deleted


class TenantRecordRetained(RuntimeError):
    """The ``tenant_metadata`` delete did not confirm and the record is
    still present."""


# Builds the answer for a delete step that failed: (status, error code or
# None for a plain-text detail, message, cause) -> the HTTPException raised.
_StepFailure = Callable[[int, Optional[str], str, BaseException], HTTPException]


def _incarnation(tenant: Optional[Tenant]) -> Optional[tuple]:
    """Which creation of a tenant id ``tenant`` is, by its creation time;
    None for no tenant."""
    return None if tenant is None else (tenant.created_at,)


async def delete_tenant_internal(tenant_full_id: str) -> Dict:
    """Delete a tenant's schemas and metadata.

    Looks up tenant_metadata under the canonical form (POST stored it that
    way) and drops every schema the tenant has in one atomic redeploy via
    ``delete_tenant_schemas`` (registry-known names plus canonical-suffix
    Vespa orphans; refuses when the redeploy would drop an unreconstructable
    peer-tenant schema). The registry APIs canonicalize tenant ids on both
    reads and writes, so the single canonical pass is complete for any input
    form.

    The delete removes the tenant it found when it arrived. It then waits for
    the tenant (``_tenant_operation``); a create or delete that held it
    meanwhile may have deleted that tenant, and created it again. Finding so,
    it answers ``deleted`` with nothing dropped and leaves the tenant as it
    is, or 404 when it found no tenant on arrival.
    """
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    if _config_manager is None:
        raise RuntimeError("Tenant ConfigManager is not configured")
    config_manager = _config_manager
    canonical_tid = canonical_tenant_id(tenant_full_id)
    found = await get_tenant_internal(canonical_tid)

    def failed(status, error, message, exc):
        if error is None:
            return HTTPException(status_code=status, detail=message)
        return failure_response(status, error, message, exc, tenant_id=canonical_tid)

    async with _tenant_operation(config_manager.store, canonical_tid, "deleted"):
        current = await get_tenant_internal(canonical_tid)
        if _incarnation(current) != _incarnation(found):
            if found is None:
                raise HTTPException(
                    status_code=404, detail=f"Tenant {canonical_tid} not found"
                )
            logger.info(
                f"The tenant {canonical_tid} this delete found was deleted by "
                "another request while it waited; nothing dropped"
            )
            return {
                "status": "deleted",
                "tenant_full_id": canonical_tid,
                "schemas_deleted": 0,
                "deleted_schemas": [],
                "organization_deleted": False,
                "workers_released": [],
            }
        result = await _delete_tenant(config_manager, canonical_tid, found, failed)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Tenant {canonical_tid} not found")
    return result


async def _finish_incomplete_delete(config_manager, tenant_full_id: str) -> None:
    """Finish the tenant's delete when one began and did not complete, so a
    create starts from a completed delete.

    The delete's steps run again; each does nothing where the delete already
    did it. A step that fails answers 503 ``tenant_delete_incomplete`` and
    the tenant stays marked deleted.
    """
    if not await asyncio.to_thread(
        tenant_delete_pending, config_manager.store, tenant_full_id
    ):
        return
    message = (
        f"Tenant {tenant_full_id} was not created: its earlier delete is "
        "incomplete and could not be finished, so it stays marked deleted; "
        "retry the create or the delete."
    )

    def unfinished(_status, _error, _message, exc):
        return failure_response(
            503, "tenant_delete_incomplete", message, exc, tenant_id=tenant_full_id
        )

    found = await get_tenant_internal(tenant_full_id)
    logger.info(f"Finishing the incomplete delete of tenant {tenant_full_id}")
    try:
        await _delete_tenant(config_manager, tenant_full_id, found, unfinished)
    except HTTPException:
        raise
    except Exception as exc:
        raise unfinished(503, None, message, exc) from exc


async def _delete_tenant(
    config_manager, canonical_tid: str, tenant: Optional[Tenant], failed: _StepFailure
) -> Optional[Dict]:
    """Every step of a tenant delete, in order, with the tenant held.

    ``tenant`` is its record, or None for a tenant with no record. A step
    that fails raises what ``failed`` builds for it, or its own exception.
    Returns the deletion summary, or None when there was nothing to delete:
    the marker is then cleared and the tenant id stays free.
    """
    if _cluster_events is None:
        raise RuntimeError("Tenant deletes need the cluster events channel wired")
    store = config_manager.store

    # Mark first: from here every process refuses the tenant's memory writes
    # and schema deploys, so nothing the delete drops below can be recreated.
    try:
        await asyncio.to_thread(mark_tenant_deleted, store, canonical_tid)
    except Exception as exc:
        raise failed(
            503,
            "tenant_delete_marker_unavailable",
            f"Tenant {canonical_tid} was not deleted: the deletion marker store "
            "did not answer; retry the delete.",
            exc,
        ) from exc
    # Every worker releases what it holds for the tenant (cached agents, warm
    # memory managers, queued memory writes) before anything is dropped. A
    # worker that cannot confirm it fails the delete: the tenant stays marked,
    # so its writes stay refused everywhere, and a retry completes the delete.
    try:
        released = await _cluster_events.publish(
            "tenant_deleted",
            {"tenant_id": canonical_tid},
            timeout_s=TENANT_DELETE_ACK_TIMEOUT_S,
        )
    except ClusterEventError as exc:
        raise failed(
            503,
            "tenant_delete_incomplete",
            f"Tenant {canonical_tid} is marked deleted and its writes are "
            "refused, but not every runtime worker released it; retry the "
            "delete.",
            exc,
        ) from exc

    with metadata_backend() as backend:
        schema_manager = backend.schema_manager

        # One atomic redeploy drops every schema the tenant has —
        # delete_tenant_schemas unions registry-known names with canonical-suffix
        # Vespa orphans itself. The per-schema loop this replaces did one
        # multi-minute redeploy per schema. Off the loop: run inline it blocks
        # every other request, /health included. Peer-orphan refusals and
        # listing failures propagate — the tenant record stays and the delete
        # is retryable.
        deleted_schemas: list = list(
            await asyncio.to_thread(schema_manager.delete_tenant_schemas, canonical_tid)
        )
        # Its schemas are gone, so is everything the config store holds for it.
        state_rows, state_deleted = await asyncio.to_thread(
            _delete_tenant_state, config_manager, canonical_tid
        )

        # Allow schema-only orphans (no tenant_metadata record) to be cleaned
        # up — they're created by /ingestion/upload auto-deploy bypassing
        # tenant create, and accumulate every test run without this branch.
        if not tenant and not deleted_schemas and not state_rows:
            # Nothing existed to delete: the tenant id stays free to use.
            await asyncio.to_thread(clear_tenant_deleted, store, canonical_tid)
            return None

        try:
            await asyncio.to_thread(HarnessKeyStore(store).revoke_tenant, canonical_tid)
        except ConfigStoreUnavailableError as exc:
            raise failed(
                503,
                "harness_key_store_unavailable",
                f"The harness key store did not answer while deleting tenant "
                f"{canonical_tid}; the tenant is retained, retry the delete.",
                exc,
            ) from exc

        if tenant:
            # delete_metadata_document reports a non-200 as False without raising.
            # Claiming "deleted" anyway leaves a routable ghost tenant with zero
            # schemas that is never retried — fail loud instead; the surviving
            # metadata record makes a retry proceed (schema drop is a no-op then).
            metadata_deleted = bool(
                await asyncio.to_thread(
                    backend.delete_metadata_document,
                    schema="tenant_metadata",
                    doc_id=canonical_tid,
                )
            )
            if not metadata_deleted:
                # delete_metadata_document reports any non-200 as False, including
                # a failed response for a record that is nevertheless gone. Re-read
                # before failing: an absent record means the delete is durable, so
                # only a record still present is unconfirmed.
                surviving = await asyncio.to_thread(
                    backend.get_metadata_document,
                    schema="tenant_metadata",
                    doc_id=canonical_tid,
                )
                if surviving:
                    detail = (
                        f"tenant_metadata delete for {canonical_tid} did not "
                        "confirm — tenant record retained, retry the delete"
                    )
                    raise failed(502, None, detail, TenantRecordRetained(detail))

        # Tenant create auto-creates the org; deleting the org's last tenant
        # removes it again so provision/teardown cycles don't accumulate orgs.
        # The tenant itself is already gone here, so a cleanup failure warns and
        # reports organization_deleted false instead of failing the delete.
        organization_deleted = False
        if tenant:
            org_id = canonical_tid.split(":", 1)[0]
            try:
                remaining = await list_tenants_for_org_internal(org_id)
                if not remaining and await get_organization_internal(org_id):
                    # delete_metadata_document reports a non-200 as False without
                    # raising — only claim the deletion when it actually happened.
                    organization_deleted = bool(
                        await asyncio.to_thread(
                            backend.delete_metadata_document,
                            schema="organization_metadata",
                            doc_id=org_id,
                        )
                    )
                    if organization_deleted:
                        logger.info(
                            f"Deleted organization {org_id} (no tenants remain)"
                        )
                    else:
                        logger.warning(
                            f"Organization {org_id} delete reported failure; it may remain"
                        )
            except Exception as e:
                logger.warning(
                    f"Organization cleanup after deleting tenant {canonical_tid} "
                    f"failed (organization {org_id} may remain): {e}"
                )

        # Every step has completed, unless some of the tenant's state could not
        # be deleted: the delete then stays pending, and its retry or the next
        # create of the tenant deletes the rest. A failure to record completion
        # leaves it pending too, so the next create runs its steps again.
        try:
            if state_deleted:
                await asyncio.to_thread(complete_tenant_delete, store, canonical_tid)
        except Exception as exc:
            logger.error(
                f"Tenant {canonical_tid} is deleted, but its delete could not be "
                f"recorded complete ({type(exc).__name__}: {exc}); the next "
                "create of it runs the delete's steps again"
            )

        logger.info(
            f"Deleted tenant {canonical_tid} with {len(deleted_schemas)} schemas"
        )

        return {
            "status": "deleted",
            "tenant_full_id": canonical_tid,
            "schemas_deleted": len(deleted_schemas),
            "deleted_schemas": deleted_schemas,
            "organization_deleted": organization_deleted,
            "workers_released": sorted(released),
        }


# ============================================================================
# Reconciliation
# ============================================================================


def _known_base_schemas() -> tuple[str, ...]:
    """Base schema names a tenant suffix can be stripped back to.

    Derived from the shipped schema files so a newly added base is
    recoverable without editing this module.
    """
    if _schema_loader is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "SchemaLoader not initialized — refusing to reconcile orphans "
                "without the shipped base-schema list. Call set_schema_loader() "
                "during app startup."
            ),
        )

    names = tuple(_schema_loader.list_available_schemas())
    if not names:
        raise HTTPException(
            status_code=503,
            detail=(
                "Schema loader reported no shipped schemas — refusing to "
                "reconcile orphans. Every orphan would be reported unrecoverable, "
                "which blocks tenant deletes."
            ),
        )
    return names


_TENANT_SWEEP_HITS = 400

_EMPTY_TENANT_ORPHAN_SELECTION = (
    "No tenant-orphan schemas to remove: every schema registered in Vespa "
    "belongs to a tenant that still has a tenant_metadata record. Refusing "
    "an empty selection."
)


def _reconcile_unavailable(message: str, exc: Exception):
    """503 for an orphan reconciliation that cannot read the state it needs."""
    return failure_response(503, "reconcile_unavailable", message, exc)


def _live_tenant_ids(backend: Backend) -> set:
    """Canonical ids of every tenant that still has a tenant_metadata record.

    Raises 503 on failed reads or truncated results; an empty successful
    read represents a registry with no tenants.
    """
    try:
        documents = backend.query_metadata_documents(
            schema="tenant_metadata",
            yql="select * from tenant_metadata where true",
            hits=_TENANT_SWEEP_HITS,
        )
    except Exception as exc:
        raise _reconcile_unavailable(
            "Cannot read the tenant registry; refusing to reconcile orphans "
            "because every schema would read as a tenant-orphan.",
            exc,
        ) from exc

    if len(documents) >= _TENANT_SWEEP_HITS:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Tenant registry returned {len(documents)} rows, at or above "
                f"the {_TENANT_SWEEP_HITS}-row page limit; refusing to reconcile "
                "orphans because a truncated tenant list marks live tenants as "
                "orphans"
            ),
        )

    if any(not fields.get("tenant_full_id") for fields in documents):
        raise HTTPException(
            status_code=503,
            detail=(
                "Tenant registry contains a row without tenant_full_id; "
                "refusing to reconcile orphans"
            ),
        )
    return {canonical_tenant_id(fields["tenant_full_id"]) for fields in documents}


def _remove_tenant_orphans(tenants: list, expected_schemas: list) -> list:
    """Drop every schema owned by a tenant with no tenant_metadata record.

    One redeploy covers all named tenants, then the deployed set is read
    back: a target still present means the drop did not take, which must
    not report success.
    """
    if not tenants:
        raise HTTPException(status_code=409, detail=_EMPTY_TENANT_ORPHAN_SELECTION)

    with metadata_backend() as backend:
        schema_manager = backend.schema_manager
        dropped = sorted(schema_manager.delete_tenant_schemas_bulk(list(tenants)))
        still_deployed = set(
            schema_manager.list_deployed_document_types(raise_on_failure=True)
        )
        survivors = sorted(set(expected_schemas) & still_deployed)
        if survivors:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Tenant-orphan removal did not take for {survivors}; they "
                    "are still deployed after the redeploy"
                ),
            )
    return dropped


def _phantom_document_type() -> str:
    """The document type pyvespa adds, named after the application, to a
    package deployed with no schemas; nothing registers it."""
    config_manager = _config_manager or _default_config_manager()
    return config_manager.get_system_config().application_name


def _list_orphan_schemas(include_document_counts: bool = False) -> Dict[str, list]:
    """Diff Vespa-deployed schemas against the registry and the tenant set.

    Reports two independent orphan classes. ``orphan_schemas`` are
    Vespa-only names with no registry record, grouped into
    ``orphan_tenants`` by stripping known base prefixes; names whose base
    is not a shipped schema go to ``unrecovered_schemas``. A schema whose
    activation is in flight in another process is live-but-unregistered for
    the whole convergence wait and is never an orphan.

    ``tenant_orphan_schemas`` are deployed AND registered, but their
    registry row names a tenant with no tenant_metadata record, so the
    registry diff alone cannot see them: they survive every deploy and cost
    a schema in each one. Their owner comes from the registry row, not from
    stripping the name.

    A tenant registry that reads successfully but holds nothing is a real
    state, not a failed read: schema auto-deploy paths create schemas
    without creating a tenant, so a cluster can hold only orphans, and that
    is exactly the cluster reconciliation exists to clean. The reads that
    CAN fabricate orphans -- an outage, a truncated page -- raise instead
    (``_live_tenant_ids``).
    """
    with metadata_backend() as backend:
        schema_manager = backend.schema_manager
        schema_registry = schema_manager._schema_registry

        try:
            deployed = set(
                schema_manager.list_deployed_document_types(raise_on_failure=True)
            )
        except Exception as exc:
            raise _reconcile_unavailable(
                "Cannot enumerate deployed schemas during reconciliation.", exc
            ) from exc
        try:
            # Strict: a cached view could list a peer's newly registered
            # schema as an orphan.
            registry_infos = list(schema_registry._get_all_schemas(strict=True) or [])
        except Exception as exc:
            raise _reconcile_unavailable(
                "Cannot read the schema registry during reconciliation.", exc
            ) from exc
        registered = {info.full_schema_name for info in registry_infos}
        try:
            phantom = _phantom_document_type()
        except Exception as exc:
            raise _reconcile_unavailable(
                "Cannot read the system config naming the application during "
                "reconciliation.",
                exc,
            ) from exc

        # Safety guard: if the registry loaded EMPTY while Vespa has non-protected
        # schemas deployed, the registry almost certainly failed to load from
        # storage (a cold pod whose data-plane read failed while the config server
        # answered). Reconciling here would report EVERY tenant's schema as an
        # orphan and the dry_run=false path would bulk-delete them all. Refuse
        # loudly instead of mass-deleting on an unconfirmed registry. The
        # phantom application type is never registered, so it alone proves
        # nothing about the read.
        non_protected_deployed = (
            deployed - schema_manager._PROTECTED_SCHEMAS - {phantom}
        )
        if not registered and non_protected_deployed:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Schema registry is empty while Vespa has deployed schemas — "
                    "refusing to reconcile orphans (this would delete every tenant's "
                    "schema). The registry likely failed to load from storage; retry "
                    "once it is reachable."
                ),
            )

        try:
            reserved = set(schema_registry.reserved_schemas(deployed))
        except RegistryStorageError as exc:
            raise _reconcile_unavailable(
                "Cannot read schema deployment intents; refusing to reconcile "
                "orphans because a mid-deploy schema would be indistinguishable "
                "from an orphan.",
                exc,
            ) from exc

        orphans = sorted(
            deployed - registered - reserved - schema_manager._PROTECTED_SCHEMAS
        )

        orphan_tenants: set = set()
        unrecovered: list = []
        # Longest base first: with first-match-wins, "document_text" would strip a
        # "document_text_semantic_<tid>" orphan to "semantic_<tid>" — a bogus
        # tenant token.
        bases_longest_first = sorted(_known_base_schemas(), key=len, reverse=True)
        for orphan in orphans:
            for base in bases_longest_first:
                prefix = f"{base}_"
                if orphan.startswith(prefix):
                    orphan_tenants.add(orphan[len(prefix) :])
                    break
            else:
                unrecovered.append(orphan)
        live_tenants = _live_tenant_ids(backend)
        tenant_orphans: Dict[str, list] = {}
        for info in registry_infos:
            if info.full_schema_name not in deployed:
                continue
            owner = canonical_tenant_id(info.tenant_id)
            if owner == SYSTEM_TENANT_ID or owner in live_tenants:
                continue
            tenant_orphans.setdefault(owner, []).append(info.full_schema_name)

        tenant_orphan_schemas = sorted(
            name for names in tenant_orphans.values() for name in names
        )
        details = []
        if include_document_counts:
            for tenant, names in sorted(tenant_orphans.items()):
                for name in sorted(names):
                    try:
                        [count_row] = backend.query_metadata_documents(
                            schema=name,
                            yql=f"select * from {name} where true limit 0 | all(output(count()))",
                            hits=0,
                        )
                        count = count_row["count()"]
                        if type(count) is not int or count < 0:
                            raise ValueError(f"Invalid document count: {count!r}")
                    except Exception as exc:
                        raise _reconcile_unavailable(
                            f"Cannot count documents in orphan schema {name}.", exc
                        ) from exc
                    details.append(
                        {
                            "schema": name,
                            "tenant": tenant,
                            "tenant_exists": tenant in live_tenants,
                            "document_count": count,
                        }
                    )
        return {
            "orphan_details": sorted(details, key=lambda row: row["schema"]),
            "orphan_schemas": orphans,
            "orphan_tenants": sorted(orphan_tenants),
            "unrecovered_schemas": unrecovered,
            "tenant_orphan_schemas": tenant_orphan_schemas,
            "tenant_orphan_tenants": sorted(tenant_orphans),
        }


@router.post("/reconcile-orphans")
async def reconcile_orphans(
    dry_run: Annotated[
        bool, Query(description="Report orphans without modifying state.")
    ] = True,
    remove_tenant_orphans: Annotated[
        bool,
        Query(description="With dry_run=false, remove schemas whose tenant is gone."),
    ] = False,
    include_document_counts: Annotated[
        bool,
        Query(
            description="Include owning tenant and document counts for tenant orphans."
        ),
    ] = False,
) -> Dict:
    """Report both orphan classes, optionally drop them.

    ``dry_run=true`` returns the diff for operator review and changes
    nothing. ``dry_run=false`` drops the registry-orphans in one redeploy
    (required because an individual tenant delete refuses while a
    peer-tenant unreconstructable orphan exists). Adding
    ``remove_tenant_orphans=true`` then drops the tenant-orphans in a
    second redeploy and reads the deployed set back.

    Registry-orphans go first: they are unreconstructable survivors, and a
    redeploy that has to carry them refuses.
    """
    diff = await asyncio.to_thread(_list_orphan_schemas, include_document_counts)

    deleted: list = []
    if not dry_run and diff["orphan_schemas"]:
        with metadata_backend() as backend:
            deleted = await asyncio.to_thread(
                backend.schema_manager.delete_orphan_schemas, diff["orphan_schemas"]
            )

    tenant_orphans_deleted: list = []
    if not dry_run and remove_tenant_orphans:
        logger.info(
            f"Removing schemas of {len(diff['tenant_orphan_tenants'])} tenant(s) "
            f"with no tenant_metadata record: {diff['tenant_orphan_tenants']}"
        )
        tenant_orphans_deleted = await asyncio.to_thread(
            _remove_tenant_orphans,
            diff["tenant_orphan_tenants"],
            diff["tenant_orphan_schemas"],
        )

    if diff["tenant_orphan_schemas"] and not tenant_orphans_deleted:
        logger.warning(
            f"{len(diff['tenant_orphan_schemas'])} schema(s) belong to tenants "
            f"with no tenant_metadata record and are redeployed with every "
            f"application package: {diff['tenant_orphan_schemas']}. Clear them "
            f"with POST /admin/reconcile-orphans"
            f"?dry_run=false&remove_tenant_orphans=true"
        )

    return {
        "dry_run": dry_run,
        "deleted": deleted,
        "tenant_orphans_deleted": tenant_orphans_deleted,
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
    set_config_manager(config_manager)
    config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)
    port = config.get("tenant_manager_port", 9000)

    logger.info(f"Starting Tenant Management API on port {port}")
    logger.info(f"Organization API: http://localhost:{port}/admin/organizations")
    logger.info(f"Tenant API: http://localhost:{port}/admin/tenants")

    uvicorn.run(app, host="0.0.0.0", port=port)
