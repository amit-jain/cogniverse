"""Tenant utilities for handling org:tenant identifiers and storage paths.

The pure, dependency-free tenant identity helpers now live in
``cogniverse_foundation.common.tenant_utils`` (so foundation config /
telemetry code can use them without importing upward into core). They are
re-exported here unchanged, so every existing
``from cogniverse_core.common.tenant_utils import ...`` keeps working.

``assert_tenant_exists`` and its existence cache stay here because they
reach the tenant registry (a runtime concern), which foundation must not
depend on. So do the tenant deletion markers, which every process reads
before deploying a tenant schema or writing a tenant's memory.
"""

from cogniverse_foundation.common.tenant_utils import (
    SYSTEM_TENANT_ID,
    TEST_TENANT_ID,
    canonical_tenant_id,
    get_tenant_storage_path,
    parse_tenant_id,
    require_tenant_id,
    sanitize_k8s_label_value,
    validate_tenant_id,
)

__all__ = [
    "SYSTEM_TENANT_ID",
    "TEST_TENANT_ID",
    "parse_tenant_id",
    "canonical_tenant_id",
    "get_tenant_storage_path",
    "validate_tenant_id",
    "require_tenant_id",
    "sanitize_k8s_label_value",
    "invalidate_tenant_exists",
    "assert_tenant_exists",
    "TenantDeletedError",
    "mark_tenant_deleted",
    "clear_tenant_deleted",
    "tenant_is_deleted",
    "raise_if_tenant_deleted",
]


# Tenants confirmed to exist, with the monotonic time of confirmation.
# Existence is effectively permanent (deletion is a rare admin action), so a
# short positive-only cache removes a Vespa GET from every request without
# caching absence — an unknown tenant is re-checked every time, so a freshly
# created tenant is visible immediately.
_TENANT_EXISTS_CACHE: dict = {}
_TENANT_EXISTS_TTL_S = 30.0


def invalidate_tenant_exists(tenant_id: str) -> None:
    """Drop a tenant from the existence cache after deletion.

    Without this, a deleted tenant keeps passing assert_tenant_exists for
    up to the TTL, so search/ingestion requests proceed against schemas
    that are being torn down instead of getting the 404.
    """
    _TENANT_EXISTS_CACHE.pop(canonical_tenant_id(tenant_id), None)


async def assert_tenant_exists(tenant_id: str) -> None:
    """
    Raise HTTPException(404) if tenant_id was never registered.

    Looks up the tenant via TenantManager.get_tenant_internal, which reads
    Vespa's tenant_metadata schema. SYSTEM_TENANT_ID bypasses the check
    (it's a runtime-internal identity that isn't registered as a user
    tenant). Positive results are cached for a short TTL — this check sits
    on every search/ingestion/graph request.

    Lazy imports keep this module free of a FastAPI / runtime dependency.
    """
    if tenant_id == SYSTEM_TENANT_ID:
        return

    import time

    from fastapi import HTTPException

    canonical = canonical_tenant_id(tenant_id)
    confirmed_at = _TENANT_EXISTS_CACHE.get(canonical)
    now = time.monotonic()
    if confirmed_at is not None and now - confirmed_at < _TENANT_EXISTS_TTL_S:
        return

    from cogniverse_runtime.admin.tenant_manager import get_tenant_internal

    tenant = await get_tenant_internal(canonical)
    if tenant is not None:
        _TENANT_EXISTS_CACHE[canonical] = now
        return
    _TENANT_EXISTS_CACHE.pop(canonical, None)
    if tenant is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant '{tenant_id}' not registered",
        )


# A deleted tenant is marked in the config store before any of its schemas are
# dropped, and the marker is read (a point read of one document) before every
# tenant schema deploy and every memory write, so no process can recreate the
# tenant's state after the delete began, whatever it still holds in memory.
TENANT_DELETIONS_SERVICE = "tenant_deletions"
_DELETED = {"deleted": True}


class TenantDeletedError(RuntimeError):
    """A write or schema deploy for a tenant that has been deleted."""

    def __init__(self, tenant_id: str):
        super().__init__(
            f"Tenant '{tenant_id}' has been deleted; its schemas and memories "
            "are not written until the tenant is created again"
        )
        self.tenant_id = tenant_id


def _deletion_coordinates(tenant_id: str):
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    return (
        SYSTEM_TENANT_ID,
        ConfigScope.SYSTEM,
        TENANT_DELETIONS_SERVICE,
        canonical_tenant_id(tenant_id),
    )


def mark_tenant_deleted(store, tenant_id: str) -> None:
    """Durably mark ``tenant_id`` deleted; marking it again is a no-op.

    A store failure raises, with nothing marked.
    """
    store.put_immutable_config(*_deletion_coordinates(tenant_id), dict(_DELETED))


def clear_tenant_deleted(store, tenant_id: str) -> bool:
    """Remove the deletion marker so the tenant can be created again.

    Returns False when the tenant was not marked deleted.
    """
    return store.delete_config(*_deletion_coordinates(tenant_id))


def tenant_is_deleted(store, tenant_id: str) -> bool:
    """Whether ``tenant_id`` is marked deleted, read from the store now.

    A store outage raises: an unreadable marker is never taken for "not
    deleted".
    """
    return store.get_immutable_config(*_deletion_coordinates(tenant_id)) is not None


def raise_if_tenant_deleted(store, tenant_id: str) -> None:
    """Raise :class:`TenantDeletedError` when ``tenant_id`` is marked deleted."""
    if tenant_is_deleted(store, tenant_id):
        raise TenantDeletedError(canonical_tenant_id(tenant_id))
