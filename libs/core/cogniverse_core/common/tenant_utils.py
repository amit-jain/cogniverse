"""Tenant utilities for handling org:tenant identifiers and storage paths.

The pure, dependency-free tenant identity helpers now live in
``cogniverse_foundation.common.tenant_utils`` (so foundation config /
telemetry code can use them without importing upward into core). They are
re-exported here unchanged, so every existing
``from cogniverse_core.common.tenant_utils import ...`` keeps working.

``assert_tenant_exists`` and its existence cache stay here because they
reach the tenant registry (a runtime concern), which foundation must not
depend on.
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
