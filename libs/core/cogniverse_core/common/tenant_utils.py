"""
Tenant utilities for handling org:tenant identifiers and storage paths.

Supports two formats:
- Simple: "acme" → single-level directory
- Org:tenant: "acme:production" → two-level directory (org/tenant)
"""

from pathlib import Path
from typing import Optional

# Reserved cluster identity — used by runtime startup for state that is not
# tenant-specific (SystemConfig lookups, startup telemetry probes, backend
# registry bootstrap). Must NEVER appear in a request body. validate_tenant_id
# rejects names with a leading "__" so user-registered tenants cannot collide.
SYSTEM_TENANT_ID = "__system__"

# Sentinel used by unit/integration test fixtures. tests/conftest.py registers
# it once per session via POST /admin/tenants; tests that need a tenant_id
# use this constant instead of a literal string.
TEST_TENANT_ID = "test:unit"


def parse_tenant_id(tenant_id: str) -> tuple[str, str]:
    """
    Parse tenant_id into org_id and tenant_name.

    Supports two formats:
    - Simple: "acme" → ("acme", "acme")
    - Org:tenant: "acme:production" → ("acme", "production")

    Args:
        tenant_id: Tenant identifier

    Returns:
        Tuple of (org_id, tenant_name)

    Raises:
        ValueError: If tenant_id is empty or has invalid format

    Examples:
        >>> parse_tenant_id("acme")
        ('acme', 'acme')
        >>> parse_tenant_id("acme:production")
        ('acme', 'production')
    """
    if not tenant_id:
        raise ValueError("tenant_id cannot be empty")

    if ":" in tenant_id:
        parts = tenant_id.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid tenant_id format: {tenant_id}. Expected 'org:tenant' with single colon"
            )
        org_id, tenant_name = parts
        if not org_id or not tenant_name:
            raise ValueError(
                f"Invalid tenant_id '{tenant_id}': both org and tenant parts must be non-empty"
            )
        return org_id, tenant_name
    else:
        # Simple format: use tenant_id as both org and tenant
        return tenant_id, tenant_id


def get_tenant_storage_path(base_dir: Path | str, tenant_id: str) -> Path:
    """
    Get tenant-specific storage path with proper org/tenant structure.

    Supports two formats:
    - Simple: "acme" → base_dir/acme/
    - Org:tenant: "acme:production" → base_dir/acme/production/

    Args:
        base_dir: Base storage directory
        tenant_id: Tenant identifier

    Returns:
        Path to tenant-specific storage directory

    Examples:
        >>> get_tenant_storage_path("data/optimization", "acme")
        Path('data/optimization/acme')
        >>> get_tenant_storage_path("data/optimization", "acme:production")
        Path('data/optimization/acme/production')
    """
    base_path = Path(base_dir)
    org_id, tenant_name = parse_tenant_id(tenant_id)

    if org_id == tenant_name:
        # Simple format: single level
        return base_path / org_id
    else:
        # Org:tenant format: two levels
        return base_path / org_id / tenant_name


def validate_tenant_id(tenant_id: str) -> None:
    """
    Validate tenant ID format.

    Raises:
        ValueError: If tenant_id is invalid
    """
    if not tenant_id:
        raise ValueError("tenant_id cannot be empty")

    if not isinstance(tenant_id, str):
        raise ValueError(f"tenant_id must be string, got {type(tenant_id)}")

    # Identifiers starting with "__" are reserved for runtime-internal
    # identities (e.g. SYSTEM_TENANT_ID). Users may not register them.
    if tenant_id.startswith("__"):
        raise ValueError(
            f"Invalid tenant_id '{tenant_id}': identifiers starting with '__' "
            f"are reserved for runtime-internal use"
        )

    # Allow alphanumeric, underscores, hyphens, and colons
    allowed_chars = tenant_id.replace("_", "").replace("-", "").replace(":", "")
    if not allowed_chars.isalnum():
        raise ValueError(
            f"Invalid tenant_id '{tenant_id}': only alphanumeric, underscore, hyphen, and colon allowed"
        )

    # If colon present, validate org:tenant format
    if ":" in tenant_id:
        parts = tenant_id.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid tenant_id format '{tenant_id}': expected 'org:tenant' with single colon"
            )
        org_id, tenant_name = parts
        if not org_id or not tenant_name:
            raise ValueError(
                f"Invalid tenant_id '{tenant_id}': both org and tenant parts must be non-empty"
            )


def require_tenant_id(tenant_id: Optional[str], *, source: str) -> str:
    """
    Enforce that a request/input/context carries an explicit tenant_id.

    Pure ValueError raiser — no framework dependency.  FastAPI routes should
    catch and translate to HTTPException(400); agent code can let the error
    propagate up to the dispatcher which already maps ValueError to a 400.

    Args:
        tenant_id: The value read off the request / input / dispatch context.
        source: Short descriptor of where the id was expected (e.g.
            "SearchRequest", "A2A metadata", "AgentInput"). Appears in the
            error message to help debugging.

    Returns:
        The validated tenant_id (unchanged, for convenient inline use).

    Raises:
        ValueError: If tenant_id is None, empty, or not a string.
    """
    if tenant_id is None or tenant_id == "":
        raise ValueError(
            f"tenant_id is required on {source}. The runtime no longer falls "
            f"back to a bootstrap tenant for user requests — pass tenant_id "
            f"explicitly in the request body or A2A metadata."
        )
    if not isinstance(tenant_id, str):
        raise ValueError(
            f"tenant_id on {source} must be a string, got {type(tenant_id).__name__}"
        )
    return tenant_id


async def assert_tenant_exists(tenant_id: str) -> None:
    """
    Raise HTTPException(404) if tenant_id was never registered.

    Looks up the tenant via TenantManager.get_tenant_internal, which reads
    Vespa's tenant_metadata schema. SYSTEM_TENANT_ID bypasses the check
    (it's a runtime-internal identity that isn't registered as a user
    tenant).

    Lazy imports keep this module free of a FastAPI / runtime dependency.
    """
    if tenant_id == SYSTEM_TENANT_ID:
        return

    from fastapi import HTTPException

    from cogniverse_runtime.admin.tenant_manager import get_tenant_internal

    tenant = await get_tenant_internal(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant '{tenant_id}' not registered",
        )
