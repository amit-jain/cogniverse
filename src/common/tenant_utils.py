"""
Tenant utilities for handling org:tenant identifiers and storage paths.

Supports two formats:
- Simple: "acme" → single-level directory
- Org:tenant: "acme:production" → two-level directory (org/tenant)
"""

from pathlib import Path


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
