"""
Data models for organization and tenant management.

These models define the structure for multi-tenant organization hierarchy.
Format: org:tenant (e.g., "acme:production", "startup:dev")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Organization:
    """
    Organization metadata.

    An organization can have multiple tenants (e.g., dev, staging, production).
    Format: org_id is simple identifier like "acme", "startup"
    """

    org_id: str  # e.g., "acme"
    org_name: str  # e.g., "Acme Corporation"
    created_at: int  # Unix timestamp (milliseconds)
    created_by: str  # User/service that created org
    status: str = "active"  # active | suspended | deleted
    tenant_count: int = 0  # Number of tenants in this org
    config: Optional[Dict] = field(default_factory=dict)  # Optional config


@dataclass
class Tenant:
    """
    Tenant metadata.

    Tenants are the basic isolation unit. Each tenant has its own schemas and data.
    Format: tenant_full_id is "org:tenant" (e.g., "acme:production")
    """

    tenant_full_id: str  # e.g., "acme:production"
    org_id: str  # e.g., "acme"
    tenant_name: str  # e.g., "production"
    created_at: int  # Unix timestamp (milliseconds)
    created_by: str  # User/service that created tenant
    status: str = "active"  # active | suspended | deleted
    schemas_deployed: List[str] = field(
        default_factory=list
    )  # Vespa schemas deployed for this tenant
    config: Optional[Dict] = field(default_factory=dict)  # Optional config


@dataclass
class CreateOrganizationRequest:
    """Request to create a new organization"""

    org_id: str  # Organization identifier (alphanumeric + underscore)
    org_name: str  # Display name
    created_by: str  # Who is creating this org


@dataclass
class CreateTenantRequest:
    """Request to create a new tenant"""

    tenant_id: str  # Can be "org:tenant" or just "tenant" (org_id required separately)
    created_by: str  # Who is creating this tenant
    org_id: Optional[str] = None  # Required if tenant_id doesn't include org
    base_schemas: Optional[List[str]] = (
        None  # Schemas to deploy (default: all standard)
    )


@dataclass
class TenantListResponse:
    """Response for listing tenants"""

    tenants: List[Tenant]
    total_count: int
    org_id: str


@dataclass
class OrganizationListResponse:
    """Response for listing organizations"""

    organizations: List[Organization]
    total_count: int
