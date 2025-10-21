"""
Vespa Metadata Schemas

Single source of truth for organization and tenant metadata schemas.
These schemas store multi-tenant management data in Vespa.

Used by:
- TenantSchemaManager: For schema deployment
- VespaTestManager: For test isolation
- Integration tests: For test setup

DO NOT duplicate these definitions elsewhere - always import from this module.
"""

from vespa.package import Document, Field, Schema


def create_organization_metadata_schema() -> Schema:
    """
    Create organization metadata schema.

    Stores organization-level information for multi-tenant management.

    Fields:
        org_id: Organization identifier (indexed, fast-search)
        org_name: Organization name (indexed)
        created_at: Creation timestamp (Unix epoch)
        created_by: Creator identifier
        status: Organization status (active/inactive/suspended)
        tenant_count: Number of tenants in this organization

    Returns:
        Schema object ready for deployment
    """
    return Schema(
        name="organization_metadata",
        document=Document(
            fields=[
                Field(
                    name="org_id",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(name="org_name", type="string", indexing=["summary", "index"]),
                Field(
                    name="created_at", type="long", indexing=["summary", "attribute"]
                ),
                Field(
                    name="created_by", type="string", indexing=["summary", "attribute"]
                ),
                Field(
                    name="status",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="tenant_count", type="int", indexing=["summary", "attribute"]
                ),
            ]
        ),
    )


def create_tenant_metadata_schema() -> Schema:
    """
    Create tenant metadata schema.

    Stores tenant-level information for multi-tenant management.

    Fields:
        tenant_full_id: Full tenant identifier (org:tenant format)
        org_id: Parent organization identifier (indexed, fast-search)
        tenant_name: Tenant name
        created_at: Creation timestamp (Unix epoch)
        created_by: Creator identifier
        status: Tenant status (active/inactive/suspended)
        schemas_deployed: List of deployed schema names for this tenant

    Returns:
        Schema object ready for deployment
    """
    return Schema(
        name="tenant_metadata",
        document=Document(
            fields=[
                Field(
                    name="tenant_full_id",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="org_id",
                    type="string",
                    indexing=["summary", "index", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="tenant_name", type="string", indexing=["summary", "attribute"]
                ),
                Field(
                    name="created_at", type="long", indexing=["summary", "attribute"]
                ),
                Field(
                    name="created_by", type="string", indexing=["summary", "attribute"]
                ),
                Field(
                    name="status",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="schemas_deployed",
                    type="array<string>",
                    indexing=["summary", "attribute"],
                ),
            ]
        ),
    )


def add_metadata_schemas_to_package(app_package) -> None:
    """
    Add both metadata schemas to an ApplicationPackage.

    This is the recommended way to include metadata schemas in deployments.

    Args:
        app_package: ApplicationPackage instance to add schemas to

    Example:
        >>> from vespa.package import ApplicationPackage
        >>> app_package = ApplicationPackage(name="videosearch")
        >>> add_metadata_schemas_to_package(app_package)
        >>> # Now app_package has organization_metadata and tenant_metadata schemas
    """
    app_package.add_schema(create_organization_metadata_schema())
    app_package.add_schema(create_tenant_metadata_schema())
