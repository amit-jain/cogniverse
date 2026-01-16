"""
Vespa Metadata Schemas

Single source of truth for organization and tenant metadata schemas.
These schemas store multi-tenant management data in Vespa.

Used by:
- SchemaRegistry: For schema deployment
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
                    indexing=["summary", "attribute"],
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


def create_config_metadata_schema() -> Schema:
    """
    Create config_metadata schema for VespaConfigStore.

    Stores configuration key-value pairs for multi-tenant configuration management.

    Fields:
        config_id: Unique config identifier (indexed for exact matching)
        tenant_id: Tenant identifier (indexed for exact matching)
        scope: Configuration scope (indexed for exact matching)
        service: Service name (indexed for exact matching)
        config_key: Configuration key name
        config_value: Configuration value (JSON string)
        version: Config version for optimistic locking (indexed, fast-search)
        created_at: Creation timestamp
        updated_at: Last update timestamp

    Returns:
        Schema object ready for deployment
    """
    return Schema(
        name="config_metadata",
        document=Document(
            fields=[
                Field(
                    name="config_id",
                    type="string",
                    indexing=["summary", "index", "attribute"],
                    attribute=["fast-search"],
                    match=["word"],
                ),
                Field(
                    name="tenant_id",
                    type="string",
                    indexing=["summary", "index", "attribute"],
                    attribute=["fast-search"],
                    match=["word"],
                ),
                Field(
                    name="scope",
                    type="string",
                    indexing=["summary", "index", "attribute"],
                    attribute=["fast-search"],
                    match=["word"],
                ),
                Field(
                    name="service",
                    type="string",
                    indexing=["summary", "index", "attribute"],
                    attribute=["fast-search"],
                    match=["word"],
                ),
                Field(
                    name="config_key",
                    type="string",
                    indexing=["summary", "index", "attribute"],
                    match=["word"],
                ),
                Field(
                    name="config_value",
                    type="string",
                    indexing=["summary"],
                ),
                Field(
                    name="version",
                    type="int",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="created_at",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="updated_at",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
            ]
        ),
    )


def create_adapter_registry_schema() -> Schema:
    """
    Create adapter_registry schema for Model Registry.

    Stores metadata for trained LoRA adapters, enabling versioning, activation,
    and deployment to agents.

    Fields:
        adapter_id: Unique adapter identifier (fast-search)
        tenant_id: Tenant identifier for multi-tenancy (fast-search)
        name: Human-readable adapter name
        version: Semantic version string
        base_model: Base model the adapter was trained on
        model_type: Type of model - "llm" or "embedding" (fast-search)
        agent_type: Target agent type (fast-search)
        training_method: Training method - "sft", "dpo", or "embedding"
        adapter_path: Filesystem path to adapter weights
        status: Adapter status - "active", "inactive", "deprecated" (fast-search)
        is_active: Boolean (0/1) for quick active adapter lookup (fast-search)
        metrics: JSON string with training metrics
        training_config: JSON string with training configuration
        experiment_run_id: MLflow experiment run ID
        created_at: Creation timestamp
        updated_at: Last update timestamp

    Returns:
        Schema object ready for deployment
    """
    return Schema(
        name="adapter_registry",
        document=Document(
            fields=[
                Field(
                    name="adapter_id",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="tenant_id",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="name",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="version",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="base_model",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="model_type",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="agent_type",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="training_method",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="adapter_path",
                    type="string",
                    indexing=["summary"],
                ),
                Field(
                    name="adapter_uri",
                    type="string",
                    indexing=["summary"],
                ),
                Field(
                    name="status",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="is_active",
                    type="int",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="metrics",
                    type="string",
                    indexing=["summary"],
                ),
                Field(
                    name="training_config",
                    type="string",
                    indexing=["summary"],
                ),
                Field(
                    name="experiment_run_id",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="created_at",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="updated_at",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
            ]
        ),
    )


def add_metadata_schemas_to_package(app_package) -> None:
    """
    Add all metadata schemas to an ApplicationPackage.

    This is the recommended way to include metadata schemas in deployments.
    Includes organization, tenant, config metadata, and adapter registry schemas.

    Args:
        app_package: ApplicationPackage instance to add schemas to

    Example:
        >>> from vespa.package import ApplicationPackage
        >>> app_package = ApplicationPackage(name="cogniverse")
        >>> add_metadata_schemas_to_package(app_package)
        >>> # Now app_package has organization_metadata, tenant_metadata,
        >>> # config_metadata, and adapter_registry schemas
    """
    app_package.add_schema(create_organization_metadata_schema())
    app_package.add_schema(create_tenant_metadata_schema())
    app_package.add_schema(create_config_metadata_schema())
    app_package.add_schema(create_adapter_registry_schema())
