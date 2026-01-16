"""
Vespa Metadata Schemas

Single source of truth for metadata schemas loaded from JSON files.
These schemas store multi-tenant management data in Vespa.

Used by:
- SchemaRegistry: For schema deployment
- VespaTestManager: For test isolation
- Integration tests: For test setup

All schemas are defined in configs/schemas/*_schema.json files.
DO NOT duplicate schema definitions - always use the JSON files as source of truth.
"""

import logging
from pathlib import Path
from typing import Optional

from vespa.package import Schema

from cogniverse_vespa.json_schema_parser import JsonSchemaParser

logger = logging.getLogger(__name__)

# Default path to schema files - can be overridden for testing
_SCHEMAS_DIR: Optional[Path] = None


def get_schemas_dir() -> Path:
    """
    Get the path to the schemas directory.

    Returns the configured path or auto-detects based on common locations.

    Returns:
        Path to configs/schemas directory
    """
    global _SCHEMAS_DIR
    if _SCHEMAS_DIR is not None:
        return _SCHEMAS_DIR

    # Try to find configs/schemas relative to common locations
    possible_paths = [
        Path("configs/schemas"),  # Current directory
        Path(__file__).parent.parent.parent.parent.parent / "configs" / "schemas",  # From libs/vespa/cogniverse_vespa/
    ]

    for path in possible_paths:
        if path.exists():
            _SCHEMAS_DIR = path.resolve()
            logger.debug(f"Found schemas directory at: {_SCHEMAS_DIR}")
            return _SCHEMAS_DIR

    # Default to configs/schemas (will be created if needed)
    _SCHEMAS_DIR = Path("configs/schemas").resolve()
    return _SCHEMAS_DIR


def set_schemas_dir(path: Path) -> None:
    """
    Set the path to the schemas directory.

    Useful for testing or when running from non-standard locations.

    Args:
        path: Path to configs/schemas directory
    """
    global _SCHEMAS_DIR
    _SCHEMAS_DIR = path.resolve()
    logger.info(f"Schemas directory set to: {_SCHEMAS_DIR}")


def _load_schema(schema_name: str) -> Schema:
    """
    Load a schema from its JSON file.

    Args:
        schema_name: Name of the schema (without _schema.json suffix)

    Returns:
        Schema object loaded from JSON file

    Raises:
        RuntimeError: If schema file cannot be loaded
    """
    schemas_dir = get_schemas_dir()
    schema_file = schemas_dir / f"{schema_name}_schema.json"

    if not schema_file.exists():
        raise RuntimeError(
            f"Schema file not found: {schema_file}. "
            f"Ensure the schema JSON file exists in {schemas_dir}"
        )

    parser = JsonSchemaParser()
    return parser.load_schema_from_json_file(str(schema_file))


def create_organization_metadata_schema() -> Schema:
    """
    Create organization metadata schema.

    Loads from configs/schemas/organization_metadata_schema.json.

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
    return _load_schema("organization_metadata")


def create_tenant_metadata_schema() -> Schema:
    """
    Create tenant metadata schema.

    Loads from configs/schemas/tenant_metadata_schema.json.

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
    return _load_schema("tenant_metadata")


def create_config_metadata_schema() -> Schema:
    """
    Create config_metadata schema for VespaConfigStore.

    Loads from configs/schemas/config_metadata_schema.json.

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
    return _load_schema("config_metadata")


def create_adapter_registry_schema() -> Schema:
    """
    Create adapter_registry schema for Model Registry.

    Loads from configs/schemas/adapter_registry_schema.json.

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
        adapter_uri: Cloud storage URI (s3://, gs://, hf://, file://)
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
    return _load_schema("adapter_registry")


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
