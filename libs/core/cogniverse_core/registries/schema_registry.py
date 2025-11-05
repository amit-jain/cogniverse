"""
Schema Registry - Tracks deployed schemas across tenants

Facade over ConfigManager that provides schema-specific operations.
Ensures all schemas are tracked and can be redeployed together.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cogniverse_core.config.manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class SchemaInfo:
    """Information about a deployed schema"""

    tenant_id: str
    base_schema_name: str
    full_schema_name: str
    schema_definition: str
    config: Dict[str, Any]
    deployment_time: str


class SchemaRegistry:
    """
    Registry for tracking deployed schemas.

    Facade over ConfigManager that provides schema-specific operations.
    Ensures all schemas are tracked and can be redeployed together.

    This prevents the critical schema wipeout bug where deploying a new schema
    would inadvertently delete all existing schemas from Vespa.

    Usage:
        registry = get_schema_registry()

        # Register a newly deployed schema
        registry.register_schema(
            tenant_id="test_tenant",
            base_schema_name="video_colpali_smol500_mv_frame",
            full_schema_name="video_colpali_smol500_mv_frame_test_tenant",
            schema_definition=str(schema),
            config={"profile": "video_colpali_smol500_mv_frame"}
        )

        # Get all schemas for a tenant before redeployment
        schemas = registry.get_tenant_schemas("test_tenant")

        # Check if schema already deployed
        exists = registry.schema_exists("test_tenant", "video_colpali_smol500_mv_frame")
    """

    _instance = None
    _lock = None

    def __new__(cls):
        """Singleton pattern with thread safety"""
        if cls._instance is None:
            import threading

            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize SchemaRegistry with in-memory schema tracking"""
        if self._initialized:
            return
        self._config_manager = ConfigManager()
        # In-memory registry of all deployed schemas
        # Key: (tenant_id, base_schema_name), Value: SchemaInfo
        self._schemas: Dict[tuple, SchemaInfo] = {}

        # Load all previously deployed schemas from persistent storage
        self._load_schemas_from_storage()

        self._initialized = True
        logger.info(f"SchemaRegistry initialized with {len(self._schemas)} schemas loaded")

    def _load_schemas_from_storage(self):
        """Load all schemas from ConfigManager into in-memory registry on startup"""
        try:
            # Load all schemas across all tenants
            all_schema_data = self._config_manager.get_deployed_schemas(tenant_id=None)
            for schema_data in all_schema_data:
                tenant_id = schema_data["tenant_id"]
                base_schema_name = schema_data["base_schema_name"]
                key = (tenant_id, base_schema_name)

                self._schemas[key] = SchemaInfo(
                    tenant_id=tenant_id,
                    base_schema_name=base_schema_name,
                    full_schema_name=schema_data["full_schema_name"],
                    schema_definition=schema_data["schema_definition"],
                    config=schema_data.get("config", {}),
                    deployment_time=schema_data["deployment_time"],
                )

            logger.info(f"Loaded {len(self._schemas)} schemas from storage")
        except Exception as e:
            logger.warning(f"Failed to load schemas from storage: {e}. Starting with empty registry.")

    def register_schema(
        self,
        tenant_id: str,
        base_schema_name: str,
        full_schema_name: str,
        schema_definition: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a newly deployed schema.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name (e.g., 'video_colpali_smol500_mv_frame')
            full_schema_name: Tenant-scoped schema name (e.g., 'video_colpali_smol500_mv_frame_test_tenant')
            schema_definition: Full .sd file content as string
            config: Optional schema configuration

        Example:
            registry.register_schema(
                tenant_id="test_tenant",
                base_schema_name="video_colpali_smol500_mv_frame",
                full_schema_name="video_colpali_smol500_mv_frame_test_tenant",
                schema_definition=str(schema),
                config={"profile": "video_colpali_smol500_mv_frame"}
            )
        """
        logger.info(
            f"Registering schema '{base_schema_name}' for tenant '{tenant_id}' "
            f"(full name: '{full_schema_name}')"
        )

        self._config_manager.register_deployed_schema(
            tenant_id=tenant_id,
            base_schema_name=base_schema_name,
            full_schema_name=full_schema_name,
            schema_definition=schema_definition,
            config=config,
        )

        # Add to in-memory registry
        from datetime import datetime, timezone
        key = (tenant_id, base_schema_name)
        self._schemas[key] = SchemaInfo(
            tenant_id=tenant_id,
            base_schema_name=base_schema_name,
            full_schema_name=full_schema_name,
            schema_definition=schema_definition,
            config=config or {},
            deployment_time=datetime.now(timezone.utc).isoformat(),
        )

    def get_tenant_schemas(self, tenant_id: str) -> List[SchemaInfo]:
        """
        Get all schemas deployed for a specific tenant.

        This is the critical method that prevents schema wipeout.
        Before deploying a new schema, call this to get ALL existing schemas
        and include them in the deployment.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of SchemaInfo objects for all deployed schemas

        Example:
            # Get all existing schemas before deploying new one
            existing_schemas = registry.get_tenant_schemas("test_tenant")

            # Create ApplicationPackage with ALL schemas
            app_package = ApplicationPackage("videosearch")

            # Add the new schema
            app_package.add_schema(new_schema)

            # Add ALL existing schemas from registry
            for schema_info in existing_schemas:
                existing_schema = reconstruct_schema(schema_info.schema_definition)
                app_package.add_schema(existing_schema)

            # Now deployment won't wipe existing schemas
            deploy_package(app_package)
        """
        # Return schemas from in-memory registry
        return [
            schema_info
            for (tid, _), schema_info in self._schemas.items()
            if tid == tenant_id
        ]

    def get_all_schemas(self) -> List[SchemaInfo]:
        """
        Get all deployed schemas across all tenants.

        CRITICAL for multi-tenant deployments: Some backends require all schemas
        to be redeployed together. This method provides cross-tenant schema access.

        Returns:
            List of all SchemaInfo objects across all tenants
        """
        return list(self._schemas.values())

    def schema_exists(self, tenant_id: str, base_schema_name: str) -> bool:
        """
        Check if schema already deployed for tenant.

        Use this before deploying to avoid unnecessary redeployments.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name

        Returns:
            True if schema is deployed and not deleted, False otherwise

        Example:
            if not registry.schema_exists("test_tenant", "video_colpali_smol500_mv_frame"):
                # Deploy schema
                deploy_schema(...)
                # Register after successful deployment
                registry.register_schema(...)
        """
        key = (tenant_id, base_schema_name)
        return key in self._schemas

    def unregister_schema(self, tenant_id: str, base_schema_name: str) -> None:
        """
        Remove schema from registry (when deleted from backend).

        Marks schema as deleted in persistent storage and removes from in-memory registry.
        This preserves audit trail in persistent storage.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name

        Example:
            # After deleting schema from Vespa
            delete_schema_from_vespa(...)
            # Unregister from registry
            registry.unregister_schema("test_tenant", "video_colpali_smol500_mv_frame")
        """
        logger.info(
            f"Unregistering schema '{base_schema_name}' for tenant '{tenant_id}'"
        )
        self._config_manager.unregister_schema(tenant_id, base_schema_name)

        # Remove from in-memory registry
        key = (tenant_id, base_schema_name)
        if key in self._schemas:
            del self._schemas[key]

    def unregister_tenant_schemas(self, tenant_id: str) -> None:
        """
        Remove all schemas for a tenant.

        Call this when deleting a tenant to clean up their schemas.

        Args:
            tenant_id: Tenant identifier

        Example:
            # When deleting a tenant
            delete_tenant_from_vespa(...)
            registry.unregister_tenant_schemas("test_tenant")
        """
        logger.info(f"Unregistering all schemas for tenant '{tenant_id}'")
        schemas = self.get_tenant_schemas(tenant_id)
        for schema in schemas:
            self.unregister_schema(tenant_id, schema.base_schema_name)


# Global accessor function
_registry_instance: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    """
    Get or create the global SchemaRegistry instance.

    Returns:
        Singleton SchemaRegistry instance

    Example:
        from cogniverse_core.registries.schema_registry import get_schema_registry

        registry = get_schema_registry()
        registry.register_schema(...)
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SchemaRegistry()
    return _registry_instance
