"""
Schema Registry - Tracks deployed schemas across tenants

Facade over ConfigManager that provides schema-specific operations.
Ensures all schemas are tracked and can be redeployed together.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cogniverse_core.registries.exceptions import (
    BackendDeploymentError,
    RegistryStorageError,
    SchemaRegistryInitializationError,
)

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
        from cogniverse_core.config.utils import create_default_config_manager
        config_manager = create_default_config_manager()
        registry = SchemaRegistry(config_manager, backend, schema_loader)

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

    def __init__(self, config_manager, backend, schema_loader, strict_mode: bool = True):
        """
        Initialize SchemaRegistry with required dependencies.

        All parameters are REQUIRED - no optional dependencies.
        Fail fast at construction if dependencies are missing.

        Args:
            config_manager: ConfigManager instance (REQUIRED)
            backend: Backend instance for schema deployment (REQUIRED)
            schema_loader: SchemaLoader instance for loading schema definitions (REQUIRED)
            strict_mode: If True, fail fast on initialization errors (default: True for production)
                        If False, continue with empty registry on errors (for testing only)

        Raises:
            ValueError: If any required parameter is None
            SchemaRegistryInitializationError: If strict_mode=True and schema loading fails
        """
        if config_manager is None:
            raise ValueError("config_manager is required")
        if backend is None:
            raise ValueError("backend is required")
        if schema_loader is None:
            raise ValueError("schema_loader is required")

        self._config_manager = config_manager
        self._backend = backend
        self._schema_loader = schema_loader
        self.strict_mode = strict_mode

        # In-memory registry of all deployed schemas
        # Key: (tenant_id, base_schema_name), Value: SchemaInfo
        self._schemas: Dict[tuple, SchemaInfo] = {}

        # Load all previously deployed schemas from persistent storage
        self._load_schemas_from_storage()

        logger.info(
            f"SchemaRegistry initialized with {len(self._schemas)} schemas loaded "
            f"(strict_mode={strict_mode})"
        )

    def _load_schemas_from_storage(self):
        """
        Load all schemas from ConfigManager into in-memory registry on startup.

        Behavior depends on strict_mode:
        - strict_mode=True: Raise exception if loading fails (production mode)
        - strict_mode=False: Log warning and continue with empty registry (test mode)

        Raises:
            SchemaRegistryInitializationError: If strict_mode=True and loading fails
        """
        try:
            # Load all schemas across all tenants using generic ConfigManager methods
            from cogniverse_core.config.store_interface import ConfigScope

            all_schema_data = self._config_manager.store.list_all_configs(
                scope=ConfigScope.SCHEMA,
                service="schema_registry",
            )

            schema_count = 0
            for entry in all_schema_data:
                schema_data = entry.config_value
                # Skip deleted schemas
                if schema_data.get("deleted", False):
                    continue

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
                schema_count += 1

            logger.info(f"Loaded {schema_count} schemas from storage")

        except Exception as e:
            if self.strict_mode:
                # Production mode - fail fast
                logger.error(
                    f"CRITICAL: Failed to load schemas from storage: {e}. "
                    f"This indicates database corruption or connectivity issues. "
                    f"Cannot initialize SchemaRegistry in strict mode."
                )
                raise SchemaRegistryInitializationError(
                    f"Cannot initialize SchemaRegistry: schema storage is unavailable. "
                    f"Error: {e}. "
                    f"This indicates database corruption or connectivity issues. "
                    f"Set strict_mode=False to start with empty registry (NOT recommended for production)."
                )
            else:
                # Test mode - continue with warning
                logger.warning(
                    f"Failed to load schemas from storage: {e}. "
                    f"Starting with empty registry (strict_mode=False)."
                )

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
        from datetime import datetime, timezone

        from cogniverse_core.config.store_interface import ConfigScope

        logger.info(
            f"Registering schema '{base_schema_name}' for tenant '{tenant_id}' "
            f"(full name: '{full_schema_name}')"
        )

        # Store schema metadata using generic ConfigManager storage
        config_key = f"schema_{base_schema_name}"
        deployment_time = datetime.now(timezone.utc).isoformat()

        value = {
            "tenant_id": tenant_id,
            "base_schema_name": base_schema_name,
            "full_schema_name": full_schema_name,
            "schema_definition": schema_definition,
            "config": config or {},
            "deployment_time": deployment_time,
        }

        self._config_manager.store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=config_key,
            config_value=value,
        )

        # Add to in-memory registry
        key = (tenant_id, base_schema_name)
        self._schemas[key] = SchemaInfo(
            tenant_id=tenant_id,
            base_schema_name=base_schema_name,
            full_schema_name=full_schema_name,
            schema_definition=schema_definition,
            config=config or {},
            deployment_time=deployment_time,
        )

    def _validate_tenant_id(self, tenant_id: str) -> None:
        """
        Validate tenant ID format.

        Args:
            tenant_id: Tenant identifier to validate

        Raises:
            ValueError: If tenant_id is invalid
            TypeError: If tenant_id is not a string
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not isinstance(tenant_id, str):
            raise TypeError(f"tenant_id must be string, got {type(tenant_id)}")

        # Validate character set (alphanumeric, underscore, colon only)
        import re
        if not re.match(r'^[a-zA-Z0-9_:]+$', tenant_id):
            raise ValueError(
                f"Invalid tenant_id '{tenant_id}': only alphanumeric, underscore, and colon allowed"
            )

    def _validate_schema_name(self, schema_name: str) -> None:
        """
        Validate schema name format.

        Args:
            schema_name: Schema name to validate

        Raises:
            ValueError: If schema_name is invalid
            TypeError: If schema_name is not a string
        """
        if not schema_name:
            raise ValueError("schema_name is required")
        if not isinstance(schema_name, str):
            raise TypeError(f"schema_name must be string, got {type(schema_name)}")

    def _rollback_deployment(
        self,
        previous_schemas: List[Dict[str, Any]],
        failed_schema_name: str
    ) -> None:
        """
        Rollback failed schema deployment by re-deploying previous schema set.

        This method is called when backend deployment succeeds but ConfigStore
        registration fails. It restores the backend to its previous state by
        re-deploying the old schema list (without the failed schema).

        The rollback uses backend.deploy_schemas() with the old list, which
        implicitly removes the newly deployed schema. This maintains consistency
        between backend and ConfigStore.

        Args:
            previous_schemas: List of schema definitions that existed before deployment
            failed_schema_name: Name of schema that failed registration (for logging)

        Note:
            Rollback is best-effort. If rollback fails, manual intervention is required.
            The system logs detailed error information to aid recovery.
        """
        logger.warning(
            f"Rolling back deployment of '{failed_schema_name}'. "
            f"Re-deploying {len(previous_schemas)} previous schemas."
        )

        try:
            # Re-deploy previous schema set (removes failed schema implicitly)
            success = self._backend.deploy_schemas(previous_schemas)
            if success:
                logger.info(
                    f"Successfully rolled back '{failed_schema_name}'. "
                    f"Backend state restored to {len(previous_schemas)} schemas."
                )
            else:
                logger.error(
                    f"Rollback of '{failed_schema_name}' reported failure. "
                    f"Backend state may be inconsistent. Manual intervention required."
                )
        except Exception as e:
            logger.error(
                f"Rollback of '{failed_schema_name}' failed with exception: {e}. "
                f"Backend state is inconsistent. Manual intervention required. "
                f"Expected schemas: {[s['name'] for s in previous_schemas]}"
            )

    def deploy_schema(
        self,
        tenant_id: str,
        base_schema_name: str,
        config: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> str:
        """
        Deploy schema for a tenant - MAIN ORCHESTRATION METHOD.

        This is the primary entry point for schema deployment. It:
        1. Validates inputs
        2. Checks if schema already deployed (unless force=True)
        3. Loads base schema definition
        4. Transforms to tenant-specific schema
        5. Collects ALL existing schemas (cross-tenant)
        6. Calls backend.deploy_schemas() with complete list
        7. Registers the newly deployed schema

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name (e.g., 'video_colpali_smol500_mv_frame')
            config: Optional schema configuration
            force: Force redeployment even if schema exists (default: False)

        Returns:
            Tenant-specific schema name (e.g., 'video_colpali_smol500_mv_frame_acme')

        Raises:
            ValueError: If backend or schema_loader not configured, or invalid inputs
            TypeError: If tenant_id or base_schema_name are not strings
            Exception: If schema deployment fails

        Example:
            registry = SchemaRegistry(backend=vespa_backend, schema_loader=loader)
            schema_name = registry.deploy_schema(
                tenant_id="acme",
                base_schema_name="video_colpali_smol500_mv_frame"
            )
            # Returns: "video_colpali_smol500_mv_frame_acme"
        """
        # Validate inputs
        self._validate_tenant_id(tenant_id)
        self._validate_schema_name(base_schema_name)

        # No need to check backend/schema_loader - guaranteed to exist (checked at construction)

        # Generate tenant-specific schema name
        tenant_schema_name = f"{base_schema_name}_{tenant_id}"

        # Check if already deployed (unless force=True)
        if not force and self.schema_exists(tenant_id, base_schema_name):
            logger.debug(f"Schema '{tenant_schema_name}' already deployed for tenant '{tenant_id}', skipping")
            return tenant_schema_name

        # Load base schema from schema loader
        try:
            base_schema_json = self._schema_loader.load_schema(base_schema_name)
        except Exception as e:
            raise Exception(f"Failed to load base schema '{base_schema_name}': {e}")

        # Transform schema name to tenant-specific
        base_schema_json['name'] = tenant_schema_name

        # Collect ALL existing schemas (including from other tenants)
        # This is critical for backends that require all schemas in each deployment
        import json

        # Save previous state for rollback (BEFORE adding new schema)
        existing_schemas = self._get_all_schemas()  # Private method
        previous_schemas = []
        for schema_info in existing_schemas:
            previous_schemas.append({
                "name": schema_info.full_schema_name,
                "definition": schema_info.schema_definition,
                "tenant_id": schema_info.tenant_id,
                "base_schema_name": schema_info.base_schema_name,
            })

        # Build new schema list (existing + new)
        all_schemas = list(previous_schemas)  # Copy for deployment
        all_schemas.append({
            "name": tenant_schema_name,
            "definition": json.dumps(base_schema_json),
            "tenant_id": tenant_id,
            "base_schema_name": base_schema_name,
        })

        logger.info(
            f"Deploying {len(all_schemas)} schemas "
            f"({len(previous_schemas)} existing + 1 new)"
        )

        # TRANSACTION PHASE 1: Deploy to backend (atomic operation)
        try:
            success = self._backend.deploy_schemas(all_schemas)
            if not success:
                raise BackendDeploymentError(
                    f"Backend failed to deploy schema '{tenant_schema_name}'"
                )
        except Exception as e:
            # Backend deployment failed - no rollback needed (nothing changed)
            logger.error(f"Backend deployment failed: {e}")
            raise BackendDeploymentError(
                f"Backend deployment failed for schema '{tenant_schema_name}': {e}"
            )

        # TRANSACTION PHASE 2: Register in ConfigStore (critical section)
        try:
            self.register_schema(
                tenant_id=tenant_id,
                base_schema_name=base_schema_name,
                full_schema_name=tenant_schema_name,
                schema_definition=json.dumps(base_schema_json),
                config=config,
            )
        except Exception as e:
            # ConfigStore registration failed AFTER backend succeeded
            # ROLLBACK: Re-deploy previous schemas to remove new one
            logger.error(
                f"ConfigStore registration failed: {e}. "
                f"Rolling back backend deployment..."
            )
            self._rollback_deployment(previous_schemas, tenant_schema_name)

            # Raise with clear error type
            raise RegistryStorageError(
                f"Failed to register schema '{tenant_schema_name}' in ConfigStore: {e}. "
                f"Backend deployment has been rolled back."
            )

        logger.info(
            f"Successfully deployed and registered schema '{tenant_schema_name}' "
            f"for tenant '{tenant_id}'"
        )
        return tenant_schema_name

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

    def _get_all_schemas(self) -> List[SchemaInfo]:
        """
        Get all deployed schemas across all tenants (PRIVATE - internal use only).

        Used internally by deploy_schema() to collect existing schemas before deployment.
        This method provides cross-tenant schema access for backends that require
        all schemas to be redeployed together.

        Returns:
            List of all SchemaInfo objects across all tenants

        Note:
            This is a private method. External code should NOT call this directly.
            Use deploy_schema() for orchestrated deployment instead.
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
        from datetime import datetime, timezone

        from cogniverse_core.config.store_interface import ConfigScope

        logger.info(
            f"Unregistering schema '{base_schema_name}' for tenant '{tenant_id}'"
        )

        # Mark schema as deleted in persistent storage (preserves audit trail)
        config_key = f"schema_{base_schema_name}"

        # Get existing entry to mark as deleted
        entry = self._config_manager.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=config_key,
        )

        if entry:
            schema_info = entry.config_value
            schema_info["deleted"] = True
            schema_info["deleted_at"] = datetime.now(timezone.utc).isoformat()

            self._config_manager.store.set_config(
                tenant_id=tenant_id,
                scope=ConfigScope.SCHEMA,
                service="schema_registry",
                config_key=config_key,
                config_value=schema_info,
            )

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
