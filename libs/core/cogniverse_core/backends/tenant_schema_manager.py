"""
Backend-agnostic tenant schema manager.

Uses BackendRegistry for backend discovery and delegates all operations
to the backend's schema management interface.
"""

import logging
from typing import Dict, List, Optional

from cogniverse_core.registries.backend_registry import BackendRegistry

logger = logging.getLogger(__name__)


class TenantSchemaManagerException(Exception):
    """Base exception for tenant schema management errors"""
    pass


class SchemaNotFoundException(TenantSchemaManagerException):
    """Raised when base schema template not found"""
    pass


class SchemaDeploymentException(TenantSchemaManagerException):
    """Raised when schema deployment fails"""
    pass


class TenantSchemaManager:
    """
    Backend-agnostic tenant schema manager.

    Delegates all operations to the backend's schema management interface
    via BackendRegistry. No direct backend dependencies.

    Architecture:
        - TenantSchemaManager (this class) lives in cogniverse-core
        - Backend interface defines schema methods
        - VespaBackend (in cogniverse-vespa) implements schema methods
        - TenantSchemaManager uses BackendRegistry to discover backends
    """

    def __init__(
        self,
        backend_name: str,
        backend_url: str,
        backend_port: int,
        http_port: int,
        config_manager,
        schema_loader,
    ):
        """
        Initialize tenant schema manager.

        Args:
            backend_name: Name of backend to use (e.g., "vespa")
            backend_url: Backend endpoint URL
            backend_port: Backend config server port for deployment
            http_port: Backend HTTP port for queries/status
            config_manager: ConfigManager instance (REQUIRED)
            schema_loader: SchemaLoader instance (REQUIRED)

        Raises:
            ValueError: If any required parameter is None
        """
        if backend_name is None:
            raise ValueError("backend_name is required")
        if backend_url is None:
            raise ValueError("backend_url is required")
        if backend_port is None:
            raise ValueError("backend_port is required")
        if http_port is None:
            raise ValueError("http_port is required")
        if config_manager is None:
            raise ValueError("config_manager is required")
        if schema_loader is None:
            raise ValueError("schema_loader is required")

        self.backend_name = backend_name
        self.backend_url = backend_url
        self.backend_port = backend_port
        self.http_port = http_port
        self._config_manager = config_manager
        self.schema_loader = schema_loader
        self._logger = logging.getLogger(self.__class__.__name__)
        self._deployed_schemas: Dict[str, List[str]] = {}  # tenant_id -> [schema_names]

        # Don't cache backend - get fresh backend per tenant operation
        # TenantSchemaManager operates across all tenants, each may have different config

    def _get_backend(self, tenant_id: str):
        """Get backend instance for given tenant."""
        registry = BackendRegistry.get_instance()
        try:
            # Create config dict with backend parameters from __init__
            config = {
                "backend": {
                    "url": self.backend_url,
                    "config_port": self.backend_port,  # Admin/schema management port
                    "port": self.http_port,  # Data operations port
                }
            }

            return registry.get_search_backend(
                name=self.backend_name,
                tenant_id=tenant_id,
                config=config,  # Pass backend configuration
                config_manager=self._config_manager,
                schema_loader=self.schema_loader
            )
        except ValueError as e:
            raise SchemaDeploymentException(
                f"Failed to get backend '{self.backend_name}': {e}"
            )

    def get_tenant_schema_name(self, tenant_id: str, base_schema_name: str) -> str:
        """
        Get tenant-specific schema name.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name

        Returns:
            Tenant-specific schema name
        """
        self._validate_tenant_id(tenant_id)
        self._validate_schema_name(base_schema_name)

        backend = self._get_backend(tenant_id)
        return backend.get_tenant_schema_name(tenant_id, base_schema_name)

    def deploy_tenant_schema(
        self,
        tenant_id: str,
        base_schema_name: str,
        force: bool = False
    ) -> str:
        """
        Deploy schema for tenant.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name to deploy
            force: Force redeployment even if schema exists

        Returns:
            Tenant-specific schema name that was deployed

        Raises:
            SchemaNotFoundException: If base schema not found
            SchemaDeploymentException: If deployment fails
        """
        self._validate_tenant_id(tenant_id)
        self._validate_schema_name(base_schema_name)

        tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)

        # Get backend for this specific tenant
        backend = self._get_backend(tenant_id)

        # Check if already exists (unless force)
        if not force and backend.schema_exists(base_schema_name, tenant_id):
            self._logger.info(
                f"Schema {tenant_schema_name} already exists for tenant {tenant_id}, skipping deployment"
            )
            self._cache_deployed_schema(tenant_id, base_schema_name)
            return tenant_schema_name

        # Deploy schema via backend
        self._logger.info(f"Deploying schema {tenant_schema_name} for tenant {tenant_id}")

        success = backend.deploy_schema(
            schema_name=base_schema_name,
            tenant_id=tenant_id
        )

        if not success:
            raise SchemaDeploymentException(
                f"Failed to deploy schema {base_schema_name} for tenant {tenant_id}"
            )

        self._cache_deployed_schema(tenant_id, base_schema_name)
        self._logger.info(f"✅ Successfully deployed {tenant_schema_name}")

        return tenant_schema_name

    def ensure_tenant_schema_exists(self, tenant_id: str, base_schema_name: str) -> bool:
        """
        Ensure schema exists for tenant (idempotent).

        Deploys schema if it doesn't exist, otherwise returns immediately.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name to deploy

        Returns:
            True if schema exists or was successfully deployed

        Raises:
            SchemaDeploymentException: If deployment fails
        """
        try:
            self.deploy_tenant_schema(tenant_id, base_schema_name, force=False)
            return True
        except SchemaDeploymentException:
            raise

    def delete_tenant_schemas(self, tenant_id: str) -> List[str]:
        """
        Delete all schemas for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of deleted schema names
        """
        self._validate_tenant_id(tenant_id)

        # Get list of schemas to delete
        schemas_to_delete = self.list_tenant_schemas(tenant_id)

        if not schemas_to_delete:
            self._logger.info(f"No schemas found for tenant {tenant_id}")
            return []

        # Get backend for this tenant
        backend = self._get_backend(tenant_id)

        deleted_schemas = []
        for base_schema_name in schemas_to_delete:
            try:
                deleted = backend.delete_schema(base_schema_name, tenant_id)
                deleted_schemas.extend(deleted)
            except Exception as e:
                self._logger.error(
                    f"Failed to delete schema {base_schema_name} for tenant {tenant_id}: {e}"
                )

        # Clear cache
        if tenant_id in self._deployed_schemas:
            del self._deployed_schemas[tenant_id]

        return deleted_schemas

    def list_tenant_schemas(self, tenant_id: str) -> List[str]:
        """
        List schemas deployed for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of base schema names (e.g., video_colpali_smol500_mv_frame)
        """
        self._validate_tenant_id(tenant_id)

        return self._deployed_schemas.get(tenant_id, [])

    def validate_tenant_schema(
        self,
        tenant_id: str,
        base_schema_name: str,
        config_manager
    ) -> bool:
        """
        Validate that schema exists for tenant.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name
            config_manager: ConfigManager instance

        Returns:
            True if schema exists, False otherwise
        """
        self._validate_tenant_id(tenant_id)
        self._validate_schema_name(base_schema_name)

        backend = self._get_backend(tenant_id)
        return backend.schema_exists(base_schema_name, tenant_id)

    def list_available_base_schemas(self) -> List[str]:
        """
        List available base schema templates.

        Returns:
            List of base schema names
        """
        # Use schema loader to list available schemas
        return self.schema_loader.list_schemas()

    def _cache_deployed_schema(self, tenant_id: str, base_schema_name: str) -> None:
        """Cache that a schema has been deployed for tenant."""
        if tenant_id not in self._deployed_schemas:
            self._deployed_schemas[tenant_id] = []
        if base_schema_name not in self._deployed_schemas[tenant_id]:
            self._deployed_schemas[tenant_id].append(base_schema_name)

    def _validate_tenant_id(self, tenant_id: str) -> None:
        """Validate tenant ID."""
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
        """Validate schema name."""
        if not schema_name:
            raise ValueError("schema_name is required")
        if not isinstance(schema_name, str):
            raise TypeError(f"schema_name must be string, got {type(schema_name)}")

    def clear_cache(self) -> None:
        """Clear the schema deployment cache."""
        self._deployed_schemas.clear()
        self._logger.info("Cleared schema deployment cache")

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        total_tenants = len(self._deployed_schemas)
        total_schemas = sum(len(schemas) for schemas in self._deployed_schemas.values())

        return {
            "total_tenants": total_tenants,
            "total_schemas": total_schemas,
            "tenants": list(self._deployed_schemas.keys()),
        }

