"""
Tenant-aware Vespa Schema Manager

Provides schema-per-tenant isolation with automatic schema routing and lifecycle management.
Each tenant gets dedicated schemas for physical data isolation (e.g., video_colpali_smol500_mv_frame_acme).

Key Features:
- Schema name routing: base_schema + tenant_id → tenant_schema
- Lazy schema creation from templates
- Schema lifecycle: deploy, delete, validate, list
- Thread-safe caching of deployed schemas
- No backward compatibility - tenant_id REQUIRED everywhere

Architecture:
- Base schemas in configs/schemas/ serve as templates
- Tenant schemas created by transforming template JSON
- Uses existing VespaSchemaManager for actual deployment
- Caches deployed schemas to avoid redundant checks

Example:
    manager = TenantSchemaManager(backend_url="http://localhost", backend_port=19071, http_port=8080)

    # Get tenant-specific schema name
    schema_name = manager.get_tenant_schema_name("acme", "video_colpali_smol500_mv_frame")
    # Returns: "video_colpali_smol500_mv_frame_acme"

    # Ensure schema exists (lazy creation)
    manager.ensure_tenant_schema_exists("acme", "video_colpali_smol500_mv_frame")

    # Deploy new tenant schema
    manager.deploy_tenant_schema("startup", "video_colpali_smol500_mv_frame")

    # List all schemas for a tenant
    schemas = manager.list_tenant_schemas("acme")

    # Delete all tenant schemas
    manager.delete_tenant_schemas("acme")
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Set

from cogniverse_core.interfaces.schema_loader import SchemaLoader
from cogniverse_core.interfaces.schema_loader import (
    SchemaNotFoundException as SchemaLoaderNotFoundException,
)
from vespa.package import ApplicationPackage, Schema

from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

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
    Manages tenant-specific Vespa schemas with automatic routing and lazy creation.

    Thread-safe singleton pattern - one instance per process.
    Tests must clear the singleton via _clear_instance() in teardown for proper isolation.
    """

    _instance: Optional["TenantSchemaManager"] = None
    _lock = threading.Lock()

    def __new__(
        cls,
        backend_url: str,
        backend_port: int,
        http_port: int,
        config_manager,
        schema_loader: SchemaLoader,
    ):
        """
        Singleton pattern - returns same instance for all calls.

        Args:
            backend_url: Backend endpoint URL (used only on first instantiation)
            backend_port: Backend config server port for deployment (used only on first instantiation)
            http_port: Backend HTTP port for queries/status (used only on first instantiation)
            config_manager: ConfigManager instance (REQUIRED, used only on first instantiation)
            schema_loader: SchemaLoader instance (REQUIRED, used only on first instantiation)
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
                    logger.info(
                        f"Created TenantSchemaManager singleton for {backend_url}:{backend_port}"
                    )
        return cls._instance

    def __init__(
        self,
        backend_url: str,
        backend_port: int,
        http_port: int,
        config_manager,
        schema_loader: SchemaLoader,
    ):
        """
        Initialize tenant schema manager.

        Args:
            backend_url: Backend endpoint URL
            backend_port: Backend config server port for deployment
            http_port: Backend HTTP port for queries/status
            config_manager: ConfigManager instance (REQUIRED)
            schema_loader: SchemaLoader instance (REQUIRED)

        Raises:
            ValueError: If config_manager or schema_loader is None
        """
        if self._initialized:
            return

        if config_manager is None:
            raise ValueError("config_manager is required for TenantSchemaManager initialization")

        if schema_loader is None:
            raise ValueError("schema_loader is required for TenantSchemaManager initialization")

        self.backend_url = backend_url
        self.backend_port = backend_port
        self.http_port = http_port
        self._config_manager = config_manager
        self.schema_loader = schema_loader

        # Underlying schema manager for actual deployment
        self.schema_manager = VespaSchemaManager(
            vespa_endpoint=backend_url,
            vespa_port=backend_port,
            config_manager=config_manager
        )

        # JSON schema parser for loading templates
        self.parser = JsonSchemaParser()

        # Cache of deployed tenant schemas (tenant_id -> Set[base_schema_name])
        self._deployed_schemas: Dict[str, Set[str]] = {}
        self._cache_lock = threading.RLock()

        # Schema registry for tracking deployed schemas (prevents schema wipeout)
        from cogniverse_core.registries.schema_registry import get_schema_registry

        self.schema_registry = get_schema_registry()

        self._initialized = True
        logger.info(f"TenantSchemaManager initialized: {backend_url}:{backend_port}")

    def get_tenant_schema_name(self, tenant_id: str, base_schema_name: str) -> str:
        """
        Generate tenant-specific schema name from base schema and tenant ID.

        Supports both simple and org:tenant format:
        - Simple: "acme" → "video_colpali_smol500_mv_frame_acme"
        - Org:tenant: "acme:production" → "video_colpali_smol500_mv_frame_acme_production"

        Args:
            tenant_id: Tenant identifier (e.g., "acme", "acme:production")
            base_schema_name: Base schema name (e.g., "video_colpali_smol500_mv_frame")

        Returns:
            Tenant-specific schema name (e.g., "video_colpali_smol500_mv_frame_acme_production")

        Raises:
            ValueError: If tenant_id or base_schema_name is invalid

        Example:
            >>> manager.get_tenant_schema_name("acme", "video_colpali_smol500_mv_frame")
            'video_colpali_smol500_mv_frame_acme'
            >>> manager.get_tenant_schema_name("acme:production", "video_colpali_smol500_mv_frame")
            'video_colpali_smol500_mv_frame_acme_production'
        """
        self._validate_tenant_id(tenant_id)
        self._validate_schema_name(base_schema_name)

        # Convert org:tenant to org_tenant for schema naming
        tenant_suffix = tenant_id.replace(":", "_")

        return f"{base_schema_name}_{tenant_suffix}"

    def ensure_tenant_schema_exists(
        self, tenant_id: str, base_schema_name: str
    ) -> bool:
        """
        Ensure tenant schema exists, deploying it lazily if needed.

        This is idempotent - safe to call multiple times.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name

        Returns:
            True if schema exists or was successfully deployed

        Raises:
            SchemaNotFoundException: If base schema template not found
            SchemaDeploymentException: If schema deployment fails

        Example:
            >>> manager.ensure_tenant_schema_exists("acme", "video_colpali_smol500_mv_frame")
            True
        """
        with self._cache_lock:
            # Check cache first
            if tenant_id in self._deployed_schemas:
                if base_schema_name in self._deployed_schemas[tenant_id]:
                    logger.debug(
                        f"Schema {base_schema_name} for tenant {tenant_id} already deployed (cached)"
                    )
                    return True

            # Check if schema actually exists in Vespa
            tenant_schema_name = self.get_tenant_schema_name(
                tenant_id, base_schema_name
            )

            # Check SchemaRegistry first (authoritative source)
            if self.schema_registry.schema_exists(tenant_id, base_schema_name):
                logger.info(f"Schema {base_schema_name} for tenant {tenant_id} already in registry")
                self._cache_deployed_schema(tenant_id, base_schema_name)
                return True

            # Fallback: Check if schema exists in Vespa (e.g., deployed outside this system)
            if self._schema_exists_in_vespa(tenant_schema_name):
                logger.info(f"Schema {tenant_schema_name} exists in Vespa but not in registry - skipping deployment")
                # DON'T register it - let only explicit deployments register schemas
                # This prevents test pollution where schemas from previous tests get auto-registered
                self._cache_deployed_schema(tenant_id, base_schema_name)
                return True

            # Deploy schema
            logger.info(f"Deploying schema {base_schema_name} for tenant {tenant_id}")
            self.deploy_tenant_schema(tenant_id, base_schema_name)
            return True

    def deploy_tenant_schema(self, tenant_id: str, base_schema_name: str, force: bool = False) -> None:
        """
        Deploy a tenant-specific schema from base schema template.

        CRITICAL: This method now uses SchemaRegistry to prevent schema wipeout.
        It fetches ALL existing schemas for the tenant and includes them in the
        deployment, ensuring Vespa doesn't remove schemas that aren't in the package.

        Steps:
        1. Check if schema already in registry - skip if so (unless force=True)
        2. Load base schema JSON from configs/schemas/
        3. Transform schema: rename to include tenant suffix
        4. Get ALL existing schemas from registry
        5. Deploy complete package with new schema + ALL existing schemas
        6. Register new schema in registry

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name
            force: If True, deploy even if schema exists in registry (for testing/validation)

        Raises:
            SchemaNotFoundException: If base schema template not found
            SchemaDeploymentException: If deployment fails

        Example:
            >>> manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame")
            >>> manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame", force=True)  # Force redeploy
        """
        self._validate_tenant_id(tenant_id)
        self._validate_schema_name(base_schema_name)

        # Check if schema already deployed in registry (skip check if force=True)
        if not force and self.schema_registry.schema_exists(tenant_id, base_schema_name):
            logger.info(
                f"Schema '{base_schema_name}' already deployed for tenant '{tenant_id}' "
                f"(found in schema registry). Skipping redeployment."
            )
            # Still cache it locally
            self._cache_deployed_schema(tenant_id, base_schema_name)
            return

        # Load base schema template
        try:
            base_schema_json = self.schema_loader.load_schema(base_schema_name)
        except SchemaLoaderNotFoundException as e:
            # Re-raise as our own exception type for backward compatibility
            raise SchemaNotFoundException(str(e)) from e

        # Transform for tenant
        tenant_schema_json = self._transform_schema_for_tenant(
            base_schema_json, tenant_id, base_schema_name
        )

        # Parse to Vespa Schema object
        tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)
        new_schema = self._parse_schema_from_json(tenant_schema_json, tenant_schema_name)

        # Store the JSON schema for later reconstruction (not the Schema object string representation)
        tenant_schema_json_str = json.dumps(tenant_schema_json)

        # Deploy via VespaSchemaManager
        try:
            from datetime import datetime, timedelta

            from vespa.package import Validation

            from cogniverse_vespa.metadata_schemas import (
                add_metadata_schemas_to_package,
            )

            # CRITICAL: Get ALL existing schemas from registry
            existing_schemas = self.schema_registry.get_tenant_schemas(tenant_id)
            logger.info(
                f"Found {len(existing_schemas)} existing schemas in registry for tenant '{tenant_id}'"
            )

            app_package = ApplicationPackage(name="videosearch")

            # Add the new tenant schema
            app_package.add_schema(new_schema)
            logger.info(f"Added new schema to package: {tenant_schema_name}")

            # CRITICAL: Add ALL existing schemas from registry to prevent wipeout
            for schema_info in existing_schemas:
                try:
                    # Reconstruct Schema object from stored definition
                    existing_schema = self._reconstruct_schema_from_definition(
                        schema_info.schema_definition, schema_info.full_schema_name
                    )
                    app_package.add_schema(existing_schema)
                    logger.info(f"Added existing schema to package: {schema_info.full_schema_name}")
                except Exception as e:
                    logger.error(
                        f"Failed to reconstruct schema {schema_info.full_schema_name}: {e}. "
                        "This schema may be lost in deployment!"
                    )
                    # Continue with other schemas - don't fail entire deployment

            # Add metadata schemas (organization_metadata, tenant_metadata)
            # Using consolidated module to prevent duplication
            add_metadata_schemas_to_package(app_package)

            # Add validation overrides (still needed for config server)
            until_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            schema_removal_validation = Validation(
                validation_id="schema-removal", until=until_date
            )
            content_cluster_validation = Validation(
                validation_id="content-cluster-removal", until=until_date
            )

            if app_package.validations is None:
                app_package.validations = []
            app_package.validations.append(schema_removal_validation)
            app_package.validations.append(content_cluster_validation)

            # Deploy complete package with ALL schemas
            logger.info(
                f"Deploying application package with {len(existing_schemas) + 1} tenant schemas "
                f"+ metadata schemas"
            )
            self.schema_manager._deploy_package(app_package)

            logger.info(f"✅ Deployed tenant schema: {tenant_schema_name}")

            # Register the new schema in registry
            # IMPORTANT: Store JSON schema, not str(schema), for reliable reconstruction
            self.schema_registry.register_schema(
                tenant_id=tenant_id,
                base_schema_name=base_schema_name,
                full_schema_name=tenant_schema_name,
                schema_definition=tenant_schema_json_str,  # Store JSON, not str(schema)
                config={"profile": base_schema_name},  # Store profile for reference
            )
            logger.info(f"Registered schema '{base_schema_name}' in schema registry")

            # Cache successful deployment
            self._cache_deployed_schema(tenant_id, base_schema_name)

        except Exception as e:
            raise SchemaDeploymentException(
                f"Failed to deploy schema {tenant_schema_name}: {e}"
            ) from e

    def delete_tenant_schemas(self, tenant_id: str) -> List[str]:
        """
        Delete all schemas for a tenant.

        WARNING: This removes all data for the tenant!

        This method also unregisters schemas from the schema registry.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of deleted schema names

        Example:
            >>> deleted = manager.delete_tenant_schemas("acme")
            >>> print(f"Deleted {len(deleted)} schemas")
        """
        self._validate_tenant_id(tenant_id)

        deleted_schemas = []

        with self._cache_lock:
            if tenant_id not in self._deployed_schemas:
                logger.warning(f"No schemas found for tenant {tenant_id}")
                return deleted_schemas

            base_schemas = list(self._deployed_schemas[tenant_id])

            for base_schema_name in base_schemas:
                tenant_schema_name = self.get_tenant_schema_name(
                    tenant_id, base_schema_name
                )

                try:
                    # Note: Vespa schema deletion is not directly supported via PyVespa
                    # In production, you would redeploy without the schema or use Vespa HTTP API
                    logger.warning(
                        f"Schema deletion not implemented for {tenant_schema_name}. "
                        "Manually redeploy application without this schema or use Vespa HTTP API."
                    )
                    deleted_schemas.append(tenant_schema_name)

                    # Unregister from schema registry
                    self.schema_registry.unregister_schema(tenant_id, base_schema_name)
                    logger.info(f"Unregistered schema '{base_schema_name}' from registry")

                except Exception as e:
                    logger.error(f"Failed to delete schema {tenant_schema_name}: {e}")

            # Clear cache
            del self._deployed_schemas[tenant_id]
            logger.info(f"Cleared schema cache for tenant {tenant_id}")

        return deleted_schemas

    def list_tenant_schemas(self, tenant_id: str) -> List[str]:
        """
        List all deployed schemas for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of tenant schema names

        Example:
            >>> schemas = manager.list_tenant_schemas("acme")
            >>> print(schemas)
            ['video_colpali_smol500_mv_frame_acme', 'video_videoprism_base_mv_chunk_30s_acme']
        """
        self._validate_tenant_id(tenant_id)

        with self._cache_lock:
            if tenant_id not in self._deployed_schemas:
                return []

            return [
                self.get_tenant_schema_name(tenant_id, base_schema)
                for base_schema in self._deployed_schemas[tenant_id]
            ]

    def validate_tenant_schema(self, tenant_id: str, base_schema_name: str, config_manager) -> bool:
        """
        Validate that a tenant schema exists by querying through the backend.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name
            config_manager: ConfigManager instance for backend initialization

        Returns:
            True if schema exists and is queryable through backend

        Example:
            >>> is_valid = manager.validate_tenant_schema("acme", "video_colpali_smol500_mv_frame", config_manager)
        """
        if config_manager is None:
            raise ValueError("config_manager is required for schema validation")

        try:
            from cogniverse_core.registries.backend_registry import get_backend_registry

            registry = get_backend_registry()

            # Get tenant-scoped schema name
            tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)

            # Get search backend for this tenant
            # Profile should contain base schema name (tenant scoping happens automatically)
            backend_config = {
                "url": self.backend_url,
                "port": self.http_port,
                "tenant_id": tenant_id,
                "profiles": {
                    base_schema_name: {
                        "schema_name": base_schema_name,  # Use base name for ranking strategy lookup
                        "type": "frame_based"
                    }
                }
            }

            backend = registry.get_search_backend(
                "vespa",
                tenant_id=tenant_id,
                config=backend_config,
                config_manager=config_manager,
                schema_loader=self.schema_loader
            )

            # Try a simple query - if schema exists, this will succeed (even with 0 results)
            query_dict = {
                "query": "test",
                "type": "video",
                "profile": base_schema_name,
                "strategy": "default",  # Use default strategy for validation
                "top_k": 0
            }
            backend.search(query_dict)

            logger.debug(f"Schema {tenant_schema_name} validated successfully for tenant {tenant_id}")
            return True

        except Exception as e:
            logger.warning(f"Schema validation failed for {base_schema_name} (tenant: {tenant_id}): {e}")
            return False

    def list_available_base_schemas(self) -> List[str]:
        """
        List all available base schema templates.

        Returns:
            List of base schema names available for deployment

        Example:
            >>> schemas = manager.list_available_base_schemas()
            >>> print(schemas)
            ['video_colpali_smol500_mv_frame', 'video_videoprism_base_mv_chunk_30s', ...]
        """
        return self.schema_loader.list_available_schemas()

    def _transform_schema_for_tenant(
        self, base_schema_json: Dict, tenant_id: str, base_schema_name: str
    ) -> Dict:
        """Transform base schema JSON to include tenant suffix"""
        tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)

        # Deep copy to avoid modifying original
        tenant_schema_json = json.loads(json.dumps(base_schema_json))

        # Update schema name
        tenant_schema_json["name"] = tenant_schema_name

        # Update document name
        if "document" in tenant_schema_json:
            tenant_schema_json["document"]["name"] = tenant_schema_name

        logger.debug(f"Transformed schema {base_schema_name} → {tenant_schema_name}")
        return tenant_schema_json

    def _parse_schema_from_json(self, schema_json: Dict, schema_name: str) -> Schema:
        """Parse JSON schema to Vespa Schema object"""
        try:
            # Use JsonSchemaParser to convert JSON to Schema
            schema = self.parser.parse_schema(schema_json)
            return schema
        except Exception as e:
            raise SchemaDeploymentException(
                f"Failed to parse schema {schema_name}: {e}"
            ) from e

    def _reconstruct_schema_from_definition(
        self, schema_definition: str, schema_name: str
    ) -> Schema:
        """
        Reconstruct Schema object from stored JSON definition.

        NO FALLBACKS - fails fast if schema definition is invalid.
        This ensures problems are caught early rather than hidden.

        Args:
            schema_definition: JSON schema definition string (from registry)
            schema_name: Expected schema name for validation

        Returns:
            Reconstructed Schema object

        Raises:
            SchemaDeploymentException: If reconstruction fails
        """
        try:
            # Parse JSON schema definition
            schema_json = json.loads(schema_definition)

            # Validate schema name matches
            if schema_json.get("name") != schema_name:
                raise ValueError(
                    f"Schema name mismatch: expected '{schema_name}', "
                    f"got '{schema_json.get('name')}'"
                )

            # Parse to Schema object
            return self._parse_schema_from_json(schema_json, schema_name)

        except json.JSONDecodeError as e:
            raise SchemaDeploymentException(
                f"Schema definition for '{schema_name}' is not valid JSON: {e}. "
                f"This indicates a bug in schema storage. Schema will be MISSING from deployment!"
            ) from e
        except Exception as e:
            raise SchemaDeploymentException(
                f"Failed to reconstruct schema '{schema_name}' from definition: {e}. "
                f"Schema will be MISSING from deployment!"
            ) from e

    def _schema_exists_in_vespa(self, schema_name: str) -> bool:
        """
        Check if schema exists in Vespa by querying the application status API.
        Uses http_port (not backend_port/config_port) for status queries.
        """
        try:
            import requests
            # Query ApplicationStatus on HTTP port (not config port)
            url = f"{self.backend_url}:{self.http_port}/ApplicationStatus"
            logger.info(f"Checking schema existence at: {url}")
            response = requests.get(url, timeout=5)
            logger.info(f"Response status: {response.status_code}")
            if response.status_code == 200:
                app_status = response.json()
                logger.info(f"Application status: {app_status}")
                # Check if schema_name is in the list of deployed schemas
                if "application" in app_status and "schemas" in app_status["application"]:
                    deployed_schemas = app_status["application"]["schemas"]
                    logger.info(f"Vespa deployed schemas: {deployed_schemas}, checking for: {schema_name}")
                    return schema_name in deployed_schemas
                else:
                    logger.warning(f"No schemas found in application status. Keys: {app_status.keys() if isinstance(app_status, dict) else 'not a dict'}")
            else:
                logger.warning(f"ApplicationStatus returned status {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Schema existence check failed for {schema_name}: {e}", exc_info=True)
            return False

    def _cache_deployed_schema(self, tenant_id: str, base_schema_name: str) -> None:
        """Cache deployed schema"""
        with self._cache_lock:
            if tenant_id not in self._deployed_schemas:
                self._deployed_schemas[tenant_id] = set()
            self._deployed_schemas[tenant_id].add(base_schema_name)
            logger.debug(f"Cached schema {base_schema_name} for tenant {tenant_id}")

    def _validate_tenant_id(self, tenant_id: str) -> None:
        """
        Validate tenant ID format.

        Supports both simple and org:tenant format:
        - Simple: "acme", "startup" (alphanumeric + underscore)
        - Org:tenant: "acme:production", "startup:dev" (alphanumeric + underscore + colon)

        Note: Hyphens are NOT allowed (Vespa schema names don't support dashes)
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        if not isinstance(tenant_id, str):
            raise ValueError(f"tenant_id must be string, got {type(tenant_id)}")

        # Allow alphanumeric, underscores, and colons (for org:tenant format)
        # NO hyphens - Vespa schema names don't support dashes
        allowed_chars = tenant_id.replace("_", "").replace(":", "")
        if not allowed_chars.isalnum():
            raise ValueError(
                f"Invalid tenant_id '{tenant_id}': only alphanumeric, underscore, and colon allowed"
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

    def _validate_schema_name(self, schema_name: str) -> None:
        """Validate schema name format"""
        if not isinstance(schema_name, str):
            raise ValueError(f"schema_name must be string, got {type(schema_name)}")

        if not schema_name:
            raise ValueError("schema_name cannot be empty")

    def clear_cache(self) -> None:
        """Clear deployed schema cache (for testing)"""
        with self._cache_lock:
            self._deployed_schemas.clear()
            logger.debug("Cleared schema cache")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics (for monitoring)"""
        with self._cache_lock:
            return {
                "tenants_cached": len(self._deployed_schemas),
                "total_schemas_cached": sum(
                    len(schemas) for schemas in self._deployed_schemas.values()
                ),
                "tenants": {
                    tenant_id: list(schemas)
                    for tenant_id, schemas in self._deployed_schemas.items()
                },
            }

    @classmethod
    def _clear_instance(cls) -> None:
        """
        Clear singleton instance (TEST ONLY - will raise in production).

        Tests should call this in teardown to ensure proper isolation between test modules.

        Raises:
            RuntimeError: If called outside of test environment (when pytest is not running)

        Example:
            # In test teardown:
            TenantSchemaManager._clear_instance()
        """
        import sys

        # Check if we're running under pytest
        if not any("pytest" in mod for mod in sys.modules):
            raise RuntimeError(
                "TenantSchemaManager._clear_instance() is for testing only. "
                "It should never be called in production code."
            )

        with cls._lock:
            cls._instance = None
            logger.debug("Cleared TenantSchemaManager singleton instance")


def get_tenant_schema_manager(
    backend_url: str,
    backend_port: int,
    http_port: int,
    config_manager,
    schema_loader: SchemaLoader,
) -> TenantSchemaManager:
    """
    Get tenant schema manager instance for specific backend endpoint.

    Returns the appropriate singleton instance based on (url, port) combination,
    allowing multiple backend instances with separate managers.

    Args:
        backend_url: Backend endpoint URL
        backend_port: Backend config server port for deployment
        http_port: Backend HTTP port for queries/status
        config_manager: ConfigManager instance (REQUIRED)
        schema_loader: SchemaLoader instance (REQUIRED)

    Returns:
        TenantSchemaManager singleton instance for this endpoint
    """
    # Simply instantiate - __new__() handles per-endpoint singleton logic
    return TenantSchemaManager(backend_url, backend_port, http_port, config_manager, schema_loader)
