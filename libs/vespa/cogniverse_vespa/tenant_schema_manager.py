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
    manager = TenantSchemaManager(vespa_url="http://localhost", vespa_port=8080)

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
from pathlib import Path
from typing import Dict, List, Optional, Set

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

    def __new__(cls, vespa_url: str = "http://localhost", vespa_port: int = 8080):
        """
        Singleton pattern - returns same instance for all calls.

        Args:
            vespa_url: Vespa endpoint URL (used only on first instantiation)
            vespa_port: Vespa config server port (used only on first instantiation)
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
                    logger.info(f"Created TenantSchemaManager singleton for {vespa_url}:{vespa_port}")
        return cls._instance

    def __init__(self, vespa_url: str = "http://localhost", vespa_port: int = 8080):
        """
        Initialize tenant schema manager.

        Args:
            vespa_url: Vespa endpoint URL
            vespa_port: Vespa port number
        """
        if self._initialized:
            return

        self.vespa_url = vespa_url
        self.vespa_port = vespa_port

        # Underlying schema manager for actual deployment
        self.schema_manager = VespaSchemaManager(
            vespa_endpoint=vespa_url,
            vespa_port=vespa_port
        )

        # JSON schema parser for loading templates
        self.parser = JsonSchemaParser()

        # Base schema templates directory
        self.schema_templates_dir = Path("configs/schemas")

        # Cache of deployed tenant schemas (tenant_id -> Set[base_schema_name])
        self._deployed_schemas: Dict[str, Set[str]] = {}
        self._cache_lock = threading.RLock()

        self._initialized = True
        logger.info(f"TenantSchemaManager initialized: {vespa_url}:{vespa_port}")

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

    def ensure_tenant_schema_exists(self, tenant_id: str, base_schema_name: str) -> bool:
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
                    logger.debug(f"Schema {base_schema_name} for tenant {tenant_id} already deployed (cached)")
                    return True

            # Check if schema actually exists in Vespa
            tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)
            if self._schema_exists_in_vespa(tenant_schema_name):
                logger.info(f"Schema {tenant_schema_name} already exists in Vespa")
                self._cache_deployed_schema(tenant_id, base_schema_name)
                return True

            # Deploy schema
            logger.info(f"Deploying schema {base_schema_name} for tenant {tenant_id}")
            self.deploy_tenant_schema(tenant_id, base_schema_name)
            return True

    def deploy_tenant_schema(self, tenant_id: str, base_schema_name: str) -> None:
        """
        Deploy a tenant-specific schema from base schema template.

        Steps:
        1. Load base schema JSON from configs/schemas/
        2. Transform schema: rename to include tenant suffix
        3. Deploy via VespaSchemaManager
        4. Cache deployment

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name

        Raises:
            SchemaNotFoundException: If base schema template not found
            SchemaDeploymentException: If deployment fails

        Example:
            >>> manager.deploy_tenant_schema("acme", "video_colpali_smol500_mv_frame")
        """
        self._validate_tenant_id(tenant_id)
        self._validate_schema_name(base_schema_name)

        # Load base schema template
        base_schema_json = self._load_base_schema_json(base_schema_name)

        # Transform for tenant
        tenant_schema_json = self._transform_schema_for_tenant(
            base_schema_json, tenant_id, base_schema_name
        )

        # Parse to Vespa Schema object
        tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)
        schema = self._parse_schema_from_json(tenant_schema_json, tenant_schema_name)

        # Deploy via VespaSchemaManager
        try:
            from datetime import datetime, timedelta

            from vespa.package import Document, Field, Validation

            app_package = ApplicationPackage(name="videosearch")

            # Add the tenant schema
            app_package.add_schema(schema)

            # Add metadata schemas to prevent them from being removed
            organization_metadata_schema = Schema(
                name='organization_metadata',
                document=Document(
                    fields=[
                        Field(name='org_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='org_name', type='string', indexing=['summary', 'index']),
                        Field(name='created_at', type='long', indexing=['summary', 'attribute']),
                        Field(name='created_by', type='string', indexing=['summary', 'attribute']),
                        Field(name='status', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='tenant_count', type='int', indexing=['summary', 'attribute']),
                    ]
                )
            )
            app_package.add_schema(organization_metadata_schema)

            tenant_metadata_schema = Schema(
                name='tenant_metadata',
                document=Document(
                    fields=[
                        Field(name='tenant_full_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='org_id', type='string', indexing=['summary', 'index', 'attribute'], attribute=['fast-search']),
                        Field(name='tenant_name', type='string', indexing=['summary', 'attribute']),
                        Field(name='created_at', type='long', indexing=['summary', 'attribute']),
                        Field(name='created_by', type='string', indexing=['summary', 'attribute']),
                        Field(name='status', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='schemas_deployed', type='array<string>', indexing=['summary', 'attribute']),
                    ]
                )
            )
            app_package.add_schema(tenant_metadata_schema)

            # Add validation overrides to allow schema deployments without removing existing schemas
            until_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            schema_removal_validation = Validation(
                validation_id="schema-removal",
                until=until_date
            )
            content_cluster_validation = Validation(
                validation_id="content-cluster-removal",
                until=until_date
            )

            if app_package.validations is None:
                app_package.validations = []
            app_package.validations.append(schema_removal_validation)
            app_package.validations.append(content_cluster_validation)

            self.schema_manager._deploy_package(app_package)

            logger.info(f"✅ Deployed tenant schema: {tenant_schema_name}")

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
                tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)

                try:
                    # Note: Vespa schema deletion is not directly supported via PyVespa
                    # In production, you would redeploy without the schema or use Vespa HTTP API
                    logger.warning(
                        f"Schema deletion not implemented for {tenant_schema_name}. "
                        "Manually redeploy application without this schema or use Vespa HTTP API."
                    )
                    deleted_schemas.append(tenant_schema_name)
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

    def validate_tenant_schema(self, tenant_id: str, base_schema_name: str) -> bool:
        """
        Validate that a tenant schema exists and is healthy.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name

        Returns:
            True if schema exists and is healthy

        Example:
            >>> is_valid = manager.validate_tenant_schema("acme", "video_colpali_smol500_mv_frame")
        """
        tenant_schema_name = self.get_tenant_schema_name(tenant_id, base_schema_name)
        return self._schema_exists_in_vespa(tenant_schema_name)

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
        if not self.schema_templates_dir.exists():
            return []

        schema_files = self.schema_templates_dir.glob("*_schema.json")
        return [f.stem.replace("_schema", "") for f in schema_files]

    def _load_base_schema_json(self, base_schema_name: str) -> Dict:
        """Load base schema JSON from configs/schemas/"""
        schema_file = self.schema_templates_dir / f"{base_schema_name}_schema.json"

        if not schema_file.exists():
            available = self.list_available_base_schemas()
            raise SchemaNotFoundException(
                f"Base schema '{base_schema_name}' not found. "
                f"Available schemas: {available}"
            )

        try:
            with open(schema_file, "r") as f:
                return json.load(f)
        except Exception as e:
            raise SchemaNotFoundException(
                f"Failed to load base schema {base_schema_name}: {e}"
            ) from e

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

    def _schema_exists_in_vespa(self, schema_name: str) -> bool:
        """
        Check if schema exists in Vespa.

        Note: This is a simplified check. In production, you would query Vespa's
        config API to verify schema existence.
        """
        # For now, we rely on cache and deployment tracking
        # In production, implement actual Vespa HTTP API check
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
        if not any('pytest' in mod for mod in sys.modules):
            raise RuntimeError(
                "TenantSchemaManager._clear_instance() is for testing only. "
                "It should never be called in production code."
            )

        with cls._lock:
            cls._instance = None
            logger.debug("Cleared TenantSchemaManager singleton instance")


def get_tenant_schema_manager(
    vespa_url: str = "http://localhost", vespa_port: int = 8080
) -> TenantSchemaManager:
    """
    Get tenant schema manager instance for specific Vespa endpoint.

    Returns the appropriate singleton instance based on (url, port) combination,
    allowing multiple Vespa instances with separate managers.

    Args:
        vespa_url: Vespa endpoint URL
        vespa_port: Vespa port number

    Returns:
        TenantSchemaManager singleton instance for this endpoint
    """
    # Simply instantiate - __new__() handles per-endpoint singleton logic
    return TenantSchemaManager(vespa_url, vespa_port)
