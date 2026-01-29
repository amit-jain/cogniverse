"""
Vespa-based configuration storage with multi-tenant support.
Stores configurations directly in Vespa backend for unified storage.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from vespa.application import Vespa

from cogniverse_sdk.interfaces.config_store import (
    ConfigEntry,
    ConfigScope,
    ConfigStore,
)

logger = logging.getLogger(__name__)


class VespaConfigStore(ConfigStore):
    """
    Vespa-based configuration store with multi-tenant support.

    Stores configurations as Vespa documents in a dedicated schema.
    Provides same interface as SQLiteConfigStore but uses Vespa backend.

    Schema: config_metadata
    Document structure:
    {
        "fields": {
            "config_id": "tenant_id:scope:service:config_key",
            "tenant_id": "default",
            "scope": "system",
            "service": "system",
            "config_key": "system_config",
            "config_value": {...},
            "version": 1,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
    }
    """

    def __init__(
        self,
        vespa_app: Optional[Vespa] = None,
        vespa_url: str = "http://localhost",
        vespa_port: int = 8080,
        schema_name: str = "config_metadata",
    ):
        """
        Initialize Vespa configuration store.

        Args:
            vespa_app: Existing Vespa application instance (optional)
            vespa_url: Vespa server URL
            vespa_port: Vespa server port
            schema_name: Vespa schema name for config storage
        """
        if vespa_app is not None:
            self.vespa_app = vespa_app
        else:
            from vespa.application import Vespa

            self.vespa_app = Vespa(url=f"{vespa_url}:{vespa_port}")

        self.schema_name = schema_name
        logger.info(
            f"VespaConfigStore initialized with schema: {schema_name} "
            f"at {vespa_url}:{vespa_port}"
        )

    def initialize(self) -> None:
        """
        Initialize the configuration store.

        For Vespa, this assumes the schema already exists.
        Schema must be deployed separately via vespa-cli or application package.
        """
        # Check if schema exists by attempting a simple query
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            logger.info(f"Vespa schema '{self.schema_name}' is accessible")
        except Exception as e:
            logger.warning(
                f"Could not verify Vespa schema '{self.schema_name}': {e}. "
                "Ensure schema is deployed before using VespaConfigStore."
            )

    def _create_document_id(
        self, tenant_id: str, scope: ConfigScope, service: str, config_key: str
    ) -> str:
        """Create Vespa document ID from config coordinates"""
        # Vespa doc ID: config_metadata::<config_id>::<version>
        config_id = f"{tenant_id}:{scope.value}:{service}:{config_key}"
        return config_id

    def _get_latest_version(
        self, tenant_id: str, scope: ConfigScope, service: str, config_key: str
    ) -> int:
        """Get latest version number for a config"""
        config_id = self._create_document_id(tenant_id, scope, service, config_key)

        # Query for all versions of this config
        # Use contains() for indexed string matching (avoids YQL colon parsing issues)
        yql = (
            f"select version from {self.schema_name} "
            f'where config_id contains "{config_id}" '
            f"order by version desc limit 1"
        )

        try:
            response = self.vespa_app.query(yql=yql)
            if response.hits and len(response.hits) > 0:
                return response.hits[0]["fields"]["version"]
            return 0
        except Exception as e:
            logger.warning(f"Could not query latest version: {e}")
            return 0

    def set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """
        Store or update a configuration entry.

        Creates a new version of the config. All updates are versioned.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            config_value: Configuration value (dict)

        Returns:
            ConfigEntry with new version number
        """
        # Get next version
        current_version = self._get_latest_version(
            tenant_id, scope, service, config_key
        )
        new_version = current_version + 1

        # Create timestamps
        now = datetime.now()
        created_at = now if new_version == 1 else None  # Only set on first version
        updated_at = now

        # Create config entry
        entry = ConfigEntry(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
            config_value=config_value,
            version=new_version,
            created_at=created_at or now,
            updated_at=updated_at,
        )

        # Create Vespa document ID
        config_id = self._create_document_id(tenant_id, scope, service, config_key)
        doc_id = f"{self.schema_name}::{config_id}::{new_version}"

        # Prepare document fields
        fields = {
            "config_id": config_id,
            "tenant_id": tenant_id,
            "scope": scope.value,
            "service": service,
            "config_key": config_key,
            "config_value": json.dumps(config_value),
            "version": new_version,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
        }

        # Feed document to Vespa
        try:
            self.vespa_app.feed_data_point(
                schema=self.schema_name,
                data_id=doc_id,
                fields=fields,
            )

            logger.info(f"Set config {entry.get_config_id()} v{new_version} in Vespa")
            return entry

        except Exception as e:
            logger.error(f"Failed to store config in Vespa: {e}")
            raise

    def get_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        version: Optional[int] = None,
    ) -> Optional[ConfigEntry]:
        """
        Retrieve a configuration entry.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            version: Specific version (None = latest)

        Returns:
            ConfigEntry if found, None otherwise
        """
        config_id = self._create_document_id(tenant_id, scope, service, config_key)

        # Build YQL query
        # Use contains() for indexed string matching (avoids YQL colon parsing issues)
        if version is None:
            # Get latest version
            yql = (
                f"select * from {self.schema_name} "
                f'where config_id contains "{config_id}" '
                f"order by version desc limit 1"
            )
        else:
            # Get specific version
            doc_id = f"{self.schema_name}::{config_id}::{version}"
            yql = f'select * from {self.schema_name} where documentid = "{doc_id}"'

        try:
            response = self.vespa_app.query(yql=yql)

            if not response.hits or len(response.hits) == 0:
                return None

            # Parse first hit
            hit = response.hits[0]["fields"]

            return ConfigEntry(
                tenant_id=hit["tenant_id"],
                scope=ConfigScope(hit["scope"]),
                service=hit["service"],
                config_key=hit["config_key"],
                config_value=json.loads(hit["config_value"]),
                version=hit["version"],
                created_at=datetime.fromisoformat(hit["created_at"]),
                updated_at=datetime.fromisoformat(hit["updated_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to retrieve config from Vespa: {e}")
            return None

    def get_config_history(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        limit: int = 10,
    ) -> List[ConfigEntry]:
        """
        Get configuration history (all versions).

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            limit: Maximum number of versions to return

        Returns:
            List of ConfigEntry sorted by version (newest first)
        """
        config_id = self._create_document_id(tenant_id, scope, service, config_key)

        yql = (
            f"select * from {self.schema_name} "
            f'where config_id contains "{config_id}" '
            f"order by version desc limit {limit}"
        )

        try:
            response = self.vespa_app.query(yql=yql)

            entries = []
            for hit in response.hits:
                fields = hit["fields"]
                entries.append(
                    ConfigEntry(
                        tenant_id=fields["tenant_id"],
                        scope=ConfigScope(fields["scope"]),
                        service=fields["service"],
                        config_key=fields["config_key"],
                        config_value=json.loads(fields["config_value"]),
                        version=fields["version"],
                        created_at=datetime.fromisoformat(fields["created_at"]),
                        updated_at=datetime.fromisoformat(fields["updated_at"]),
                    )
                )

            return entries

        except Exception as e:
            logger.error(f"Failed to retrieve config history from Vespa: {e}")
            return []

    def list_configs(
        self,
        tenant_id: str,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        List all configurations matching criteria.

        Returns only latest versions.

        Args:
            tenant_id: Tenant identifier
            scope: Filter by scope (None = all scopes)
            service: Filter by service (None = all services)

        Returns:
            List of latest version ConfigEntry objects
        """
        # Build YQL query with filters
        # Use contains() for indexed string matching (avoids YQL colon parsing issues)
        conditions = [f'tenant_id contains "{tenant_id}"']

        if scope is not None:
            conditions.append(f'scope contains "{scope.value}"')

        if service is not None:
            conditions.append(f'service contains "{service}"')

        where_clause = " and ".join(conditions)

        # Query all matching configs, then filter to latest versions
        # Note: This is a simplified approach - for production, consider using
        # Vespa grouping or ranking to get only latest versions efficiently
        yql = f"select * from {self.schema_name} where {where_clause} limit 400"

        try:
            response = self.vespa_app.query(yql=yql)

            # Group by config_id and keep only latest version
            latest_configs: Dict[str, ConfigEntry] = {}

            for hit in response.hits:
                fields = hit["fields"]
                config_id = fields["config_id"]

                entry = ConfigEntry(
                    tenant_id=fields["tenant_id"],
                    scope=ConfigScope(fields["scope"]),
                    service=fields["service"],
                    config_key=fields["config_key"],
                    config_value=json.loads(fields["config_value"]),
                    version=fields["version"],
                    created_at=datetime.fromisoformat(fields["created_at"]),
                    updated_at=datetime.fromisoformat(fields["updated_at"]),
                )

                # Keep only latest version for each config_id
                if (
                    config_id not in latest_configs
                    or entry.version > latest_configs[config_id].version
                ):
                    latest_configs[config_id] = entry

            return list(latest_configs.values())

        except Exception as e:
            logger.error(f"Failed to list configs from Vespa: {e}")
            return []

    def list_all_configs(
        self,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        List all configurations across all tenants.

        Returns only latest versions.

        Args:
            scope: Filter by scope (None = all scopes)
            service: Filter by service (None = all services)

        Returns:
            List of latest version ConfigEntry objects from all tenants
        """
        # Build YQL query with filters (no tenant_id filter)
        # Use contains() for indexed string matching (avoids YQL colon parsing issues)
        conditions = []

        if scope is not None:
            conditions.append(f'scope contains "{scope.value}"')

        if service is not None:
            conditions.append(f'service contains "{service}"')

        where_clause = " and ".join(conditions) if conditions else "true"

        # Query all matching configs, then filter to latest versions
        yql = f"select * from {self.schema_name} where {where_clause} limit 400"

        try:
            response = self.vespa_app.query(yql=yql)

            # Group by config_id and keep only latest version
            latest_configs: Dict[str, ConfigEntry] = {}

            for hit in response.hits:
                fields = hit["fields"]
                config_id = fields["config_id"]

                entry = ConfigEntry(
                    tenant_id=fields["tenant_id"],
                    scope=ConfigScope(fields["scope"]),
                    service=fields["service"],
                    config_key=fields["config_key"],
                    config_value=json.loads(fields["config_value"]),
                    version=fields["version"],
                    created_at=datetime.fromisoformat(fields["created_at"]),
                    updated_at=datetime.fromisoformat(fields["updated_at"]),
                )

                # Keep only latest version for each config_id
                if (
                    config_id not in latest_configs
                    or entry.version > latest_configs[config_id].version
                ):
                    latest_configs[config_id] = entry

            return list(latest_configs.values())

        except Exception as e:
            logger.error(f"Failed to list all configs from Vespa: {e}")
            return []

    def delete_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
    ) -> bool:
        """
        Delete all versions of a configuration entry.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key

        Returns:
            True if deleted, False if not found
        """
        config_id = self._create_document_id(tenant_id, scope, service, config_key)

        # Get all versions
        history = self.get_config_history(
            tenant_id, scope, service, config_key, limit=1000
        )

        if not history:
            return False

        # Delete each version
        deleted_count = 0
        for entry in history:
            doc_id = f"{self.schema_name}::{config_id}::{entry.version}"
            try:
                self.vespa_app.delete_data(schema=self.schema_name, data_id=doc_id)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete version {entry.version}: {e}")

        logger.info(
            f"Deleted {deleted_count} versions of config "
            f"{tenant_id}:{scope.value}:{service}:{config_key}"
        )

        return deleted_count > 0

    def export_configs(
        self,
        tenant_id: str,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Export all configurations for a tenant.

        Args:
            tenant_id: Tenant identifier
            include_history: Include all versions (True) or just latest (False)

        Returns:
            Dictionary with all configurations
        """
        if include_history:
            # Get all versions
            yql = f'select * from {self.schema_name} where tenant_id contains "{tenant_id}" limit 400'
        else:
            # Get only latest versions
            configs = self.list_configs(tenant_id)
            return {
                "tenant_id": tenant_id,
                "include_history": False,
                "configs": [
                    {
                        "tenant_id": c.tenant_id,
                        "scope": c.scope.value,
                        "service": c.service,
                        "config_key": c.config_key,
                        "config_value": c.config_value,
                        "version": c.version,
                        "created_at": c.created_at.isoformat(),
                        "updated_at": c.updated_at.isoformat(),
                    }
                    for c in configs
                ],
                "exported_at": datetime.now().isoformat(),
            }

        try:
            response = self.vespa_app.query(yql=yql)

            configs = []
            for hit in response.hits:
                fields = hit["fields"]
                configs.append(
                    {
                        "tenant_id": fields["tenant_id"],
                        "scope": fields["scope"],
                        "service": fields["service"],
                        "config_key": fields["config_key"],
                        "config_value": json.loads(fields["config_value"]),
                        "version": fields["version"],
                        "created_at": fields["created_at"],
                        "updated_at": fields["updated_at"],
                    }
                )

            return {
                "tenant_id": tenant_id,
                "include_history": True,
                "configs": configs,
                "exported_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to export configs from Vespa: {e}")
            return {
                "tenant_id": tenant_id,
                "include_history": include_history,
                "configs": [],
                "exported_at": datetime.now().isoformat(),
                "error": str(e),
            }

    def import_configs(
        self,
        tenant_id: str,
        configs: Dict[str, Any],
    ) -> int:
        """
        Import configurations for a tenant.

        Args:
            tenant_id: Tenant identifier
            configs: Dictionary of configurations to import

        Returns:
            Number of configurations imported
        """
        imported_count = 0

        for config_data in configs.get("configs", []):
            try:
                self.set_config(
                    tenant_id=tenant_id,
                    scope=ConfigScope(config_data["scope"]),
                    service=config_data["service"],
                    config_key=config_data["config_key"],
                    config_value=config_data["config_value"],
                )
                imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import config: {e}")

        logger.info(f"Imported {imported_count} configs for tenant {tenant_id}")
        return imported_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            # Select all fields needed for stats
            yql_total = f"select config_id, tenant_id, scope from {self.schema_name} where true limit 400"
            response = self.vespa_app.query(yql=yql_total)

            total_versions = len(response.hits)
            unique_config_ids = len(
                set(hit["fields"]["config_id"] for hit in response.hits)
            )

            # Count tenants
            unique_tenants = len(
                set(hit["fields"]["tenant_id"] for hit in response.hits)
            )

            # Count per scope
            scope_counts: Dict[str, int] = {}
            for hit in response.hits:
                scope = hit["fields"]["scope"]
                scope_counts[scope] = scope_counts.get(scope, 0) + 1

            return {
                "total_configs": unique_config_ids,
                "total_versions": total_versions,
                "total_tenants": unique_tenants,
                "configs_per_scope": scope_counts,
                "storage_backend": "vespa",
                "schema_name": self.schema_name,
            }

        except Exception as e:
            logger.error(f"Failed to get stats from Vespa: {e}")
            return {
                "total_configs": 0,
                "total_versions": 0,
                "total_tenants": 0,
                "configs_per_scope": {},
                "storage_backend": "vespa",
                "schema_name": self.schema_name,
                "error": str(e),
            }

    def health_check(self) -> bool:
        """
        Check if storage backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            return True
        except Exception as e:
            logger.error(f"Vespa health check failed: {e}")
            return False
