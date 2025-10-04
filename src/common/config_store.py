"""
SQLite-based configuration storage with multi-tenant support.
Provides versioned configuration persistence with history tracking.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.config_store_interface import ConfigEntry, ConfigScope, ConfigStore

logger = logging.getLogger(__name__)


class SQLiteConfigStore(ConfigStore):
    """
    SQLite-based configuration store with multi-tenant support.

    Schema:
        configurations (
            tenant_id TEXT,
            scope TEXT,
            service TEXT,
            config_key TEXT,
            config_value JSON,
            version INTEGER,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            PRIMARY KEY (tenant_id, scope, service, config_key, version)
        )
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize configuration store.

        Args:
            db_path: Path to SQLite database file. Defaults to data/config/config.db
        """
        if db_path is None:
            db_path = Path("data/config/config.db")

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.initialize()
        logger.info(f"SQLiteConfigStore initialized at {self.db_path}")

    def initialize(self) -> None:
        """Initialize the configuration store (implements ConfigStore interface)"""
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create configurations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS configurations (
                    tenant_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    service TEXT NOT NULL,
                    config_key TEXT NOT NULL,
                    config_value TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, scope, service, config_key, version)
                )
                """
            )

            # Create indexes for fast lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tenant_service
                ON configurations(tenant_id, service)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tenant_scope
                ON configurations(tenant_id, scope)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_config_id
                ON configurations(tenant_id, scope, service, config_key)
                """
            )

            # Create current_version view for easy access to latest configs
            cursor.execute(
                """
                CREATE VIEW IF NOT EXISTS current_configurations AS
                SELECT
                    tenant_id,
                    scope,
                    service,
                    config_key,
                    config_value,
                    version,
                    created_at,
                    updated_at
                FROM configurations c1
                WHERE version = (
                    SELECT MAX(version)
                    FROM configurations c2
                    WHERE c1.tenant_id = c2.tenant_id
                        AND c1.scope = c2.scope
                        AND c1.service = c2.service
                        AND c1.config_key = c2.config_key
                )
                """
            )

            conn.commit()

        logger.info("Database schema initialized")

    def set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """
        Set configuration value (creates new version).

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            config_value: Configuration value as dictionary

        Returns:
            ConfigEntry with new version
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get current version
            cursor.execute(
                """
                SELECT MAX(version) FROM configurations
                WHERE tenant_id = ? AND scope = ? AND service = ? AND config_key = ?
                """,
                (tenant_id, scope.value, service, config_key),
            )
            result = cursor.fetchone()
            current_version = result[0] if result[0] is not None else 0
            new_version = current_version + 1

            # Insert new version
            now = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO configurations
                (tenant_id, scope, service, config_key, config_value, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    scope.value,
                    service,
                    config_key,
                    json.dumps(config_value),
                    new_version,
                    now,
                    now,
                ),
            )

            conn.commit()

        entry = ConfigEntry(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
            config_value=config_value,
            version=new_version,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )

        logger.info(f"Set config {entry.get_config_id()} v{new_version}")
        return entry

    def get_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        version: Optional[int] = None,
    ) -> Optional[ConfigEntry]:
        """
        Get configuration value.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            version: Specific version to retrieve (None for latest)

        Returns:
            ConfigEntry or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if version is None:
                # Get latest version
                cursor.execute(
                    """
                    SELECT tenant_id, scope, service, config_key, config_value,
                           version, created_at, updated_at
                    FROM current_configurations
                    WHERE tenant_id = ? AND scope = ? AND service = ? AND config_key = ?
                    """,
                    (tenant_id, scope.value, service, config_key),
                )
            else:
                # Get specific version
                cursor.execute(
                    """
                    SELECT tenant_id, scope, service, config_key, config_value,
                           version, created_at, updated_at
                    FROM configurations
                    WHERE tenant_id = ? AND scope = ? AND service = ?
                          AND config_key = ? AND version = ?
                    """,
                    (tenant_id, scope.value, service, config_key, version),
                )

            row = cursor.fetchone()
            if not row:
                return None

            return ConfigEntry(
                tenant_id=row[0],
                scope=ConfigScope(row[1]),
                service=row[2],
                config_key=row[3],
                config_value=json.loads(row[4]),
                version=row[5],
                created_at=datetime.fromisoformat(row[6]),
                updated_at=datetime.fromisoformat(row[7]),
            )

    def get_config_history(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        limit: int = 10,
    ) -> List[ConfigEntry]:
        """
        Get configuration history.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            limit: Maximum number of versions to return

        Returns:
            List of ConfigEntry ordered by version descending
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT tenant_id, scope, service, config_key, config_value,
                       version, created_at, updated_at
                FROM configurations
                WHERE tenant_id = ? AND scope = ? AND service = ? AND config_key = ?
                ORDER BY version DESC
                LIMIT ?
                """,
                (tenant_id, scope.value, service, config_key, limit),
            )

            entries = []
            for row in cursor.fetchall():
                entries.append(
                    ConfigEntry(
                        tenant_id=row[0],
                        scope=ConfigScope(row[1]),
                        service=row[2],
                        config_key=row[3],
                        config_value=json.loads(row[4]),
                        version=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                    )
                )

            return entries

    def list_configs(
        self,
        tenant_id: str,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        Get all current configurations for a tenant.

        Args:
            tenant_id: Tenant identifier
            scope: Optional scope filter
            service: Optional service filter

        Returns:
            List of current ConfigEntry objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Build query based on filters
            if scope is None and service is None:
                cursor.execute(
                    """
                    SELECT tenant_id, scope, service, config_key, config_value,
                           version, created_at, updated_at
                    FROM current_configurations
                    WHERE tenant_id = ?
                    ORDER BY scope, service, config_key
                    """,
                    (tenant_id,),
                )
            elif scope is not None and service is None:
                cursor.execute(
                    """
                    SELECT tenant_id, scope, service, config_key, config_value,
                           version, created_at, updated_at
                    FROM current_configurations
                    WHERE tenant_id = ? AND scope = ?
                    ORDER BY service, config_key
                    """,
                    (tenant_id, scope.value),
                )
            elif scope is None and service is not None:
                cursor.execute(
                    """
                    SELECT tenant_id, scope, service, config_key, config_value,
                           version, created_at, updated_at
                    FROM current_configurations
                    WHERE tenant_id = ? AND service = ?
                    ORDER BY scope, config_key
                    """,
                    (tenant_id, service),
                )
            else:
                cursor.execute(
                    """
                    SELECT tenant_id, scope, service, config_key, config_value,
                           version, created_at, updated_at
                    FROM current_configurations
                    WHERE tenant_id = ? AND scope = ? AND service = ?
                    ORDER BY config_key
                    """,
                    (tenant_id, scope.value, service),
                )

            entries = []
            for row in cursor.fetchall():
                entries.append(
                    ConfigEntry(
                        tenant_id=row[0],
                        scope=ConfigScope(row[1]),
                        service=row[2],
                        config_key=row[3],
                        config_value=json.loads(row[4]),
                        version=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                    )
                )

            return entries

    def delete_config(
        self, tenant_id: str, scope: ConfigScope, service: str, config_key: str
    ) -> bool:
        """
        Delete all versions of a configuration.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM configurations
                WHERE tenant_id = ? AND scope = ? AND service = ? AND config_key = ?
                """,
                (tenant_id, scope.value, service, config_key),
            )

            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted:
            logger.info(
                f"Deleted config {tenant_id}:{scope.value}:{service}:{config_key}"
            )

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total configs
            cursor.execute(
                "SELECT COUNT(DISTINCT tenant_id || scope || service || config_key) FROM configurations"
            )
            total_configs = cursor.fetchone()[0]

            # Total versions
            cursor.execute("SELECT COUNT(*) FROM configurations")
            total_versions = cursor.fetchone()[0]

            # Tenants
            cursor.execute("SELECT COUNT(DISTINCT tenant_id) FROM configurations")
            total_tenants = cursor.fetchone()[0]

            # Configs per scope
            cursor.execute(
                """
                SELECT scope, COUNT(DISTINCT tenant_id || service || config_key)
                FROM configurations
                GROUP BY scope
                """
            )
            configs_per_scope = {row[0]: row[1] for row in cursor.fetchall()}

            # Database size
            db_size_bytes = self.db_path.stat().st_size

            return {
                "total_configs": total_configs,
                "total_versions": total_versions,
                "total_tenants": total_tenants,
                "configs_per_scope": configs_per_scope,
                "db_size_bytes": db_size_bytes,
                "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
                "db_path": str(self.db_path),
            }

    def export_configs(
        self,
        tenant_id: str,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Export all configurations for a tenant

        Args:
            tenant_id: Tenant identifier
            include_history: Include all versions (True) or just latest (False)

        Returns:
            Dictionary with all configurations
        """
        if include_history:
            # Get all versions
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT tenant_id, scope, service, config_key, config_value,
                           version, created_at, updated_at
                    FROM configurations
                    WHERE tenant_id = ?
                    ORDER BY scope, service, config_key, version
                    """,
                    (tenant_id,),
                )

                configs = []
                for row in cursor.fetchall():
                    configs.append(
                        {
                            "tenant_id": row[0],
                            "scope": row[1],
                            "service": row[2],
                            "config_key": row[3],
                            "config_value": json.loads(row[4]),
                            "version": row[5],
                            "created_at": row[6],
                            "updated_at": row[7],
                        }
                    )

                return {
                    "tenant_id": tenant_id,
                    "include_history": True,
                    "configs": configs,
                    "exported_at": datetime.now().isoformat(),
                }
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

    def import_configs(
        self,
        tenant_id: str,
        configs: Dict[str, Any],
    ) -> int:
        """
        Import configurations for a tenant

        Args:
            tenant_id: Tenant identifier
            configs: Dictionary of configurations to import

        Returns:
            Number of configurations imported
        """
        imported_count = 0

        for config_data in configs.get("configs", []):
            # Import each config
            self.set_config(
                tenant_id=tenant_id,
                scope=ConfigScope(config_data["scope"]),
                service=config_data["service"],
                config_key=config_data["config_key"],
                config_value=config_data["config_value"],
            )
            imported_count += 1

        logger.info(f"Imported {imported_count} configs for tenant {tenant_id}")
        return imported_count

    def health_check(self) -> bool:
        """
        Check if storage backend is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return cursor.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
