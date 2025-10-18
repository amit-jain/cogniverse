"""
ConfigStore Abstract Interface

Defines the interface for configuration storage backends.
Supports multiple implementations: SQLite, Vespa, Elasticsearch, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ConfigScope(Enum):
    """Configuration scope levels"""

    SYSTEM = "system"
    AGENT = "agent"
    ROUTING = "routing"
    TELEMETRY = "telemetry"
    SCHEMA = "schema"


@dataclass
class ConfigEntry:
    """
    Configuration entry with versioning and tenant support

    Attributes:
        tenant_id: Multi-tenant identifier
        scope: Configuration scope (system, agent, routing, etc.)
        service: Service name (e.g., "text_analysis_agent", "video_agent")
        config_key: Configuration key
        config_value: Configuration value as dictionary
        version: Version number (increments on updates)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    tenant_id: str
    scope: ConfigScope
    service: str
    config_key: str
    config_value: Dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime

    def get_config_id(self) -> str:
        """Generate unique config ID: tenant_id:scope:service:config_key"""
        return f"{self.tenant_id}:{self.scope.value}:{self.service}:{self.config_key}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "tenant_id": self.tenant_id,
            "scope": self.scope.value,
            "service": self.service,
            "config_key": self.config_key,
            "config_value": self.config_value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigEntry":
        """Create from dictionary"""
        return cls(
            tenant_id=data["tenant_id"],
            scope=ConfigScope(data["scope"]),
            service=data["service"],
            config_key=data["config_key"],
            config_value=data["config_value"],
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class ConfigStore(ABC):
    """
    Abstract interface for configuration storage

    Implementations:
    - SQLiteConfigStore: Local SQLite database (default)
    - VespaConfigStore: Vespa backend storage
    - ElasticsearchConfigStore: Elasticsearch backend storage (future)
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the configuration store

        Creates necessary tables/schemas/indices for storage.
        """
        pass

    @abstractmethod
    def set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """
        Store or update a configuration entry

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
        pass

    @abstractmethod
    def get_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        version: Optional[int] = None,
    ) -> Optional[ConfigEntry]:
        """
        Retrieve a configuration entry

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            version: Specific version (None = latest)

        Returns:
            ConfigEntry if found, None otherwise
        """
        pass

    @abstractmethod
    def get_config_history(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        limit: int = 10,
    ) -> List[ConfigEntry]:
        """
        Get configuration history (all versions)

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            limit: Maximum number of versions to return

        Returns:
            List of ConfigEntry sorted by version (newest first)
        """
        pass

    @abstractmethod
    def list_configs(
        self,
        tenant_id: str,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        List all configurations matching criteria

        Args:
            tenant_id: Tenant identifier
            scope: Filter by scope (None = all scopes)
            service: Filter by service (None = all services)

        Returns:
            List of latest version ConfigEntry objects
        """
        pass

    @abstractmethod
    def delete_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
    ) -> bool:
        """
        Delete all versions of a configuration entry

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Dictionary with stats (total configs, tenants, versions, etc.)
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if storage backend is healthy

        Returns:
            True if healthy, False otherwise
        """
        pass
