"""
In-memory implementation of ConfigStore for unit testing.

Provides a simple dict-based storage that doesn't require
any external backend (Vespa, SQLite, etc.).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from cogniverse_sdk.interfaces.config_store import (
    ConfigEntry,
    ConfigScope,
    ConfigStore,
)


class InMemoryConfigStore(ConfigStore):
    """
    In-memory ConfigStore implementation for unit tests.

    Stores all configurations in memory using dictionaries.
    No persistence - data is lost when the instance is destroyed.
    """

    def __init__(self):
        # Storage: {config_id: {version: ConfigEntry}}
        self._storage: Dict[str, Dict[int, ConfigEntry]] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the in-memory store."""
        self._initialized = True

    def set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """Store or update a configuration entry."""
        config_id = f"{tenant_id}:{scope.value}:{service}:{config_key}"
        now = datetime.utcnow()

        # Get next version number
        if config_id in self._storage:
            versions = self._storage[config_id]
            next_version = max(versions.keys()) + 1
            created_at = min(v.created_at for v in versions.values())
        else:
            self._storage[config_id] = {}
            next_version = 1
            created_at = now

        entry = ConfigEntry(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
            config_value=config_value,
            version=next_version,
            created_at=created_at,
            updated_at=now,
        )

        self._storage[config_id][next_version] = entry
        return entry

    def get_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        version: Optional[int] = None,
    ) -> Optional[ConfigEntry]:
        """Retrieve a configuration entry."""
        config_id = f"{tenant_id}:{scope.value}:{service}:{config_key}"

        if config_id not in self._storage:
            return None

        versions = self._storage[config_id]
        if not versions:
            return None

        if version is not None:
            return versions.get(version)
        else:
            # Return latest version
            latest_version = max(versions.keys())
            return versions[latest_version]

    def get_config_history(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        limit: int = 10,
    ) -> List[ConfigEntry]:
        """Get configuration history (all versions)."""
        config_id = f"{tenant_id}:{scope.value}:{service}:{config_key}"

        if config_id not in self._storage:
            return []

        versions = self._storage[config_id]
        sorted_entries = sorted(
            versions.values(),
            key=lambda e: e.version,
            reverse=True,
        )
        return sorted_entries[:limit]

    def list_configs(
        self,
        tenant_id: str,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """List all configurations matching criteria."""
        results = []

        for config_id, versions in self._storage.items():
            if not versions:
                continue

            latest_version = max(versions.keys())
            entry = versions[latest_version]

            # Filter by tenant
            if entry.tenant_id != tenant_id:
                continue

            # Filter by scope
            if scope is not None and entry.scope != scope:
                continue

            # Filter by service
            if service is not None and entry.service != service:
                continue

            results.append(entry)

        return results

    def list_all_configs(
        self,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """List all configurations across all tenants."""
        results = []

        for config_id, versions in self._storage.items():
            if not versions:
                continue

            latest_version = max(versions.keys())
            entry = versions[latest_version]

            # Filter by scope
            if scope is not None and entry.scope != scope:
                continue

            # Filter by service
            if service is not None and entry.service != service:
                continue

            results.append(entry)

        return results

    def delete_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
    ) -> bool:
        """Delete all versions of a configuration entry."""
        config_id = f"{tenant_id}:{scope.value}:{service}:{config_key}"

        if config_id in self._storage:
            del self._storage[config_id]
            return True
        return False

    def export_configs(
        self,
        tenant_id: str,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        """Export all configurations for a tenant."""
        result = {"tenant_id": tenant_id, "configs": []}

        for config_id, versions in self._storage.items():
            if not versions:
                continue

            # Check if any version belongs to this tenant
            sample = next(iter(versions.values()))
            if sample.tenant_id != tenant_id:
                continue

            if include_history:
                for entry in versions.values():
                    result["configs"].append(entry.to_dict())
            else:
                latest_version = max(versions.keys())
                result["configs"].append(versions[latest_version].to_dict())

        return result

    def import_configs(
        self,
        tenant_id: str,
        configs: Dict[str, Any],
    ) -> int:
        """Import configurations for a tenant."""
        count = 0
        for config_data in configs.get("configs", []):
            entry = ConfigEntry.from_dict(config_data)
            config_id = entry.get_config_id()

            if config_id not in self._storage:
                self._storage[config_id] = {}

            self._storage[config_id][entry.version] = entry
            count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_entries = sum(len(v) for v in self._storage.values())
        tenants = set()
        for versions in self._storage.values():
            for entry in versions.values():
                tenants.add(entry.tenant_id)

        return {
            "total_configs": len(self._storage),
            "total_versions": total_entries,
            "tenants": len(tenants),
            "backend": "memory",
        }

    def health_check(self) -> bool:
        """Check if storage is healthy."""
        return self._initialized
