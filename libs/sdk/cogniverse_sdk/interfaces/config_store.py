"""
ConfigStore Abstract Interface

Defines the interface for configuration storage backends.
Supports multiple implementations: SQLite, Vespa, Elasticsearch, etc.
"""

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Hashable, List, Optional


class ConfigScope(Enum):
    """Configuration scope levels"""

    SYSTEM = "system"
    AGENT = "agent"
    ROUTING = "routing"
    TELEMETRY = "telemetry"
    SCHEMA = "schema"
    BACKEND = "backend"
    DURABLE = "durable"


def _utc_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(
            f"ConfigEntry.{field_name} must be a datetime, got {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"ConfigEntry.{field_name} must include timezone information")
    return value.astimezone(timezone.utc)


def _datetime_from_payload(data: Dict[str, Any], field_name: str) -> datetime:
    value = data[field_name]
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be an ISO-8601 string, got {type(value).__name__}"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not valid ISO-8601: {exc}") from None
    canonical = _utc_datetime(parsed, field_name)
    if value != canonical.isoformat():
        raise ValueError(
            f"{field_name} must use canonical UTC ISO-8601 form, got {value!r}"
        )
    return canonical


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

    def __post_init__(self) -> None:
        for field_name in ("tenant_id", "service", "config_key"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(
                    f"ConfigEntry.{field_name} must be a str, "
                    f"got {type(value).__name__}"
                )
        if not isinstance(self.scope, ConfigScope):
            raise TypeError(
                f"ConfigEntry.scope must be a ConfigScope, "
                f"got {type(self.scope).__name__}"
            )
        if not isinstance(self.config_value, dict):
            raise TypeError(
                f"ConfigEntry.config_value must be a dict, "
                f"got {type(self.config_value).__name__}"
            )
        if type(self.version) is not int or self.version < 1:
            raise ValueError(
                f"ConfigEntry.version must be a positive integer, got {self.version!r}"
            )
        self.created_at = _utc_datetime(self.created_at, "created_at")
        self.updated_at = _utc_datetime(self.updated_at, "updated_at")

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
        """Create from dictionary.

        A corrupt stored entry raises ValueError naming the offending
        field instead of a bare KeyError/ValueError with no context.
        """
        try:
            if not isinstance(data, dict):
                raise ValueError(f"payload must be a dict, got {type(data).__name__}")
            expected = {item.name for item in fields(cls)}
            unknown = set(data) - expected
            if unknown:
                raise ValueError(f"unknown fields: {sorted(unknown)}")
            missing = expected - set(data)
            if missing:
                raise ValueError(f"missing fields: {sorted(missing)}")
            return cls(
                tenant_id=data["tenant_id"],
                scope=ConfigScope(data["scope"]),
                service=data["service"],
                config_key=data["config_key"],
                config_value=data["config_value"],
                version=data["version"],
                created_at=_datetime_from_payload(data, "created_at"),
                updated_at=_datetime_from_payload(data, "updated_at"),
            )
        except KeyError as e:
            raise ValueError(
                f"ConfigEntry.from_dict: missing required field {e.args[0]!r}"
            ) from None
        except (TypeError, ValueError) as e:
            raise ValueError(f"ConfigEntry.from_dict: {e}") from None


class ConfigStoreUnavailableError(RuntimeError):
    """The backing store did not answer a read within the implementation's
    retry budget. Raised only for transient failures (connection refused,
    timeouts, 5xx) that persisted across every attempt, or for a read the
    backend answered degraded (partial coverage, which can hide the rows it
    asked for); a clean absence returns None and a non-transient response
    propagates as itself. Callers that wait for the store at startup retry
    on this; nothing else should catch it silently."""


class ConfigWriteConflictError(RuntimeError):
    """A read-modify-write lost its compare-and-set to concurrent writers on
    every attempt. Nothing was written by the call that raised it."""

    def __init__(self, config_id: str, attempts: int):
        super().__init__(
            f"config {config_id} changed under every one of {attempts} "
            "compare-and-set attempts; nothing was written"
        )
        self.config_id = config_id
        self.attempts = attempts


# Read-modify-write attempts before ConfigWriteConflictError, and the full-
# jitter backoff between them: uniform in [0, min(cap, base * 2**(n-1))].
CONFIG_UPDATE_MAX_ATTEMPTS = 10
CONFIG_UPDATE_BACKOFF_BASE_S = 0.05
CONFIG_UPDATE_BACKOFF_CAP_S = 1.0


class ConfigStore(ABC):
    """
    Abstract interface for configuration storage

    Implementations:
    - VespaConfigStore: Vespa backend storage (default)
    """

    @property
    @abstractmethod
    def source(self) -> Hashable:
        """Where this store keeps its entries.

        Two stores with equal sources read and write the same configuration,
        so either can stand in for the other.
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
    def compare_and_set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
        *,
        expected_version: int,
    ) -> Optional[ConfigEntry]:
        """Conditionally append version ``expected_version + 1``.

        Version zero requires an absent key; positive versions must match
        the latest stored revision, including when older history is pruned.
        Competing writers for the same revision cannot both append it.

        Returns the new entry when confirmed current, or None on contention.
        None can also mean a successful write was superseded before its
        confirmation read; callers must reread before retrying. Successful
        writes obey the implementation's history retention policy.

        Raises ValueError for a negative expected_version. Storage failures
        propagate to the caller rather than returning None.
        """
        pass

    def update_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        update: Callable[[Optional[ConfigEntry]], Optional[Dict[str, Any]]],
        *,
        max_attempts: int = CONFIG_UPDATE_MAX_ATTEMPTS,
    ) -> Optional[ConfigEntry]:
        """Read-modify-write one config through ``compare_and_set_config``.

        ``update`` receives the latest entry (None when the key is absent)
        and returns the value to write, or None to leave the config as it
        is. A write that loses to a concurrent writer re-reads the entry and
        calls ``update`` again, so ``update`` must derive its result from the
        entry it is given and nothing it saw on an earlier call.

        Returns the written entry, or the entry ``update`` declined to change
        (None when absent). Raises ConfigWriteConflictError once
        ``max_attempts`` writes have all lost; storage failures and anything
        ``update`` raises propagate, with nothing written.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        for attempt in range(1, max_attempts + 1):
            current = self.get_config(tenant_id, scope, service, config_key)
            value = update(current)
            if value is None:
                return current
            written = self.compare_and_set_config(
                tenant_id,
                scope,
                service,
                config_key,
                value,
                expected_version=0 if current is None else current.version,
            )
            if written is not None:
                return written
            if attempt < max_attempts:
                time.sleep(
                    random.uniform(
                        0,
                        min(
                            CONFIG_UPDATE_BACKOFF_CAP_S,
                            CONFIG_UPDATE_BACKOFF_BASE_S * 2 ** (attempt - 1),
                        ),
                    )
                )
        raise ConfigWriteConflictError(
            f"{tenant_id}:{scope.value}:{service}:{config_key}", max_attempts
        )

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
    def list_all_configs(
        self,
        scope: Optional[ConfigScope] = None,
        service: Optional[str] = None,
        config_key_suffix: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """
        List all configurations across all tenants

        Args:
            scope: Filter by scope (None = all scopes)
            service: Filter by service (None = all services)
            config_key_suffix: Keep only rows whose config_key ends with it
                (None = every key). The store applies it, so a caller after
                one owner's rows within a service never carries the rest back.

        Returns:
            List of latest version ConfigEntry objects from all tenants
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
        Export all configurations for a tenant, without schema-scope rows

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

        A payload carrying a schema-scope row raises ``ValueError`` before
        any write: those rows are written only by the schema registry. A row
        that cannot be written raises ``RuntimeError`` after the versions the
        import already wrote are removed: an import lands whole or not at all.

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


class ImmutableConfigStore(ConfigStore):
    """ConfigStore with immutable version-one records and bounded visits."""

    @abstractmethod
    def put_immutable_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """Write once and confirm; an identical existing value is idempotent.

        A different existing value raises ValueError. A failed write or
        confirmation raises ConfigStoreUnavailableError with the cause.
        """

    @abstractmethod
    def get_immutable_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
    ) -> Optional[ConfigEntry]:
        """Read version one directly; absence is None, outages raise."""

    @abstractmethod
    def list_immutable_configs(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        *,
        page_size: int = 100,
        continuation: Optional[str] = None,
    ) -> tuple[List[ConfigEntry], Optional[str]]:
        """Visit one page of at most ``page_size`` version-one records.

        The continuation is an opaque cursor; following it to None returns
        every record present for the whole scan exactly once. An empty page
        can have a continuation; callers must follow it to exhaust the
        namespace. Concurrent insertions need a subsequent scan.
        """
