"""
Centralized configuration manager with multi-tenant support.
Provides unified interface for all configuration operations with caching.
"""

import copy
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cogniverse_foundation.common.tenant_utils import (
    SYSTEM_TENANT_ID,
    require_tenant_id,
)
from cogniverse_foundation.config.agent_config import AgentConfig
from cogniverse_foundation.config.unified_config import (
    AgentConfigUnified,
    BackendConfig,
    BackendProfileConfig,
    DurableExecutionConfig,
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_sdk.interfaces.config_store import ConfigScope, ConfigStore

logger = logging.getLogger(__name__)


# Listener signature: (event_type, profile_name, profile_config_or_none)
# event_type is "added" or "removed". profile_config is a dict when added,
# None when removed. Listeners must not raise — we log and swallow.
ProfileChangeListener = Callable[[str, str, Optional[Dict[str, Any]]], None]


class ConfigManager:
    """
    Centralized configuration manager with multi-tenant support and caching.

    Provides unified interface for:
    - System configuration (agent URLs, backends, infrastructure)
    - Agent configuration (DSPy modules, optimizers, LLM settings)
    - Routing configuration (tiers, strategies, optimization)
    - Telemetry configuration (Phoenix, metrics, tracing)

    All configurations are:
    - Versioned (full history tracking)
    - Tenant-scoped (multi-tenant ready)
    - Persisted through a pluggable ConfigStore (default: VespaConfigStore)
    - Backed by an in-process cache of the system config (hot path)
    """

    def __init__(
        self,
        store: ConfigStore,
        profile_change_listener: Optional[ProfileChangeListener] = None,
        scoped_config_cache_ttl_s: float = 5.0,
    ):
        """
        Initialize configuration manager with required ConfigStore.

        Args:
            store: ConfigStore implementation (REQUIRED, no fallback)
            profile_change_listener: Optional callable invoked when a
                backend profile is added or removed. Signature
                ``(event_type, profile_name, profile_config_or_none)``
                where event_type is ``"added"`` or ``"removed"``. Used by
                the runtime to propagate profile changes from the
                ConfigStore into live `VespaSearchBackend` instances via
                `BackendRegistry.add_profile_to_backends`. Kept as a
                callback rather than a direct import so the foundation
                layer doesn't depend on core.
            scoped_config_cache_ttl_s: How long per-tenant scoped configs
                (routing/telemetry/backend) are served from memory before
                the store is consulted again. Same-manager setters
                invalidate immediately; the TTL bounds staleness for
                writes made by OTHER processes (another pod's admin API).
                Set to 0 to disable.

        Raises:
            ValueError: If store is None
        """
        if store is None:
            raise ValueError("store is required")

        self.store = store
        self._backend_lock = threading.Lock()
        self._profile_change_listener = profile_change_listener
        # System config doesn't change after the runtime applies its env
        # overrides at startup, but `get_system_config` is hot — every
        # `create_dspy_lm` calls it. Without a cache each call hits the
        # backend (Vespa query timeout in tests where Vespa isn't
        # reachable burns ~20s per test). The cache is busted by
        # `set_system_config` so live updates still propagate.
        self._system_config_cache: Optional[SystemConfig] = None
        # Per-tenant scoped configs (routing/telemetry/backend) are read on
        # every request via ConfigUtils' ensure cascade — each read was a
        # separate store round-trip (a YQL query against Vespa). Cache the
        # raw config_value per (scope, tenant, service, key) under a short
        # TTL; entries are deep-copied on the way out so callers can't
        # mutate shared state.
        self._scoped_config_cache_ttl_s = scoped_config_cache_ttl_s
        self._scoped_config_cache: Dict[tuple, tuple[float, Optional[dict]]] = {}
        self._scoped_config_lock = threading.Lock()

        logger.info(
            "ConfigManager initialized with %s, profile_change_listener=%s",
            type(self.store).__name__,
            "set" if profile_change_listener else "unset",
        )

    def set_profile_change_listener(
        self, listener: Optional[ProfileChangeListener]
    ) -> None:
        """Install or replace the profile-change listener.

        Allows late wiring — the listener can be set after construction
        (e.g., during runtime startup once `BackendRegistry` is ready).
        """
        self._profile_change_listener = listener

    def _notify_profile_change(
        self,
        event_type: str,
        profile_name: str,
        profile_config: Optional[Dict[str, Any]],
    ) -> None:
        """Invoke the profile-change listener, swallowing any error."""
        listener = self._profile_change_listener
        if listener is None:
            return
        try:
            listener(event_type, profile_name, profile_config)
        except Exception as exc:
            logger.warning(
                "profile_change_listener(%s, %s) raised: %s",
                event_type,
                profile_name,
                exc,
            )

    # ========== System Configuration ==========

    # Fixed sentinel tenant_id for system-wide config storage.
    # SystemConfig is global — not per-tenant.
    _SYSTEM_TENANT_ID = "_system"

    def get_system_config(self) -> SystemConfig:
        """Get system-wide infrastructure configuration.

        Cached on the instance after the first call — `set_system_config`
        is the only path that writes, and it invalidates the cache.

        Returns:
            SystemConfig instance
        """
        # Return a copy so a caller mutating a field (the get-modify-set path,
        # or a nested dict) cannot poison the shared cache other callers read.
        # The cache still saves the expensive store round-trip; the copy of a
        # small dataclass is cheap by comparison.
        if self._system_config_cache is not None:
            return copy.deepcopy(self._system_config_cache)

        entry = self.store.get_config(
            tenant_id=self._SYSTEM_TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
        )

        if entry is None:
            logger.warning("No system config found, using defaults")
            cfg = SystemConfig()
        else:
            cfg = SystemConfig.from_dict(entry.config_value)

        self._system_config_cache = cfg
        return copy.deepcopy(cfg)

    def set_system_config(self, system_config: SystemConfig) -> SystemConfig:
        """Set system-wide infrastructure configuration.

        Args:
            system_config: SystemConfig instance

        Returns:
            Updated SystemConfig
        """
        self.store.set_config(
            tenant_id=self._SYSTEM_TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
            config_value=system_config.to_dict(),
        )
        # Bust the get_system_config cache so the new write is visible
        # on the next read in this process.
        self._system_config_cache = system_config

        logger.info("System config updated")
        return system_config

    # ========== Agent Configuration ==========

    def get_agent_config(
        self, tenant_id: str, agent_name: str
    ) -> Optional[AgentConfig]:
        """
        Get agent configuration.

        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name

        Returns:
            AgentConfig or None if not found
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_agent_config"
        )
        # Served from the scoped TTL cache — this read sits on the
        # per-dispatch answer path (behavior toggles for every summarizer /
        # report dispatch), so an uncached read cost one synchronous Vespa
        # query per dispatch while the sibling scopes were cached.
        value = self._cached_config_value(
            ConfigScope.AGENT, tenant_id, agent_name, "agent_config"
        )

        if value is None:
            return None

        unified = AgentConfigUnified.from_dict(value)
        return unified.agent_config

    def set_agent_config(
        self, tenant_id: str, agent_name: str, agent_config: AgentConfig
    ) -> AgentConfig:
        """
        Set agent configuration.

        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name
            agent_config: AgentConfig instance

        Returns:
            Updated AgentConfig
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.set_agent_config"
        )
        unified = AgentConfigUnified(tenant_id=tenant_id, agent_config=agent_config)

        self.store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.AGENT,
            service=agent_name,
            config_key="agent_config",
            # Persist the real key, not the display-redacted "***".
            config_value=unified.to_dict(redact=False),
        )
        self._invalidate_scoped_config(ConfigScope.AGENT, tenant_id)

        logger.info(f"Set agent config for {tenant_id}:{agent_name}")
        return agent_config

    def get_agent_config_history(
        self, tenant_id: str, agent_name: str, limit: int = 10
    ) -> List[AgentConfig]:
        """
        Get agent configuration history.

        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name
            limit: Maximum number of versions

        Returns:
            List of AgentConfig ordered by version descending
        """
        entries = self.store.get_config_history(
            tenant_id=tenant_id,
            scope=ConfigScope.AGENT,
            service=agent_name,
            config_key="agent_config",
            limit=limit,
        )

        configs = []
        for entry in entries:
            unified = AgentConfigUnified.from_dict(entry.config_value)
            configs.append(unified.agent_config)

        return configs

    # ========== Scoped-config cache ==========

    def _cached_config_value(
        self, scope: ConfigScope, tenant_id: str, service: str, config_key: str
    ) -> Optional[dict]:
        """Return the raw ``config_value`` for a scoped config, served from
        the TTL cache when fresh. ``None`` (config absent) is cached too, so
        tenants without overrides don't re-query the store per request."""
        key = (scope, tenant_id, service, config_key)
        now = time.monotonic()
        with self._scoped_config_lock:
            hit = self._scoped_config_cache.get(key)
            if hit is not None and now - hit[0] < self._scoped_config_cache_ttl_s:
                return copy.deepcopy(hit[1])
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
        )
        value = entry.config_value if entry is not None else None
        with self._scoped_config_lock:
            self._scoped_config_cache[key] = (now, value)
        return copy.deepcopy(value)

    def _invalidate_scoped_config(self, scope: ConfigScope, tenant_id: str) -> None:
        """Drop cached entries for a (scope, tenant) after a write."""
        with self._scoped_config_lock:
            for key in [
                k
                for k in self._scoped_config_cache
                if k[0] == scope and k[1] == tenant_id
            ]:
                del self._scoped_config_cache[key]

    # ========== Tenant Instructions ==========

    def get_tenant_instructions_config(self, tenant_id: str) -> Optional[Any]:
        """Get the raw tenant-instructions value (the SOUL.md equivalent).

        Served from the scoped TTL cache — every memory-aware agent reads
        the instructions on the per-dispatch enrichment path, so an uncached
        read cost one synchronous store query per dispatch while the sibling
        scopes were cached. Returns the stored ``config_value`` (typically
        ``{"text": ..., "updated_at": ...}``) or ``None`` when unset.
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_tenant_instructions_config"
        )
        return self._cached_config_value(
            ConfigScope.SYSTEM, tenant_id, "tenant_instructions", "system_prompt"
        )

    # ========== Routing Configuration ==========

    def get_routing_config(
        self, tenant_id: str = None, service: str = "gateway_agent"
    ) -> RoutingConfigUnified:
        """
        Get routing configuration.

        Args:
            tenant_id: Tenant identifier (required)
            service: Service name

        Returns:
            RoutingConfigUnified instance
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_routing_config"
        )
        value = self._cached_config_value(
            ConfigScope.ROUTING, tenant_id, service, "routing_config"
        )

        if value is None:
            logger.debug(
                f"No routing config found for {tenant_id}:{service}, using defaults"
            )
            return RoutingConfigUnified(tenant_id=tenant_id)

        return RoutingConfigUnified.from_dict(value)

    def set_routing_config(
        self,
        routing_config: RoutingConfigUnified,
        tenant_id: Optional[str] = None,
        service: str = "gateway_agent",
    ) -> RoutingConfigUnified:
        """
        Set routing configuration.

        Args:
            routing_config: RoutingConfigUnified instance
            tenant_id: Optional tenant override
            service: Service name

        Returns:
            Updated RoutingConfigUnified
        """
        if tenant_id:
            routing_config.tenant_id = tenant_id

        # Canonicalize so the storage key always matches what get_routing_config
        # looks up (which goes through require_tenant_id → canonical_tenant_id).
        routing_config.tenant_id = require_tenant_id(
            routing_config.tenant_id, source="ConfigManager.set_routing_config"
        )

        self.store.set_config(
            tenant_id=routing_config.tenant_id,
            scope=ConfigScope.ROUTING,
            service=service,
            config_key="routing_config",
            config_value=routing_config.to_dict(),
        )
        self._invalidate_scoped_config(ConfigScope.ROUTING, routing_config.tenant_id)

        logger.info(f"Set routing config for {routing_config.tenant_id}:{service}")
        return routing_config

    # ========== Durable Execution Configuration ==========

    def get_durable_execution_config(
        self, tenant_id: str = None, service: str = "optimization"
    ) -> DurableExecutionConfig:
        """Get per-tenant durable-execution config (defaults to disabled)."""
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_durable_execution_config"
        )
        value = self._cached_config_value(
            ConfigScope.DURABLE, tenant_id, service, "durable_execution_config"
        )
        if value is None:
            return DurableExecutionConfig(tenant_id=tenant_id)
        return DurableExecutionConfig.from_dict(value)

    def set_durable_execution_config(
        self,
        durable_config: DurableExecutionConfig,
        tenant_id: Optional[str] = None,
        service: str = "optimization",
    ) -> DurableExecutionConfig:
        """Set per-tenant durable-execution config."""
        if tenant_id:
            durable_config.tenant_id = tenant_id
        durable_config.tenant_id = require_tenant_id(
            durable_config.tenant_id,
            source="ConfigManager.set_durable_execution_config",
        )
        self.store.set_config(
            tenant_id=durable_config.tenant_id,
            scope=ConfigScope.DURABLE,
            service=service,
            config_key="durable_execution_config",
            config_value=durable_config.to_dict(),
        )
        self._invalidate_scoped_config(ConfigScope.DURABLE, durable_config.tenant_id)
        logger.info(
            f"Set durable execution config for {durable_config.tenant_id}:{service}"
        )
        return durable_config

    # ========== Telemetry Configuration ==========

    def get_telemetry_config(
        self, tenant_id: str = None, service: str = "telemetry"
    ) -> TelemetryConfig:
        """
        Get telemetry configuration.

        Args:
            tenant_id: Tenant identifier (required — pass
                ``SYSTEM_TENANT_ID`` for cluster-wide telemetry config).
            service: Service name

        Returns:
            TelemetryConfig instance
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_telemetry_config"
        )
        value = self._cached_config_value(
            ConfigScope.TELEMETRY, tenant_id, service, "telemetry_config"
        )

        if value is None:
            logger.debug(
                f"No telemetry config found for {tenant_id}:{service}, using defaults"
            )
            return TelemetryConfig()

        return TelemetryConfig.from_dict(value)

    def set_telemetry_config(
        self,
        telemetry_config: TelemetryConfig,
        tenant_id: str = None,
        service: str = "telemetry",
    ) -> TelemetryConfig:
        """
        Set telemetry configuration.

        Args:
            telemetry_config: TelemetryConfig instance
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            Updated TelemetryConfig
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.set_telemetry_config"
        )
        self.store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.TELEMETRY,
            service=service,
            config_key="telemetry_config",
            config_value=telemetry_config.to_dict(),
        )
        self._invalidate_scoped_config(ConfigScope.TELEMETRY, tenant_id)

        logger.info(f"Set telemetry config for {tenant_id}:{service}")
        return telemetry_config

    # ========== Backend Configuration ==========

    def get_backend_config(
        self, tenant_id: str = None, service: str = "backend"
    ) -> BackendConfig:
        """
        Get backend configuration for tenant.

        Args:
            tenant_id: Tenant identifier (required)
            service: Service name

        Returns:
            BackendConfig instance (may be empty if no tenant overrides)
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_backend_config"
        )
        value = self._cached_config_value(
            ConfigScope.BACKEND, tenant_id, service, "backend_config"
        )

        if value is None:
            # Return empty backend config - system config will be merged in ConfigUtils
            logger.debug(
                f"No backend config found for {tenant_id}:{service}, using empty config"
            )
            return BackendConfig(tenant_id=tenant_id)

        return BackendConfig.from_dict(value)

    def set_backend_config(
        self,
        backend_config: BackendConfig,
        tenant_id: Optional[str] = None,
        service: str = "backend",
    ) -> BackendConfig:
        """
        Set backend configuration.

        Args:
            backend_config: BackendConfig instance
            tenant_id: Optional tenant override
            service: Service name

        Returns:
            Updated BackendConfig
        """
        if tenant_id:
            backend_config.tenant_id = tenant_id

        # Canonicalize so the storage key always matches what get_backend_config
        # looks up (which goes through require_tenant_id → canonical_tenant_id).
        backend_config.tenant_id = require_tenant_id(
            backend_config.tenant_id, source="ConfigManager.set_backend_config"
        )

        self.store.set_config(
            tenant_id=backend_config.tenant_id,
            scope=ConfigScope.BACKEND,
            service=service,
            config_key="backend_config",
            config_value=backend_config.to_dict(),
        )
        self._invalidate_scoped_config(ConfigScope.BACKEND, backend_config.tenant_id)

        logger.info(f"Set backend config for {backend_config.tenant_id}:{service}")
        return backend_config

    def get_backend_profile(
        self, profile_name: str, tenant_id: str = None, service: str = "backend"
    ) -> Optional[BackendProfileConfig]:
        """
        Get a specific backend profile for tenant.

        Args:
            profile_name: Profile name
            tenant_id: Tenant identifier (required)
            service: Service name

        Returns:
            BackendProfileConfig if found, None otherwise
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_backend_profile"
        )
        backend_config = self.get_backend_config(tenant_id=tenant_id, service=service)
        return backend_config.get_profile(profile_name)

    def add_backend_profile(
        self,
        profile: BackendProfileConfig,
        tenant_id: str = None,
        service: str = "backend",
    ) -> BackendProfileConfig:
        """
        Add or update a backend profile for tenant.

        This adds a complete profile to the tenant's backend config.
        For partial updates, use update_backend_profile().

        Args:
            profile: BackendProfileConfig instance
            tenant_id: Tenant identifier (required)
            service: Service name

        Returns:
            Updated BackendProfileConfig
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.add_backend_profile"
        )
        with self._backend_lock:
            backend_config = self.get_backend_config(
                tenant_id=tenant_id, service=service
            )
            backend_config.add_profile(profile)
            self.set_backend_config(
                backend_config, tenant_id=tenant_id, service=service
            )

            logger.info(
                f"Added backend profile '{profile.profile_name}' for {tenant_id}:{service}"
            )

        # Notify outside the lock to avoid holding _backend_lock across
        # potentially slow listener work (e.g. backend dict updates).
        profile_dict = (
            profile.to_dict() if hasattr(profile, "to_dict") else dict(profile.__dict__)
        )
        self._notify_profile_change("added", profile.profile_name, profile_dict)
        return profile

    def update_backend_profile(
        self,
        profile_name: str,
        overrides: Dict[str, Any],
        base_tenant_id: str = SYSTEM_TENANT_ID,
        target_tenant_id: Optional[str] = None,
        service: str = "backend",
    ) -> BackendProfileConfig:
        """
        Update specific fields of a backend profile (tenant-specific tweak).

        This supports partial updates - only specified fields are overridden.
        Useful for tenant-specific customization of system profiles.

        Args:
            profile_name: Name of profile to update
            overrides: Dictionary of fields to override (supports deep merge)
            base_tenant_id: Tenant to inherit the base profile from. Defaults
                to ``SYSTEM_TENANT_ID`` (cluster-wide system profiles).
            target_tenant_id: Tenant to save updated profile to (defaults to
                ``base_tenant_id`` when omitted).
            service: Service name

        Returns:
            Updated BackendProfileConfig

        Raises:
            ValueError: If profile doesn't exist in base tenant

        Example:
            # Tenant "acme" wants to tweak the embedding model in a system profile
            manager.update_backend_profile(
                profile_name="video_colpali_smol500_mv_frame",
                overrides={"embedding_model": "custom/model"},
                base_tenant_id=SYSTEM_TENANT_ID,  # Inherit from cluster base
                target_tenant_id="acme",           # Save to acme tenant
            )
        """
        if target_tenant_id is None:
            target_tenant_id = base_tenant_id

        with self._backend_lock:
            # Get base profile (may be from default tenant or another tenant)
            base_config = self.get_backend_config(
                tenant_id=base_tenant_id, service=service
            )
            merged_profile = base_config.merge_profile(profile_name, overrides)

            # Save to target tenant
            target_config = self.get_backend_config(
                tenant_id=target_tenant_id, service=service
            )
            target_config.add_profile(merged_profile)
            self.set_backend_config(
                target_config, tenant_id=target_tenant_id, service=service
            )

            logger.info(
                f"Updated backend profile '{profile_name}' for {target_tenant_id}:{service} "
                f"(based on {base_tenant_id})"
            )
            return merged_profile

    def list_backend_profiles(
        self, tenant_id: str = None, service: str = "backend"
    ) -> Dict[str, BackendProfileConfig]:
        """
        List all backend profiles for a tenant.

        Args:
            tenant_id: Tenant identifier (required)
            service: Service name

        Returns:
            Dictionary mapping profile names to BackendProfileConfig instances
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.list_backend_profiles"
        )
        backend_config = self.get_backend_config(tenant_id=tenant_id, service=service)
        return backend_config.profiles

    def delete_backend_profile(
        self, profile_name: str, tenant_id: str = None, service: str = "backend"
    ) -> bool:
        """
        Delete a backend profile for a tenant.

        Args:
            profile_name: Name of profile to delete
            tenant_id: Tenant identifier (required)
            service: Service name

        Returns:
            True if profile was deleted, False if profile didn't exist

        Example:
            manager.delete_backend_profile("custom_profile", tenant_id="acme")
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.delete_backend_profile"
        )
        with self._backend_lock:
            backend_config = self.get_backend_config(
                tenant_id=tenant_id, service=service
            )

            # Check if profile exists
            if profile_name not in backend_config.profiles:
                logger.warning(
                    f"Profile '{profile_name}' not found for {tenant_id}:{service}"
                )
                return False

            # Remove profile
            del backend_config.profiles[profile_name]

            # Save updated config
            self.set_backend_config(
                backend_config, tenant_id=tenant_id, service=service
            )

            logger.info(
                f"Deleted backend profile '{profile_name}' from {tenant_id}:{service}"
            )
        # Notify outside the lock so listeners can't deadlock on _backend_lock.
        self._notify_profile_change("removed", profile_name, None)
        return True

    # ========== Generic Configuration Access ==========

    def get_config_value(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        default: Optional[Any] = None,
    ) -> Any:
        """
        Get arbitrary configuration value.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_config_value"
        )
        entry = self.store.get_config(
            tenant_id=tenant_id, scope=scope, service=service, config_key=config_key
        )

        if entry is None:
            return default

        return entry.config_value

    def set_config_value(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ):
        """
        Set arbitrary configuration value.

        Args:
            tenant_id: Tenant identifier
            scope: Configuration scope
            service: Service name
            config_key: Configuration key
            config_value: Configuration value
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.set_config_value"
        )
        self.store.set_config(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
            config_value=config_value,
        )
        # Same-manager setters invalidate immediately (the TTL only bounds
        # staleness for writes from other processes) — the typed setters all
        # do this, and reads routed through the scoped cache rely on it.
        self._invalidate_scoped_config(scope, tenant_id)

    # ========== Bulk Operations ==========

    def get_all_configs(
        self, tenant_id: str, scope: Optional[ConfigScope] = None
    ) -> Dict[str, Any]:
        """
        Get all configurations for a tenant.

        Args:
            tenant_id: Tenant identifier
            scope: Optional scope filter

        Returns:
            Dictionary of all configurations
        """
        tenant_id = require_tenant_id(tenant_id, source="ConfigManager.get_all_configs")
        entries = self.store.list_configs(tenant_id=tenant_id, scope=scope)

        configs = {}
        for entry in entries:
            key = f"{entry.scope.value}:{entry.service}:{entry.config_key}"
            configs[key] = {
                "value": entry.config_value,
                "version": entry.version,
                "updated_at": entry.updated_at.isoformat(),
            }

        return configs

    def export_configs(self, tenant_id: str, output_path: Path):
        """
        Export all configurations for a tenant to JSON file.

        Args:
            tenant_id: Tenant identifier
            output_path: Output file path
        """
        import json

        configs = self.get_all_configs(tenant_id=tenant_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "tenant_id": tenant_id,
                    "exported_at": __import__("datetime").datetime.now().isoformat(),
                    "configs": configs,
                },
                f,
                indent=2,
            )

        logger.info(f"Exported configs for {tenant_id} to {output_path}")

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """
        Get configuration statistics.

        Returns:
            Dictionary with statistics
        """
        return self.store.get_stats()
