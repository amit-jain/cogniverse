"""
Centralized configuration manager with multi-tenant support.
Provides unified interface for all configuration operations with caching.
"""

import copy
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cogniverse_foundation.caching.refreshing_cache import RefreshingCache
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
from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope, ConfigStore

logger = logging.getLogger(__name__)


# Most (scope, tenant, service, key) scoped configs one manager holds in memory.
SCOPED_CONFIG_MAX_ENTRIES = 512

# The one key the system config is held under.
_SYSTEM_CONFIG_KEY = "system_config"


class BackendProfileExistsError(ValueError):
    """A create-only profile add found the profile already stored."""


class BackendProfileNotFoundError(ValueError):
    """A profile update found no profile of that name stored."""


@dataclass(frozen=True)
class BackendProfileWrite:
    """A stored profile add or update: the profile as written, and the
    version of the tenant's backend config this write produced (or, for a
    write that changed nothing, the version already holding it)."""

    profile: BackendProfileConfig
    version: int


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
    - Held in memory and refreshed off the reading thread
    """

    def __init__(
        self,
        store: ConfigStore,
        scoped_config_refresh_s: float = 5.0,
        scoped_config_max_staleness_s: float = 60.0,
        system_config_refresh_s: float = 5.0,
        system_config_max_staleness_s: float = 60.0,
    ):
        """
        Initialize configuration manager with required ConfigStore.

        Args:
            store: ConfigStore implementation (REQUIRED, no fallback)
            scoped_config_refresh_s: Age at which a per-tenant scoped
                config (routing/telemetry/backend/agent/durable/tenant
                instructions) is re-read from the store. The read runs on a
                background thread while callers keep getting the held value.
            scoped_config_max_staleness_s: Age at which a held scoped config
                is no longer served: the caller reads the store itself and a
                failed read raises. Same-manager setters invalidate at once;
                a write made by another process (another worker or pod) is
                served here within this bound, and within about
                ``scoped_config_refresh_s`` for a key read continuously.
                Both 0 reads the store on every call.
            system_config_refresh_s: Age at which the held system config is
                re-read from the store on a background thread while callers
                keep getting the held value.
            system_config_max_staleness_s: Age at which the held system
                config is no longer served: the caller reads the store itself
                and a failed read raises. ``set_system_config`` on this
                manager replaces it at once; another process's write is served
                here within this bound.

        Raises:
            ValueError: If store is None, or either pair of bounds is negative
                or has a refresh age above its staleness bound
        """
        if store is None:
            raise ValueError("store is required")

        self.store = store
        # `get_system_config` is hot, so the system config is held in memory
        # and re-read off the caller's thread; another process's write (the
        # runtime storing its deployment overrides, a dashboard edit) is
        # served within the staleness bound.
        self._system_config: RefreshingCache[str, SystemConfig] = RefreshingCache(
            name="system-config",
            refresh_after_s=system_config_refresh_s,
            max_staleness_s=system_config_max_staleness_s,
            max_entries=1,
        )
        # A process's explicitly deployed inference endpoints, served in
        # place of the persisted ones and never written to the store.
        self._pinned_inference_service_urls: Optional[Dict[str, str]] = None
        # Per-tenant scoped configs are read on every request through
        # ConfigUtils' ensure cascade, and each store read is a document visit
        # costing a few hundred milliseconds. The raw config_value per
        # (scope, tenant, service, key) is held in memory and refreshed off the
        # request thread; values are deep-copied on the way out so callers
        # can't mutate shared state.
        self._scoped_configs: RefreshingCache[tuple, Optional[dict]] = RefreshingCache(
            name="scoped-config",
            refresh_after_s=scoped_config_refresh_s,
            max_staleness_s=scoped_config_max_staleness_s,
            max_entries=SCOPED_CONFIG_MAX_ENTRIES,
        )

        logger.info("ConfigManager initialized with %s", type(self.store).__name__)

    # ========== System Configuration ==========

    # Fixed sentinel tenant_id for system-wide config storage.
    # SystemConfig is global — not per-tenant.
    _SYSTEM_TENANT_ID = "_system"

    def get_system_config(self) -> SystemConfig:
        """Get system-wide infrastructure configuration.

        Served from memory within ``system_config_refresh_s`` and
        ``system_config_max_staleness_s`` (see ``__init__``); a store failure
        past the staleness bound raises.

        Returns:
            SystemConfig instance
        """
        # A copy, so a caller mutating a field (the get-modify-set path, or a
        # nested dict) cannot change the value other callers are served.
        return self._served_system_config(
            self._system_config.get(_SYSTEM_CONFIG_KEY, self._stored_system_config)
        )

    def _stored_system_config(self) -> SystemConfig:
        """The system config as the store holds it now."""
        entry = self.store.get_config(
            tenant_id=self._SYSTEM_TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
        )
        if entry is None:
            logger.warning("No system config found, using defaults")
            return SystemConfig()
        return SystemConfig.from_dict(entry.config_value)

    def _served_system_config(self, cached: SystemConfig) -> SystemConfig:
        served = copy.deepcopy(cached)
        pinned = self._pinned_inference_service_urls
        if pinned is not None:
            served.inference_service_urls = dict(pinned)
        return served

    def pin_inference_service_urls(self, service_urls: Dict[str, str]) -> None:
        """Serve ``service_urls`` as the system config's inference endpoints.

        Every later ``get_system_config`` from this manager carries exactly
        these endpoints in place of the persisted discovery. Nothing is
        written to the store, so other processes keep reading the persisted
        endpoints.
        """
        self._pinned_inference_service_urls = dict(service_urls)

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
            # Persist the real key, not the display-redacted "***".
            config_value=system_config.to_dict(redact=False),
        )
        # Hold what was written, detaching any read in flight, so no read
        # that began before the write is served after it.
        self._system_config.put(_SYSTEM_CONFIG_KEY, copy.deepcopy(system_config))

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
        # Served from the scoped-config cache — this read sits on the
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
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_agent_config_history"
        )
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
        """Return the raw ``config_value`` for a scoped config from memory,
        within the refresh and staleness bounds. ``None`` (config absent) is
        held too, so tenants without overrides don't re-query the store per
        request."""

        def read() -> Optional[dict]:
            return self._stored_config_value(scope, tenant_id, service, config_key)

        return copy.deepcopy(
            self._scoped_configs.get((scope, tenant_id, service, config_key), read)
        )

    def _stored_config_value(
        self, scope: ConfigScope, tenant_id: str, service: str, config_key: str
    ) -> Optional[dict]:
        """The scoped config's ``config_value`` as the store holds it now."""
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
        )
        return entry.config_value if entry is not None else None

    def _invalidate_scoped_config(self, scope: ConfigScope, tenant_id: str) -> None:
        """Drop held entries for a (scope, tenant) after a write."""
        self._scoped_configs.invalidate(
            lambda key: key[0] == scope and key[1] == tenant_id
        )

    def forget_held_configs(self, tenant_id: str) -> None:
        """Drop everything held for ``tenant_id`` (the system config for the
        system tenant), so the next read of each comes from the store."""
        if tenant_id == self._SYSTEM_TENANT_ID:
            self._system_config.invalidate(lambda key: True)
        self._scoped_configs.invalidate(lambda key: key[1] == tenant_id)

    def compare_and_set_entry(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
        *,
        expected_version: int,
    ) -> Optional[ConfigEntry]:
        """Store ``config_value`` as the next version when the stored one is
        ``expected_version`` (0: none stored); None when another write landed
        first. Either way this manager next reads the entry from the store.
        """
        try:
            return self.store.compare_and_set_config(
                tenant_id,
                scope,
                service,
                config_key,
                config_value,
                expected_version=expected_version,
            )
        finally:
            self.forget_held_configs(tenant_id)

    # ========== Tenant Instructions ==========

    def get_tenant_instructions_config(self, tenant_id: str) -> Optional[Any]:
        """Get the raw tenant-instructions value (the SOUL.md equivalent).

        Served from the scoped-config cache — every memory-aware agent reads
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
        return self._backend_config_from(value, tenant_id, service)

    def get_stored_backend_config(
        self, tenant_id: str, service: str = "backend"
    ) -> BackendConfig:
        """The tenant's backend config as the store holds it now.

        ``get_backend_config`` serves the held copy, which may predate another
        process's write by up to the staleness bound; a write that decides on
        what the tenant has stored reads this instead.
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.get_stored_backend_config"
        )
        value = self._stored_config_value(
            ConfigScope.BACKEND, tenant_id, service, "backend_config"
        )
        return self._backend_config_from(value, tenant_id, service)

    def _update_backend_config(
        self,
        tenant_id: str,
        service: str,
        change: Callable[[BackendConfig], None],
    ) -> tuple[bool, int]:
        """Apply ``change`` to the stored backend config with compare-and-set.

        Every profile change is a read-modify-write of the tenant's whole
        backend config, and other processes write it too: ``change`` runs on
        the config as the store holds it, and again on the newer one each
        time a concurrent write lands first, so no process's change is
        overwritten. Returns whether ``change`` altered the stored config and
        the version holding the result: the version this call wrote, or,
        when ``change`` left the config as stored and nothing was written,
        the stored version (0 when the tenant has none). Raises
        ``ConfigWriteConflictError`` when every attempt lost, and storage
        failures, with nothing written.
        """

        changed = False

        def update(entry):
            nonlocal changed
            config = self._backend_config_from(
                entry.config_value if entry is not None else None, tenant_id, service
            )
            config.tenant_id = tenant_id
            # Deep: a profile merge rewrites nested dicts in place.
            before = copy.deepcopy(config.to_dict())
            change(config)
            after = config.to_dict()
            changed = after != before
            return after if changed else None

        entry = self.store.update_config(
            tenant_id, ConfigScope.BACKEND, service, "backend_config", update
        )
        # Even a change that wrote nothing read the config as stored now, and
        # what this manager holds may predate another process's write.
        self._invalidate_scoped_config(ConfigScope.BACKEND, tenant_id)
        return changed, 0 if entry is None else entry.version

    @staticmethod
    def _backend_config_from(
        value: Optional[dict], tenant_id: str, service: str
    ) -> BackendConfig:
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
        *,
        replace: bool = True,
    ) -> BackendProfileWrite:
        """
        Add or update a backend profile for tenant.

        This adds a complete profile to the tenant's backend config.
        For partial updates, use update_backend_profile().

        Args:
            profile: BackendProfileConfig instance
            tenant_id: Tenant identifier (required)
            service: Service name
            replace: When False, raise ``BackendProfileExistsError`` if the
                stored config already holds a profile of this name; checked
                in the same compare-and-set as the write, so of concurrent
                creators on any process exactly one succeeds.

        Returns:
            The profile and the backend config version holding it
        """
        tenant_id = require_tenant_id(
            tenant_id, source="ConfigManager.add_backend_profile"
        )

        def add(backend_config: BackendConfig) -> None:
            if not replace and profile.profile_name in backend_config.profiles:
                raise BackendProfileExistsError(
                    f"Profile '{profile.profile_name}' already exists for "
                    f"tenant '{tenant_id}'"
                )
            backend_config.add_profile(profile)

        written, version = self._update_backend_config(tenant_id, service, add)
        logger.info(
            f"{'Added' if written else 'Already held'} backend profile "
            f"'{profile.profile_name}' for {tenant_id}:{service} (version {version})"
        )
        return BackendProfileWrite(profile=profile, version=version)

    def update_backend_profile(
        self,
        profile_name: str,
        overrides: Dict[str, Any],
        base_tenant_id: str = SYSTEM_TENANT_ID,
        target_tenant_id: Optional[str] = None,
        service: str = "backend",
    ) -> BackendProfileWrite:
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
            The merged profile and the target tenant's backend config version
            holding it

        Raises:
            BackendProfileNotFoundError: If the profile is not stored for the
                base tenant, checked in the same compare-and-set as the write

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
        base_tenant_id = require_tenant_id(
            base_tenant_id, source="ConfigManager.update_backend_profile"
        )
        target_tenant_id = require_tenant_id(
            target_tenant_id, source="ConfigManager.update_backend_profile"
        )

        # Another tenant's base is read once; the target's own profile is
        # merged on every compare-and-set attempt, so a concurrent change to
        # it is the base this update overrides.
        base_config = (
            None
            if base_tenant_id == target_tenant_id
            else self.get_stored_backend_config(base_tenant_id, service)
        )
        merged: Dict[str, BackendProfileConfig] = {}

        def merge(target_config: BackendConfig) -> None:
            source = target_config if base_config is None else base_config
            if profile_name not in source.profiles:
                raise BackendProfileNotFoundError(
                    f"Profile '{profile_name}' not found for tenant '{base_tenant_id}'"
                )
            merged["profile"] = source.merge_profile(profile_name, overrides)
            target_config.add_profile(merged["profile"])

        _, version = self._update_backend_config(target_tenant_id, service, merge)

        logger.info(
            f"Updated backend profile '{profile_name}' for "
            f"{target_tenant_id}:{service} (based on {base_tenant_id}, "
            f"version {version})"
        )
        return BackendProfileWrite(profile=merged["profile"], version=version)

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
        deleted, _ = self._update_backend_config(
            tenant_id,
            service,
            lambda backend_config: backend_config.profiles.pop(profile_name, None),
        )
        if not deleted:
            logger.warning(
                f"Profile '{profile_name}' not found for {tenant_id}:{service}"
            )
            return False

        logger.info(
            f"Deleted backend profile '{profile_name}' from {tenant_id}:{service}"
        )
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
        # Same-manager setters invalidate immediately (the staleness bound
        # only covers writes from other processes) — the typed setters all do
        # this, and reads routed through the scoped cache rely on it.
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
                    "exported_at": datetime.now(timezone.utc).isoformat(),
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
