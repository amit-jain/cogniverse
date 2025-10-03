"""
Centralized configuration manager with multi-tenant support.
Provides unified interface for all configuration operations with caching.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.agent_config import AgentConfig
from src.common.config_store import SQLiteConfigStore
from src.common.config_store_interface import ConfigScope, ConfigStore
from src.common.unified_config import (
    AgentConfigUnified,
    RoutingConfigUnified,
    SystemConfig,
    TelemetryConfigUnified,
)

logger = logging.getLogger(__name__)


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
    - Cached (LRU cache with configurable size)
    - Persistent (SQLite storage)
    """

    _instance = None
    _db_path = None

    def __new__(cls, db_path: Optional[Path] = None, *args, **kwargs):
        """Singleton pattern with db_path tracking"""
        # If db_path changed, reset singleton
        if db_path is not None and db_path != cls._db_path:
            cls._instance = None
            cls._db_path = db_path

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self, db_path: Optional[Path] = None, cache_size: int = 100, store: Optional[ConfigStore] = None):
        """
        Initialize configuration manager.

        Args:
            db_path: Path to SQLite database (ignored if store provided)
            cache_size: LRU cache size (number of configs per tenant)
            store: Optional ConfigStore implementation (defaults to SQLiteConfigStore)
        """
        if self._initialized:
            return

        # Use provided store or create SQLiteConfigStore
        if store is not None:
            self.store = store
        else:
            self.store = SQLiteConfigStore(db_path)

        self.cache_size = cache_size
        self._initialized = True

        logger.info(f"ConfigManager initialized with {type(self.store).__name__}, cache size: {cache_size}")

    # ========== System Configuration ==========

    def get_system_config(self, tenant_id: str = "default") -> SystemConfig:
        """
        Get system configuration for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            SystemConfig instance
        """
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
        )

        if entry is None:
            # Return default system config
            logger.warning(
                f"No system config found for tenant {tenant_id}, using defaults"
            )
            return SystemConfig(tenant_id=tenant_id)

        return SystemConfig.from_dict(entry.config_value)

    def set_system_config(
        self, system_config: SystemConfig, tenant_id: Optional[str] = None
    ) -> SystemConfig:
        """
        Set system configuration.

        Args:
            system_config: SystemConfig instance
            tenant_id: Optional tenant override

        Returns:
            Updated SystemConfig
        """
        if tenant_id:
            system_config.tenant_id = tenant_id

        self.store.set_config(
            tenant_id=system_config.tenant_id,
            scope=ConfigScope.SYSTEM,
            service="system",
            config_key="system_config",
            config_value=system_config.to_dict(),
        )

        logger.info(f"Set system config for tenant {system_config.tenant_id}")
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
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.AGENT,
            service=agent_name,
            config_key="agent_config",
        )

        if entry is None:
            return None

        unified = AgentConfigUnified.from_dict(entry.config_value)
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
        unified = AgentConfigUnified(tenant_id=tenant_id, agent_config=agent_config)

        self.store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.AGENT,
            service=agent_name,
            config_key="agent_config",
            config_value=unified.to_dict(),
        )

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

    # ========== Routing Configuration ==========

    def get_routing_config(
        self, tenant_id: str = "default", service: str = "routing_agent"
    ) -> RoutingConfigUnified:
        """
        Get routing configuration.

        Args:
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            RoutingConfigUnified instance
        """
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.ROUTING,
            service=service,
            config_key="routing_config",
        )

        if entry is None:
            logger.warning(
                f"No routing config found for {tenant_id}:{service}, using defaults"
            )
            return RoutingConfigUnified(tenant_id=tenant_id)

        return RoutingConfigUnified.from_dict(entry.config_value)

    def set_routing_config(
        self,
        routing_config: RoutingConfigUnified,
        tenant_id: Optional[str] = None,
        service: str = "routing_agent",
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

        self.store.set_config(
            tenant_id=routing_config.tenant_id,
            scope=ConfigScope.ROUTING,
            service=service,
            config_key="routing_config",
            config_value=routing_config.to_dict(),
        )

        logger.info(f"Set routing config for {routing_config.tenant_id}:{service}")
        return routing_config

    # ========== Telemetry Configuration ==========

    def get_telemetry_config(
        self, tenant_id: str = "default", service: str = "telemetry"
    ) -> TelemetryConfigUnified:
        """
        Get telemetry configuration.

        Args:
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            TelemetryConfigUnified instance
        """
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.TELEMETRY,
            service=service,
            config_key="telemetry_config",
        )

        if entry is None:
            logger.warning(
                f"No telemetry config found for {tenant_id}:{service}, using defaults"
            )
            return TelemetryConfigUnified(tenant_id=tenant_id)

        return TelemetryConfigUnified.from_dict(entry.config_value)

    def set_telemetry_config(
        self,
        telemetry_config: TelemetryConfigUnified,
        tenant_id: Optional[str] = None,
        service: str = "telemetry",
    ) -> TelemetryConfigUnified:
        """
        Set telemetry configuration.

        Args:
            telemetry_config: TelemetryConfigUnified instance
            tenant_id: Optional tenant override
            service: Service name

        Returns:
            Updated TelemetryConfigUnified
        """
        if tenant_id:
            telemetry_config.tenant_id = tenant_id

        self.store.set_config(
            tenant_id=telemetry_config.tenant_id,
            scope=ConfigScope.TELEMETRY,
            service=service,
            config_key="telemetry_config",
            config_value=telemetry_config.to_dict(),
        )

        logger.info(
            f"Set telemetry config for {telemetry_config.tenant_id}:{service}"
        )
        return telemetry_config

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
        self.store.set_config(
            tenant_id=tenant_id,
            scope=scope,
            service=service,
            config_key=config_key,
            config_value=config_value,
        )

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


# Singleton accessor
def get_config_manager() -> ConfigManager:
    """Get singleton ConfigManager instance"""
    return ConfigManager()
