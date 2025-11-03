"""
Centralized configuration manager with multi-tenant support.
Provides unified interface for all configuration operations with caching.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cogniverse_core.common.config_store_interface import ConfigScope, ConfigStore
from cogniverse_core.config.agent_config import AgentConfig
from cogniverse_core.config.store import SQLiteConfigStore
from cogniverse_core.config.unified_config import (
    AgentConfigUnified,
    BackendConfig,
    BackendProfileConfig,
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

    def __init__(
        self,
        db_path: Optional[Path] = None,
        cache_size: int = 100,
        store: Optional[ConfigStore] = None,
    ):
        """
        Initialize configuration manager.

        Args:
            db_path: Path to SQLite database (ignored if store provided)
            cache_size: LRU cache size (number of configs per tenant)
            store: Optional ConfigStore implementation (defaults to SQLiteConfigStore)
        """
        # Use provided store or create SQLiteConfigStore
        if store is not None:
            self.store = store
        else:
            self.store = SQLiteConfigStore(db_path)

        self.cache_size = cache_size

        logger.info(
            f"ConfigManager initialized with {type(self.store).__name__}, cache size: {cache_size}"
        )

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

        logger.info(f"Set telemetry config for {telemetry_config.tenant_id}:{service}")
        return telemetry_config

    # ========== Backend Configuration ==========

    def get_backend_config(
        self, tenant_id: str = "default", service: str = "backend"
    ) -> BackendConfig:
        """
        Get backend configuration for tenant.

        Args:
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            BackendConfig instance (may be empty if no tenant overrides)
        """
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.BACKEND,
            service=service,
            config_key="backend_config",
        )

        if entry is None:
            # Return empty backend config - system config will be merged in ConfigUtils
            logger.debug(
                f"No backend config found for {tenant_id}:{service}, using empty config"
            )
            return BackendConfig(tenant_id=tenant_id)

        return BackendConfig.from_dict(entry.config_value)

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

        self.store.set_config(
            tenant_id=backend_config.tenant_id,
            scope=ConfigScope.BACKEND,
            service=service,
            config_key="backend_config",
            config_value=backend_config.to_dict(),
        )

        logger.info(f"Set backend config for {backend_config.tenant_id}:{service}")
        return backend_config

    def get_backend_profile(
        self, profile_name: str, tenant_id: str = "default", service: str = "backend"
    ) -> Optional[BackendProfileConfig]:
        """
        Get a specific backend profile for tenant.

        Args:
            profile_name: Profile name
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            BackendProfileConfig if found, None otherwise
        """
        backend_config = self.get_backend_config(tenant_id=tenant_id, service=service)
        return backend_config.get_profile(profile_name)

    def add_backend_profile(
        self,
        profile: BackendProfileConfig,
        tenant_id: str = "default",
        service: str = "backend",
    ) -> BackendProfileConfig:
        """
        Add or update a backend profile for tenant.

        This adds a complete profile to the tenant's backend config.
        For partial updates, use update_backend_profile().

        Args:
            profile: BackendProfileConfig instance
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            Updated BackendProfileConfig
        """
        backend_config = self.get_backend_config(tenant_id=tenant_id, service=service)
        backend_config.add_profile(profile)
        self.set_backend_config(backend_config, tenant_id=tenant_id, service=service)

        logger.info(
            f"Added backend profile '{profile.profile_name}' for {tenant_id}:{service}"
        )
        return profile

    def update_backend_profile(
        self,
        profile_name: str,
        overrides: Dict[str, Any],
        base_tenant_id: str = "default",
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
            base_tenant_id: Tenant to get base profile from (default = "default")
            target_tenant_id: Tenant to save updated profile to (default = base_tenant_id)
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
                base_tenant_id="default",  # Get system profile
                target_tenant_id="acme"     # Save to acme tenant
            )
        """
        if target_tenant_id is None:
            target_tenant_id = base_tenant_id

        # Get base profile (may be from default tenant or another tenant)
        base_config = self.get_backend_config(tenant_id=base_tenant_id, service=service)
        merged_profile = base_config.merge_profile(profile_name, overrides)

        # Save to target tenant
        target_config = self.get_backend_config(tenant_id=target_tenant_id, service=service)
        target_config.add_profile(merged_profile)
        self.set_backend_config(target_config, tenant_id=target_tenant_id, service=service)

        logger.info(
            f"Updated backend profile '{profile_name}' for {target_tenant_id}:{service} "
            f"(based on {base_tenant_id})"
        )
        return merged_profile

    def list_backend_profiles(
        self, tenant_id: str = "default", service: str = "backend"
    ) -> Dict[str, BackendProfileConfig]:
        """
        List all backend profiles for a tenant.

        Args:
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            Dictionary mapping profile names to BackendProfileConfig instances
        """
        backend_config = self.get_backend_config(tenant_id=tenant_id, service=service)
        return backend_config.profiles

    def delete_backend_profile(
        self, profile_name: str, tenant_id: str = "default", service: str = "backend"
    ) -> bool:
        """
        Delete a backend profile for a tenant.

        Args:
            profile_name: Name of profile to delete
            tenant_id: Tenant identifier
            service: Service name

        Returns:
            True if profile was deleted, False if profile didn't exist

        Example:
            manager.delete_backend_profile("custom_profile", tenant_id="acme")
        """
        backend_config = self.get_backend_config(tenant_id=tenant_id, service=service)

        # Check if profile exists
        if profile_name not in backend_config.profiles:
            logger.warning(
                f"Profile '{profile_name}' not found for {tenant_id}:{service}"
            )
            return False

        # Remove profile
        del backend_config.profiles[profile_name]

        # Save updated config
        self.set_backend_config(backend_config, tenant_id=tenant_id, service=service)

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

    # ========== Schema Registry Methods ==========

    def register_deployed_schema(
        self,
        tenant_id: str,
        base_schema_name: str,
        full_schema_name: str,
        schema_definition: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a deployed schema in the config store.

        Stores:
        - tenant_id: Tenant identifier
        - base_schema_name: Original schema name (e.g., 'video_colpali_smol500_mv_frame')
        - full_schema_name: Tenant-scoped name (e.g., 'video_colpali_smol500_mv_frame_test_tenant')
        - schema_definition: Full .sd file content as string
        - config: Schema configuration (profile, model, etc.)
        - deployment_time: Timestamp of deployment

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name
            full_schema_name: Tenant-scoped schema name
            schema_definition: Full schema definition as string
            config: Optional schema configuration
        """
        from datetime import datetime

        config_key = f"schema_{base_schema_name}"
        value = {
            "tenant_id": tenant_id,
            "base_schema_name": base_schema_name,
            "full_schema_name": full_schema_name,
            "schema_definition": schema_definition,
            "config": config or {},
            "deployment_time": datetime.now().isoformat(),
        }

        self.store.set_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=config_key,
            config_value=value,
        )

        logger.info(
            f"Registered schema '{base_schema_name}' for tenant '{tenant_id}' "
            f"(full name: '{full_schema_name}')"
        )

    def get_deployed_schemas(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all deployed schemas, optionally filtered by tenant.

        Args:
            tenant_id: Optional tenant filter (None = all tenants)

        Returns:
            List of schema info dicts with:
            - tenant_id
            - base_schema_name
            - full_schema_name
            - schema_definition
            - config
            - deployment_time
        """
        if tenant_id:
            # Get schemas for specific tenant
            entries = self.store.list_configs(
                tenant_id=tenant_id,
                scope=ConfigScope.SCHEMA,
                service="schema_registry",
            )
        else:
            # This is trickier - need to get across all tenants
            # For now, we'll require tenant_id to be specified
            # In future, could extend store interface to support cross-tenant queries
            raise ValueError(
                "tenant_id must be specified. Cross-tenant queries not yet supported."
            )

        schemas = []
        for entry in entries:
            schema_info = entry.config_value
            # Filter out deleted schemas
            if not schema_info.get("deleted", False):
                schemas.append(schema_info)

        return schemas

    def schema_deployed(self, tenant_id: str, base_schema_name: str) -> bool:
        """
        Check if schema already deployed for tenant.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name

        Returns:
            True if schema is deployed, False otherwise
        """
        config_key = f"schema_{base_schema_name}"

        try:
            entry = self.store.get_config(
                tenant_id=tenant_id,
                scope=ConfigScope.SCHEMA,
                service="schema_registry",
                config_key=config_key,
            )
            logger.info(f"[SCHEMA_DEPLOYED] Query: {tenant_id}:{base_schema_name} -> Found: {entry is not None}")

            if entry is None:
                return False

            # Check if schema is marked as deleted
            schema_info = entry.config_value
            return not schema_info.get("deleted", False)

        except Exception as e:
            logger.debug(
                f"Error checking schema deployment for {tenant_id}:{base_schema_name}: {e}"
            )
            return False

    def unregister_schema(self, tenant_id: str, base_schema_name: str) -> None:
        """
        Remove schema from registry (mark as deleted).

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Original schema name
        """
        from datetime import datetime

        config_key = f"schema_{base_schema_name}"

        # Mark schema as deleted rather than actually deleting
        # This preserves history and allows for audit trail
        entry = self.store.get_config(
            tenant_id=tenant_id,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=config_key,
        )

        if entry:
            schema_info = entry.config_value
            schema_info["deleted"] = True
            schema_info["deleted_at"] = datetime.now().isoformat()

            self.store.set_config(
                tenant_id=tenant_id,
                scope=ConfigScope.SCHEMA,
                service="schema_registry",
                config_key=config_key,
                config_value=schema_info,
            )

            logger.info(
                f"Unregistered schema '{base_schema_name}' for tenant '{tenant_id}'"
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
