"""
Configuration utilities for accessing ConfigManager with dict-like interface.
Provides get_config() helper that wraps ConfigManager for convenient access.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendConfig

logger = logging.getLogger(__name__)


class ConfigUtils:
    """
    Config utility wrapper that provides dict-like access to ConfigManager.

    Provides convenient .get() interface while using ConfigManager backend.
    """

    def __init__(self, tenant_id: str, config_manager: ConfigManager):
        """
        Initialize ConfigUtils.

        Args:
            tenant_id: Tenant identifier
            config_manager: ConfigManager instance (REQUIRED)
        """
        if config_manager is None:
            raise ValueError(
                "config_manager is required for ConfigUtils initialization"
            )
        self.tenant_id = tenant_id
        self._config_manager = config_manager
        self._system_config = None
        self._routing_config = None
        self._telemetry_config = None
        self._backend_config = None  # Merged backend config (system + tenant)
        self._json_config = None  # Cache for JSON config (auto-discovered)

    def _ensure_system_config(self):
        """Lazy load system config"""
        if self._system_config is None:
            self._system_config = self._config_manager.get_system_config(self.tenant_id)

    def _ensure_routing_config(self):
        """Lazy load routing config"""
        if self._routing_config is None:
            self._routing_config = self._config_manager.get_routing_config(
                self.tenant_id
            )

    def _ensure_telemetry_config(self):
        """Lazy load telemetry config"""
        if self._telemetry_config is None:
            self._telemetry_config = self._config_manager.get_telemetry_config(
                self.tenant_id
            )

    @staticmethod
    def _discover_config_file() -> Optional[Path]:
        """
        Auto-discover config.json from standard locations.

        Search order:
        1. COGNIVERSE_CONFIG env var (if set)
        2. configs/config.json (from current directory)
        3. ../configs/config.json (one level up)
        4. ../../configs/config.json (two levels up)

        Returns:
            Path to config.json if found, None otherwise
        """
        # Check COGNIVERSE_CONFIG env var first (backward compatibility)
        env_path = os.environ.get("COGNIVERSE_CONFIG")
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.debug(f"Found config via COGNIVERSE_CONFIG: {path}")
                return path

        # Check standard locations
        search_paths = [
            Path("configs/config.json"),
            Path("../configs/config.json"),
            Path("../../configs/config.json"),
        ]

        for path in search_paths:
            if path.exists():
                logger.debug(f"Found config at: {path.resolve()}")
                return path.resolve()

        return None

    def _load_json_config(self):
        """Load config from auto-discovered JSON file"""
        if self._json_config is not None:
            return  # Already loaded

        config_path = self._discover_config_file()
        if not config_path:
            logger.warning("No config.json found in standard locations")
            self._json_config = {}
            return

        try:
            with open(config_path, "r") as f:
                self._json_config = json.load(f)
            logger.debug(f"Loaded system config from {config_path}")
        except Exception as e:
            logger.error(f"Error loading JSON config from {config_path}: {e}")
            self._json_config = {}

    def _ensure_backend_config(self):
        """
        Lazy load and merge backend config.

        Merges:
        1. System base config from config.json (backend section)
        2. Tenant-specific overrides from ConfigManager

        Result is a merged BackendConfig with system profiles + tenant overrides.
        """
        if self._backend_config is not None:
            return  # Already loaded

        # Load system config from JSON
        self._load_json_config()
        system_backend_data = self._json_config.get("backend", {})

        # Get tenant-specific overrides from ConfigManager
        tenant_backend_config = self._config_manager.get_backend_config(self.tenant_id)

        # Merge system base with tenant overrides
        if system_backend_data:
            # Create BackendConfig from system JSON
            system_backend_data["tenant_id"] = "default"  # Mark as system config
            system_backend_config = BackendConfig.from_dict(system_backend_data)

            # Deep merge tenant overrides into system base
            merged_profiles = dict(
                system_backend_config.profiles
            )  # Start with system profiles

            # Add/override with tenant-specific profiles
            for profile_name, tenant_profile in tenant_backend_config.profiles.items():
                merged_profiles[profile_name] = tenant_profile

            # Create merged config
            self._backend_config = BackendConfig(
                tenant_id=self.tenant_id,
                backend_type=(
                    tenant_backend_config.backend_type
                    or system_backend_config.backend_type
                ),
                url=(
                    tenant_backend_config.url
                    if tenant_backend_config.url != "http://localhost"
                    else system_backend_config.url
                ),
                port=(
                    tenant_backend_config.port
                    if tenant_backend_config.port != 8080
                    else system_backend_config.port
                ),
                profiles=merged_profiles,
                metadata={
                    **system_backend_config.metadata,
                    **tenant_backend_config.metadata,
                },
            )
        else:
            # No system config, just use tenant config
            self._backend_config = tenant_backend_config

        logger.debug(
            f"Merged backend config for tenant '{self.tenant_id}': "
            f"{len(self._backend_config.profiles)} profiles available"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key with dict-like interface.

        Maps config keys to ConfigManager structure.
        """
        self._ensure_system_config()
        self._ensure_routing_config()
        self._ensure_telemetry_config()
        self._ensure_backend_config()

        # Backend config - return merged system + tenant config
        if key == "backend":
            return self._backend_config.to_dict()

        # System config mappings
        system_keys = {
            "routing_agent_url": lambda: self._system_config.routing_agent_url,
            "video_agent_url": lambda: self._system_config.video_agent_url,
            "text_agent_url": lambda: self._system_config.text_agent_url,
            "summarizer_agent_url": lambda: self._system_config.summarizer_agent_url,
            "text_analysis_agent_url": lambda: self._system_config.text_analysis_agent_url,
            "detailed_report_agent_url": lambda: self._system_config.summarizer_agent_url,  # Alias
            "composing_agent_port": lambda: 8000,  # Hardcoded default
            "routing_agent_port": lambda: 8001,  # Hardcoded default
            "text_analysis_port": lambda: 8005,  # Hardcoded default
            "search_backend": lambda: self._system_config.search_backend,
            "backend_url": lambda: self._system_config.backend_url,
            "backend_port": lambda: self._system_config.backend_port,
            "url": lambda: self._system_config.backend_url,  # Alias for backend dict access
            "port": lambda: self._system_config.backend_port,  # Alias for backend dict access
            "elasticsearch_url": lambda: self._system_config.elasticsearch_url,
            "llm_model": lambda: self._system_config.llm_model,
            "local_llm_model": lambda: self._system_config.llm_model,  # Alias
            "base_url": lambda: self._system_config.base_url,
            "llm_api_key": lambda: self._system_config.llm_api_key,
            "phoenix_url": lambda: self._system_config.phoenix_url,
            "phoenix_collector_endpoint": lambda: self._system_config.phoenix_collector_endpoint,
            "environment": lambda: self._system_config.environment,
            # Nested llm dict for backward compatibility with agents expecting config.get("llm")
            "llm": lambda: {
                "model_name": self._system_config.llm_model,
                "base_url": self._system_config.base_url,
                "api_key": self._system_config.llm_api_key,
            },
        }

        # Routing config mappings
        routing_keys = {
            "routing_mode": lambda: self._routing_config.routing_mode,
            "enable_caching": lambda: self._routing_config.enable_caching,
            "cache_ttl_seconds": lambda: self._routing_config.cache_ttl_seconds,
        }

        # Telemetry config mappings
        telemetry_keys = {
            "telemetry_enabled": lambda: self._telemetry_config.enabled,
            "telemetry_level": lambda: self._telemetry_config.level,
        }

        # Try system keys first
        if key in system_keys:
            return system_keys[key]()

        # Try routing keys
        if key in routing_keys:
            return routing_keys[key]()

        # Try telemetry keys
        if key in telemetry_keys:
            return telemetry_keys[key]()

        # Check JSON config for other keys (fallback for legacy keys)
        self._load_json_config()
        if key in self._json_config:
            return self._json_config[key]

        # Return default if not found
        logger.warning(
            f"Config key '{key}' not found in ConfigManager, returning default"
        )
        return default

    def keys(self):
        """Return all available config keys (dict-like interface)"""
        self._ensure_system_config()
        self._ensure_routing_config()
        self._ensure_telemetry_config()
        self._ensure_backend_config()
        self._load_json_config()

        # Collect all available keys
        all_keys = set()

        # Add system keys
        all_keys.update(
            [
                "backend",
                "routing_agent_url",
                "video_agent_url",
                "text_agent_url",
                "summarizer_agent_url",
                "text_analysis_agent_url",
                "detailed_report_agent_url",
                "composing_agent_port",
                "routing_agent_port",
                "text_analysis_port",
                "search_backend",
                "backend_url",
                "backend_port",
                "url",
                "port",
                "elasticsearch_url",
                "llm_model",
                "local_llm_model",
                "base_url",
                "llm_api_key",
                "phoenix_url",
                "phoenix_collector_endpoint",
                "environment",
                "llm",
            ]
        )

        # Add routing keys
        all_keys.update(["routing_mode", "enable_caching", "cache_ttl_seconds"])

        # Add telemetry keys
        all_keys.update(["telemetry_enabled", "telemetry_level"])

        # Add JSON config keys
        all_keys.update(self._json_config.keys())

        return list(all_keys)

    def __contains__(self, key):
        """Support 'in' operator (dict-like interface)"""
        return key in self.keys()


def get_config(tenant_id: str, config_manager: ConfigManager) -> ConfigUtils:
    """
    Get config utility wrapper for dict-like access.

    Args:
        tenant_id: Tenant identifier
        config_manager: Optional ConfigManager instance

    Returns:
        ConfigUtils instance that delegates to ConfigManager
    """
    return ConfigUtils(tenant_id, config_manager=config_manager)


def get_config_value(
    key: str,
    default: Any = None,
    tenant_id: str = "default",
    config_manager: "ConfigManager" = None,
) -> Any:
    """
    Get single config value by key.

    Args:
        key: Configuration key
        default: Default value if not found
        tenant_id: Tenant identifier
        config_manager: ConfigManager instance (REQUIRED - no fallback)

    Returns:
        Configuration value

    Raises:
        ValueError: If config_manager is not provided
    """
    if config_manager is None:
        raise ValueError(
            "config_manager is required for get_config_value(). "
            "Dependency injection is mandatory - pass ConfigManager() explicitly."
        )
    config = ConfigUtils(tenant_id, config_manager)
    return config.get(key, default)


def create_default_config_manager(cache_size: int = 100) -> ConfigManager:
    """
    Factory function to create ConfigManager with backend store.

    Requires:
        Environment variables: BACKEND_URL, BACKEND_PORT (optional)
        Config file: configs/config.json with backend.type

    Args:
        cache_size: LRU cache size (number of configs per tenant)

    Returns:
        ConfigManager instance with appropriate backend store

    Raises:
        ValueError: If BACKEND_URL not set or unsupported backend type

    Example:
        # Set environment variables first
        # export BACKEND_URL=http://localhost
        # export BACKEND_PORT=8080

        config_manager = create_default_config_manager()

        # Custom store: Create ConfigManager directly
        mock_store = MockConfigStore()
        config_manager = ConfigManager(store=mock_store)
    """
    from cogniverse_foundation.config.bootstrap import BootstrapConfig

    bootstrap = BootstrapConfig.from_environment()

    if bootstrap.backend_type == "vespa":
        from cogniverse_vespa.config.config_store import VespaConfigStore

        store = VespaConfigStore(
            vespa_url=bootstrap.backend_url,
            vespa_port=bootstrap.backend_port,
        )
    else:
        raise ValueError(f"Unsupported backend type: {bootstrap.backend_type}")

    return ConfigManager(store=store, cache_size=cache_size)


# Singleton instance for convenience (optional, can use factory directly)
_config_manager_singleton: Optional[ConfigManager] = None


def get_config_manager_singleton() -> ConfigManager:
    """
    Get or create singleton ConfigManager instance.

    This is a convenience function for applications that want a global
    ConfigManager instance. For better testability, use create_default_config_manager()
    or explicit dependency injection instead.

    Requires BACKEND_URL environment variable to be set.

    Returns:
        Singleton ConfigManager instance

    Warning:
        Using a singleton can make testing harder. Prefer explicit dependency
        injection where possible.
    """
    global _config_manager_singleton
    if _config_manager_singleton is None:
        _config_manager_singleton = create_default_config_manager()
    return _config_manager_singleton
