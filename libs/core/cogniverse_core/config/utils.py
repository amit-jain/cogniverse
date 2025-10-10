"""
Configuration utilities for accessing ConfigManager with dict-like interface.
Provides get_config() helper that wraps ConfigManager for convenient access.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from cogniverse_core.config.manager import get_config_manager

logger = logging.getLogger(__name__)


class ConfigUtils:
    """
    Config utility wrapper that provides dict-like access to ConfigManager.

    Provides convenient .get() interface while using ConfigManager backend.
    """

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self._config_manager = get_config_manager()
        self._system_config = None
        self._routing_config = None
        self._telemetry_config = None
        self._json_config = None  # Cache for JSON config loaded from COGNIVERSE_CONFIG

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

    def _load_json_config(self):
        """Load config from JSON file if COGNIVERSE_CONFIG is set"""
        if self._json_config is not None:
            return  # Already loaded

        config_path = os.environ.get('COGNIVERSE_CONFIG')
        if not config_path:
            self._json_config = {}
            return

        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self._json_config = json.load(f)
                logger.debug(f"Loaded JSON config from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_path}")
                self._json_config = {}
        except Exception as e:
            logger.error(f"Error loading JSON config from {config_path}: {e}")
            self._json_config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key with dict-like interface.

        Maps config keys to ConfigManager structure.
        """
        self._ensure_system_config()
        self._ensure_routing_config()
        self._ensure_telemetry_config()

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
            "vespa_url": lambda: self._system_config.vespa_url,
            "vespa_port": lambda: self._system_config.vespa_port,
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

        # Check JSON config for keys not in ConfigManager (like video_processing_profiles)
        self._load_json_config()
        if key in self._json_config:
            return self._json_config[key]

        # Return default if not found
        logger.warning(
            f"Config key '{key}' not found in ConfigManager, returning default"
        )
        return default


def get_config(tenant_id: str = "default") -> ConfigUtils:
    """
    Get config utility wrapper for dict-like access.

    Args:
        tenant_id: Tenant identifier

    Returns:
        ConfigUtils instance that delegates to ConfigManager
    """
    return ConfigUtils(tenant_id)


def get_config_value(key: str, default: Any = None, tenant_id: str = "default") -> Any:
    """
    Get single config value by key.

    Args:
        key: Configuration key
        default: Default value if not found
        tenant_id: Tenant identifier

    Returns:
        Configuration value
    """
    config = ConfigUtils(tenant_id)
    return config.get(key, default)
