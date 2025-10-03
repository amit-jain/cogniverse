"""
Shared configuration management for Cogniverse.

DEPRECATED: This module is deprecated in favor of ConfigManager.
Use src.common.config_manager.get_config_manager() instead.

Kept for backward compatibility during migration.
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Migration warning
warnings.warn(
    "src.common.config is deprecated. Use ConfigManager instead:\n"
    "  from src.common.config_manager import get_config_manager\n"
    "  config_manager = get_config_manager()\n"
    "  system_config = config_manager.get_system_config('default')",
    DeprecationWarning,
    stacklevel=2,
)


class Config:
    """
    Singleton configuration manager.

    DEPRECATED: Use ConfigManager instead.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config_data = {}
        self.config_path = None
        self.load_config()
        self._initialized = True
    
    def load_config(self):
        """Load configuration from file or environment."""
        # Start with environment defaults
        self.config_data = {
            "vespa_url": os.getenv("VESPA_URL", None),
            "vespa_port": int(os.getenv("VESPA_PORT", "8080")) if os.getenv("VESPA_PORT") else None,
            "phoenix_url": os.getenv("PHOENIX_URL", None),
            "search_backend": os.getenv("SEARCH_BACKEND", None),
        }
        
        # Find and load config file
        config_file = self._find_config_file()
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    # Environment variables take precedence over file config
                    for key, value in file_config.items():
                        if self.config_data.get(key) is None:  # Only use file value if env var not set
                            self.config_data[key] = value
                    self.config_path = config_file
                    logger.info(f"Configuration loaded from: {config_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to load config from {config_file}: {e}")
        else:
            # Check if we have minimum required config from environment
            if not self.config_data.get("vespa_url"):
                raise RuntimeError(
                    "No configuration found! "
                    "Please provide config using one of:\n"
                    "  1. Config file: Place config.json in standard location\n"
                    "  2. Environment variable: export COGNIVERSE_CONFIG=/path/to/config.json\n"
                    "  3. Or set individual env vars: VESPA_URL, VESPA_PORT, etc."
                )
    
    def _find_config_file(self) -> Optional[Path]:
        """Find config file in standard locations."""
        # Check explicit path first
        if os.environ.get('COGNIVERSE_CONFIG'):
            path = Path(os.environ['COGNIVERSE_CONFIG'])
            if path.exists():
                return path
            else:
                raise RuntimeError(f"Config file specified but not found: {path}")
        
        # Search standard locations
        search_paths = [
            Path.cwd() / "config.json",
            Path.cwd() / "configs" / "config.json",
            Path.home() / ".cogniverse" / "config.json",
            Path("/etc/cogniverse/config.json"),
            # Relative to package when installed
            Path(__file__).parent.parent.parent / "configs" / "config.json",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config_data.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration."""
        self.config_data.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self.config_data.copy()


# Global instance
_config = Config()


def get_config() -> Dict[str, Any]:
    """Get configuration dictionary."""
    return _config.to_dict()


def get_config_value(key: str, default: Any = None) -> Any:
    """Get specific configuration value."""
    return _config.get(key, default)


def update_config(updates: Dict[str, Any]):
    """Update configuration."""
    _config.update(updates)
