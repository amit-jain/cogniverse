"""Bootstrap configuration from environment variables and config file.

Breaks circular dependency: Backend connection info comes from env vars + config.json,
not from ConfigStore (which needs backend to connect).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BootstrapConfig:
    """Minimal config loaded from environment and config file."""

    backend_type: str  # "vespa", "elasticsearch", etc.
    backend_url: str
    backend_port: int = 8080

    @classmethod
    def from_environment(cls, config_path: Optional[Path] = None) -> "BootstrapConfig":
        """Load bootstrap config from environment and config file.

        Environment variables:
            BACKEND_URL: Required. Backend server URL (e.g., http://localhost)
            BACKEND_PORT: Optional. Backend HTTP port (default: 8080)

        Config file (configs/config.json):
            backend.type: Backend type (vespa, elasticsearch, etc.)

        Args:
            config_path: Path to config.json. Defaults to configs/config.json

        Returns:
            BootstrapConfig instance

        Raises:
            ValueError: If BACKEND_URL is not set or config file missing
        """
        backend_url = os.getenv("BACKEND_URL")
        if not backend_url:
            raise ValueError(
                "BACKEND_URL environment variable is required. "
                "Set it to your backend server URL, e.g., BACKEND_URL=http://localhost"
            )

        # Load backend type from config.json
        if config_path is None:
            config_path = Path("configs/config.json")

        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        backend_type = config.get("backend", {}).get("type")
        if not backend_type:
            raise ValueError("backend.type not found in config.json")

        return cls(
            backend_type=backend_type,
            backend_url=backend_url,
            backend_port=int(os.getenv("BACKEND_PORT", "8080")),
        )
