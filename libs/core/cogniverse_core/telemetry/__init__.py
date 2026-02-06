"""
Backward compatibility shim for moved telemetry module.

Telemetry module has moved to cogniverse_foundation.telemetry.
This module provides backward compatibility by re-exporting from foundation.
"""

# Submodule re-exports (enables `from cogniverse_core.telemetry.context import ...`)
from cogniverse_foundation.telemetry import (  # noqa: F401
    TelemetryConfig,
    TelemetryManager,
    config,
    context,
    exporter,
    get_telemetry_manager,
    manager,
    providers,
    registry,
)

__all__ = [
    "TelemetryConfig",
    "TelemetryManager",
    "get_telemetry_manager",
    "config",
    "context",
    "exporter",
    "manager",
    "providers",
    "registry",
]
