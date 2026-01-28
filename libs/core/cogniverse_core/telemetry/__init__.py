"""
Backward compatibility shim for moved telemetry module.

Telemetry module has moved to cogniverse_foundation.telemetry.
This module provides backward compatibility by re-exporting from foundation.
"""

from cogniverse_foundation.telemetry import *  # noqa: F401, F403
from cogniverse_foundation.telemetry import (  # noqa: F401
    config,
    context,
    exporter,
    manager,
    modality_metrics,
    providers,
    registry,
)
