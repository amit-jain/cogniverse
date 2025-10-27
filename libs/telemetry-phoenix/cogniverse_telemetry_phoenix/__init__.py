"""
Phoenix telemetry provider module.

Provides Phoenix-specific implementations of telemetry interfaces.
Auto-registers with telemetry registry via entry points.
"""

from .provider import PhoenixProvider

__all__ = [
    "PhoenixProvider",
]
