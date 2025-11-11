"""
Multi-tenant telemetry system for Cogniverse.

This module provides:
- Multi-tenant tracer provider management
- Lazy initialization with caching
- Batch export with queue management
- Environment-based configuration
- Graceful degradation when telemetry unavailable
"""

from .config import TelemetryConfig
from .manager import TelemetryManager

__all__ = ["TelemetryManager", "TelemetryConfig"]
