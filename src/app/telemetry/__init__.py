"""
Multi-tenant telemetry system for Cogniverse.

This module provides:
- Multi-tenant tracer provider management
- Lazy initialization with caching
- Batch export with queue management
- Environment-based configuration
- Graceful degradation when telemetry unavailable
"""

from .manager import TelemetryManager
from .config import TelemetryConfig

__all__ = [
    "TelemetryManager", 
    "TelemetryConfig"
]