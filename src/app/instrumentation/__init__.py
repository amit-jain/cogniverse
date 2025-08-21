"""
OpenTelemetry instrumentation for Cogniverse application
"""

from .phoenix import CogniverseInstrumentor, instrument_cogniverse

__all__ = ['CogniverseInstrumentor', 'instrument_cogniverse']