"""Telemetry provider registry — entry-point auto-discovery.

A thin subclass of :class:`cogniverse_foundation.registry.EntryPointRegistry`.
The base handles discovery, manual registration, conflict detection,
tenant-scoped caching, and lifecycle-style initialization
(``klass()`` + ``.initialize(config)``). This module is the
foundation-package entry point that the rest of the codebase imports.

Implementations register via the ``cogniverse.telemetry.providers``
entry-point group::

    [project.entry-points."cogniverse.telemetry.providers"]
    phoenix = "cogniverse_telemetry_phoenix:PhoenixProvider"
"""

from __future__ import annotations

from cogniverse_foundation.registry import EntryPointRegistry
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider


class TelemetryRegistry(EntryPointRegistry[TelemetryProvider]):
    """Plugin registry for ``TelemetryProvider`` implementations.

    Providers are tenant-scoped: each ``get()`` call returns a per-tenant
    cached instance, constructed via ``klass()`` then handed the merged
    ``{tenant_id, **config}`` dict via ``.initialize(...)``.
    """

    _entry_point_group = "cogniverse.telemetry.providers"
    _label = "telemetry provider"
    _tenant_scoped = True


_telemetry_registry = TelemetryRegistry()


def get_telemetry_registry() -> TelemetryRegistry:
    """Get the global ``TelemetryRegistry`` instance."""
    return _telemetry_registry
