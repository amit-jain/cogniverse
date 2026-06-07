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

    @classmethod
    def _cache_key(cls, name, config, tenant_id):
        """Key telemetry providers per (tenant, project).

        Unlike the default tenant-only key, a tenant can register distinct
        endpoints per project (manager.register_project), so projects must not
        share one cached provider — else the second project silently reuses the
        first's endpoints. Falls back to the tenant-only key when no project.
        """
        base = super()._cache_key(name, config, tenant_id)
        project = (config or {}).get("project_name")
        return f"{base}_{project}" if project else base


_telemetry_registry = TelemetryRegistry()


def get_telemetry_registry() -> TelemetryRegistry:
    """Get the global ``TelemetryRegistry`` instance."""
    return _telemetry_registry
