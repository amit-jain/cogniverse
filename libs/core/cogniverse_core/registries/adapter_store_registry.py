"""AdapterStore registry — entry-point auto-discovery for adapter stores.

A thin subclass of :class:`cogniverse_foundation.registry.EntryPointRegistry`.
Implementations register via the ``cogniverse.adapter.stores`` entry-point
group, e.g.::

    [project.entry-points."cogniverse.adapter.stores"]
    vespa = "cogniverse_vespa.registry.adapter_store:VespaAdapterStore"

Callers fetch instances with ``AdapterStoreRegistry.get(name="vespa",
config={"backend_url": ..., "backend_port": ...})``.
"""

from __future__ import annotations

from cogniverse_foundation.registry import EntryPointRegistry
from cogniverse_sdk.interfaces.adapter_store import AdapterStore


class AdapterStoreRegistry(EntryPointRegistry[AdapterStore]):
    """Plugin registry for ``AdapterStore`` implementations."""

    _entry_point_group = "cogniverse.adapter.stores"
    _label = "adapter store"
