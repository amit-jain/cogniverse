"""WorkflowStore registry — entry-point auto-discovery for workflow stores.

A thin subclass of :class:`cogniverse_foundation.registry.EntryPointRegistry`.
Implementations register via the ``cogniverse.workflow.stores`` entry-point
group, e.g.::

    [project.entry-points."cogniverse.workflow.stores"]
    vespa = "cogniverse_vespa.workflow.workflow_store:VespaWorkflowStore"

Callers fetch instances with ``WorkflowStoreRegistry.get(name="vespa",
config={"backend_url": ..., "backend_port": ...})``.
"""

from __future__ import annotations

from cogniverse_foundation.registry import EntryPointRegistry
from cogniverse_sdk.interfaces.workflow_store import WorkflowStore


class WorkflowStoreRegistry(EntryPointRegistry[WorkflowStore]):
    """Plugin registry for ``WorkflowStore`` implementations."""

    _entry_point_group = "cogniverse.workflow.stores"
    _label = "workflow store"
