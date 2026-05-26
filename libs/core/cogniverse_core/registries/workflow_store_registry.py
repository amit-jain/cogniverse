"""WorkflowStore registry — entry-point auto-discovery for workflow stores.

A thin subclass of :class:`cogniverse_foundation.registry.EntryPointRegistry`.
Implementations register via the ``cogniverse.workflow.stores`` entry-point
group, e.g.::

    [project.entry-points."cogniverse.workflow.stores"]
    telemetry = "cogniverse_agents.workflow.telemetry_workflow_store:TelemetryWorkflowStore"

Callers fetch instances with ``WorkflowStoreRegistry.get(name="telemetry",
config={"telemetry_provider": provider})``.
"""

from __future__ import annotations

from cogniverse_foundation.registry import EntryPointRegistry
from cogniverse_sdk.interfaces.workflow_store import WorkflowStore


class WorkflowStoreRegistry(EntryPointRegistry[WorkflowStore]):
    """Plugin registry for ``WorkflowStore`` implementations."""

    _entry_point_group = "cogniverse.workflow.stores"
    _label = "workflow store"
