"""
Registries for auto-discovery of implementations.

Provides pluggable architecture where implementations self-register
via entry points, allowing core modules to use them without direct imports.
"""

from cogniverse_core.registries.adapter_store_registry import AdapterStoreRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.registries.workflow_store_registry import WorkflowStoreRegistry

__all__ = [
    "AdapterStoreRegistry",
    "BackendRegistry",
    "SchemaRegistry",
    "WorkflowStoreRegistry",
]
