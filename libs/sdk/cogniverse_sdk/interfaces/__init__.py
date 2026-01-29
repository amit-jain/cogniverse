"""
Cogniverse SDK Interfaces.

Defines the core interfaces that backend implementations must satisfy:
- Backend: Document/vector storage interface
- ConfigStore: Configuration storage interface
- SchemaLoader: Schema template loading interface
- WorkflowStore: Workflow intelligence storage interface
- AdapterStore: Adapter registry storage interface
"""

from cogniverse_sdk.interfaces.adapter_store import AdapterStore
from cogniverse_sdk.interfaces.backend import Backend, IngestionBackend, SearchBackend
from cogniverse_sdk.interfaces.config_store import ConfigStore
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader
from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformanceRecord,
    AgentStats,
    ExecutionRecord,
    WorkflowStore,
    WorkflowTemplate,
)

__all__ = [
    "AdapterStore",
    "Backend",
    "IngestionBackend",
    "SearchBackend",
    "ConfigStore",
    "SchemaLoader",
    "WorkflowStore",
    "ExecutionRecord",
    "AgentPerformanceRecord",
    "AgentStats",
    "WorkflowTemplate",
]
