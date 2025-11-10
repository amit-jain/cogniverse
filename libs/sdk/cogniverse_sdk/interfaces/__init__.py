"""
Cogniverse SDK Interfaces.

Defines the core interfaces that backend implementations must satisfy:
- Backend: Document/vector storage interface
- ConfigStore: Configuration storage interface
- SchemaLoader: Schema template loading interface
"""

from cogniverse_sdk.interfaces.backend import Backend, IngestionBackend, SearchBackend
from cogniverse_sdk.interfaces.config_store import ConfigStore
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

__all__ = [
    "Backend",
    "IngestionBackend",
    "SearchBackend",
    "ConfigStore",
    "SchemaLoader",
]
