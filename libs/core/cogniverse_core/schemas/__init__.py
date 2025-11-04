"""
Cogniverse Schema Loaders.

This package contains concrete implementations of the SchemaLoader interface
for loading Vespa schema definitions from various sources.
"""

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

__all__ = [
    "FilesystemSchemaLoader",
]
