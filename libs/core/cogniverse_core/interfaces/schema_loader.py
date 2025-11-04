"""
Abstract interface for loading Vespa schema definitions.

This module defines the SchemaLoader interface and related exceptions
for loading schema templates and ranking strategies from various sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SchemaNotFoundException(Exception):
    """Raised when a requested schema template is not found."""

    pass


class SchemaLoadError(Exception):
    """Raised when a schema fails to load or parse."""

    pass


class SchemaLoader(ABC):
    """
    Abstract interface for loading Vespa schema definitions.

    Implementations can load schemas from various sources:
    - Filesystem directories (FilesystemSchemaLoader)
    - S3 buckets (S3SchemaLoader)
    - Databases (DatabaseSchemaLoader)
    - Remote APIs (RemoteSchemaLoader)

    All schema loaders must implement the four core methods for:
    - Loading individual schemas
    - Listing available schemas
    - Checking schema existence
    - Loading ranking strategies
    """

    @abstractmethod
    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Load a schema definition by name.

        Args:
            schema_name: Name of the schema to load (without _schema.json suffix)

        Returns:
            Dictionary containing the complete schema definition

        Raises:
            SchemaNotFoundException: If schema does not exist
            SchemaLoadError: If schema exists but fails to load/parse
        """
        pass

    @abstractmethod
    def list_available_schemas(self) -> List[str]:
        """
        List all available schema names.

        Returns:
            List of schema names (without _schema.json suffix)

        Raises:
            SchemaLoadError: If unable to list schemas
        """
        pass

    @abstractmethod
    def schema_exists(self, schema_name: str) -> bool:
        """
        Check if a schema exists.

        Args:
            schema_name: Name of the schema to check

        Returns:
            True if schema exists, False otherwise
        """
        pass

    @abstractmethod
    def load_ranking_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Load ranking strategies configuration.

        Returns:
            Dictionary mapping strategy names to their configurations.
            Expected structure:
            {
                "strategy_name": {
                    "ranking_profile": "profile_name",
                    "parameters": {...}
                },
                ...
            }

        Raises:
            SchemaLoadError: If ranking strategies fail to load/parse
        """
        pass
