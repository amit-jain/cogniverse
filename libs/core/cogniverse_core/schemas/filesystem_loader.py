"""
Filesystem-based schema loader implementation.

This module provides a concrete implementation of SchemaLoader that loads
schema definitions from a local filesystem directory.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from cogniverse_sdk.interfaces.schema_loader import (
    SchemaLoader,
    SchemaLoadError,
    SchemaNotFoundException,
)


class FilesystemSchemaLoader(SchemaLoader):
    """
    Load Vespa schemas from filesystem directory.

    This loader expects:
    - Schema files named: {schema_name}_schema.json
    - Ranking strategies file: ranking_strategies.json
    - All files in the same base directory

    Example directory structure:
        configs/schemas/
            video_colpali_schema.json
            video_videoprism_schema.json
            ranking_strategies.json
    """

    def __init__(self, base_path: Path):
        """
        Initialize filesystem schema loader.

        Args:
            base_path: Directory containing schema JSON files

        Raises:
            ValueError: If base_path is None or directory does not exist
        """
        if base_path is None:
            raise ValueError("base_path is required")

        self.base_path = Path(base_path)

        if not self.base_path.exists():
            raise ValueError(f"Schema directory does not exist: {base_path}")

        if not self.base_path.is_dir():
            raise ValueError(f"Schema path is not a directory: {base_path}")

    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Load a schema definition by name.

        Args:
            schema_name: Name of the schema to load (without _schema.json suffix)

        Returns:
            Dictionary containing the complete schema definition

        Raises:
            SchemaNotFoundException: If schema file does not exist
            SchemaLoadError: If schema exists but fails to load/parse
        """
        if not schema_name:
            raise ValueError("schema_name cannot be empty")

        schema_file = self.base_path / f"{schema_name}_schema.json"

        if not schema_file.exists():
            raise SchemaNotFoundException(
                f"Schema '{schema_name}' not found at {schema_file}"
            )

        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaLoadError(f"Failed to parse schema '{schema_name}': {e}") from e
        except Exception as e:
            raise SchemaLoadError(f"Failed to load schema '{schema_name}': {e}") from e

    def list_available_schemas(self) -> List[str]:
        """
        List all available schema names.

        Returns:
            List of schema names (without _schema.json suffix)

        Raises:
            SchemaLoadError: If unable to list schemas
        """
        try:
            schema_files = self.base_path.glob("*_schema.json")
            return [f.stem.replace("_schema", "") for f in schema_files if f.is_file()]
        except Exception as e:
            raise SchemaLoadError(f"Failed to list schemas: {e}") from e

    def schema_exists(self, schema_name: str) -> bool:
        """
        Check if a schema exists.

        Args:
            schema_name: Name of the schema to check

        Returns:
            True if schema exists, False otherwise
        """
        if not schema_name:
            return False

        schema_file = self.base_path / f"{schema_name}_schema.json"
        return schema_file.exists() and schema_file.is_file()

    def load_ranking_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Load ranking strategies configuration.

        Returns:
            Dictionary mapping strategy names to their configurations

        Raises:
            SchemaLoadError: If ranking strategies fail to load/parse
        """
        strategies_file = self.base_path / "ranking_strategies.json"

        if not strategies_file.exists():
            raise SchemaLoadError(
                f"Ranking strategies file not found at {strategies_file}"
            )

        try:
            with open(strategies_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaLoadError(f"Failed to parse ranking strategies: {e}") from e
        except Exception as e:
            raise SchemaLoadError(f"Failed to load ranking strategies: {e}") from e

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"FilesystemSchemaLoader(base_path={self.base_path})"
