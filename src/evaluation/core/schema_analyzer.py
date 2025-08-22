"""
Schema-driven analyzer system with pluggable domain-specific extensions.

This module provides a generic, extensible system for analyzing queries
and extracting ground truth based on the actual schema being used,
without any hardcoded assumptions about the domain (video, documents, etc).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class SchemaAnalyzer(ABC):
    """Base class for schema analyzers."""

    @abstractmethod
    def can_handle(self, schema_name: str, schema_fields: Dict[str, Any]) -> bool:
        """
        Check if this analyzer can handle the given schema.

        Args:
            schema_name: Name of the schema
            schema_fields: Available fields in the schema

        Returns:
            True if this analyzer can handle the schema
        """
        pass

    @abstractmethod
    def analyze_query(
        self, query: str, schema_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze query based on schema fields.

        Args:
            query: The search query
            schema_fields: Available fields in the schema

        Returns:
            Dictionary of extracted constraints and metadata
        """
        pass

    @abstractmethod
    def extract_item_id(self, document: Any) -> Optional[str]:
        """
        Extract the item ID from a document.

        Args:
            document: The document/result object

        Returns:
            The extracted item ID or None
        """
        pass

    @abstractmethod
    def get_expected_field_name(self) -> str:
        """
        Get the field name for expected results.

        Returns:
            Field name like "expected_items", "expected_results", etc.
        """
        pass


class DefaultSchemaAnalyzer(SchemaAnalyzer):
    """Default analyzer that works with any schema."""

    def can_handle(self, schema_name: str, schema_fields: Dict[str, Any]) -> bool:
        """Generic analyzer can handle any schema as fallback."""
        return True

    def analyze_query(
        self, query: str, schema_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic query analysis based on available fields."""
        query_lower = query.lower()

        constraints = {
            "query_type": "generic",
            "field_constraints": {},
            "text_match": [],
            "available_fields": schema_fields,
        }

        # Check for field-specific queries (field:value pattern)
        field_pattern = r'(\w+):\s*"?([^"\s]+)"?'
        matches = re.findall(field_pattern, query)
        for field, value in matches:
            if self._field_exists(field, schema_fields):
                constraints["field_constraints"][field] = value

        # Extract general text tokens for matching
        tokens = [token for token in query_lower.split() if ":" not in token]
        constraints["text_match"] = tokens

        return constraints

    def extract_item_id(self, document: Any) -> Optional[str]:
        """Extract ID from document using common patterns."""
        # Try dict first (most common case)
        if isinstance(document, dict):
            # Try common ID field names
            id_fields = [
                "id",
                "_id",
                "item_id",
                "document_id",
                "doc_id",
                "video_id",
                "image_id",
                "audio_id",
                "file_id",
                "source_id",
                "uid",
                "key",
                "product_id",
                "sku",
            ]
            for id_field in id_fields:
                if id_field in document:
                    return str(document[id_field])

        # Try object attributes (new Document structure)
        if hasattr(document, "id"):
            return document.id

        if hasattr(document, "metadata"):
            # Try common ID field names
            for id_field in ["id", "item_id", "document_id", "source_id", "uid"]:
                if id_field in document.metadata:
                    return document.metadata[id_field]

        # Try to get any field ending with '_id'
        if hasattr(document, "__dict__"):
            for key, value in document.__dict__.items():
                if key.endswith("_id") and value:
                    return str(value)

        return None

    def get_expected_field_name(self) -> str:
        """Default expected field name."""
        return "expected_items"

    def _field_exists(self, field: str, schema_fields: Dict[str, Any]) -> bool:
        """Check if field exists in any field category."""
        for category in schema_fields.values():
            if isinstance(category, list) and field in category:
                return True
        return False


class TemporalSchemaAnalyzer(SchemaAnalyzer):
    """Analyzer for schemas with temporal data."""

    def can_handle(self, schema_name: str, schema_fields: Dict[str, Any]) -> bool:
        """Check if schema has temporal fields."""
        temporal_fields = schema_fields.get("temporal_fields", [])
        return bool(temporal_fields)

    def analyze_query(
        self, query: str, schema_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze temporal queries."""
        query_lower = query.lower()

        constraints = {
            "query_type": "temporal",
            "temporal_constraints": {},
            "available_fields": schema_fields,
        }

        temporal_fields = schema_fields.get("temporal_fields", [])

        # Only extract temporal patterns if we have the fields to support them
        if "start_time" in temporal_fields or "start" in temporal_fields:
            patterns = [
                (r"after (\d+)", "after_time"),
                (r"from (\d+)", "from_time"),
            ]
            for pattern, constraint_type in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    constraints["temporal_constraints"][
                        constraint_type
                    ] = match.groups()

        if "end_time" in temporal_fields or "end" in temporal_fields:
            patterns = [
                (r"before (\d+)", "before_time"),
                (r"until (\d+)", "until_time"),
                (r"first (\d+)", "first_n"),
            ]
            for pattern, constraint_type in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    constraints["temporal_constraints"][
                        constraint_type
                    ] = match.groups()

        if "duration" in temporal_fields:
            patterns = [
                (r"(\d+) seconds? long", "duration"),
                (r"duration (\d+)", "duration"),
            ]
            for pattern, constraint_type in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    constraints["temporal_constraints"][
                        constraint_type
                    ] = match.groups()

        return constraints

    def extract_item_id(self, document: Any) -> Optional[str]:
        """Extract ID from temporal document."""
        # Delegate to default analyzer
        return DefaultSchemaAnalyzer().extract_item_id(document)

    def get_expected_field_name(self) -> str:
        """Temporal expected field name."""
        return "expected_items"


class SchemaAnalyzerRegistry:
    """Registry for schema analyzers with plugin support."""

    def __init__(self):
        self._analyzers: List[SchemaAnalyzer] = []
        self._register_default_analyzers()

    def _register_default_analyzers(self):
        """Register default analyzers."""
        # Only register the default fallback
        # Domain-specific analyzers should be registered by plugins
        self._analyzers.append(DefaultSchemaAnalyzer())

    def register(self, analyzer: SchemaAnalyzer):
        """
        Register a new analyzer.

        Args:
            analyzer: The analyzer to register
        """
        # Insert before the default analyzer (last one)
        if self._analyzers and isinstance(self._analyzers[-1], DefaultSchemaAnalyzer):
            self._analyzers.insert(-1, analyzer)
        else:
            self._analyzers.append(analyzer)

        logger.info(f"Registered analyzer: {analyzer.__class__.__name__}")

    def get_analyzer(
        self, schema_name: str, schema_fields: Dict[str, Any]
    ) -> SchemaAnalyzer:
        """
        Get the appropriate analyzer for a schema.

        Args:
            schema_name: Name of the schema
            schema_fields: Available fields in the schema

        Returns:
            The most appropriate analyzer
        """
        for analyzer in self._analyzers:
            if analyzer.can_handle(schema_name, schema_fields):
                logger.info(
                    f"Using {analyzer.__class__.__name__} for schema {schema_name}"
                )
                return analyzer

        # Should never reach here due to DefaultSchemaAnalyzer
        return DefaultSchemaAnalyzer()

    def register_plugin(self, plugin_module: str):
        """
        Register analyzers from a plugin module.

        Args:
            plugin_module: Module path like 'src.evaluation.plugins.video'
        """
        try:
            import importlib

            module = importlib.import_module(plugin_module)

            # Look for classes that inherit from SchemaAnalyzer
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, SchemaAnalyzer)
                    and obj != SchemaAnalyzer
                ):
                    self.register(obj())

        except ImportError as e:
            logger.warning(f"Could not load plugin {plugin_module}: {e}")


# Global registry
_registry = SchemaAnalyzerRegistry()


def get_schema_analyzer(
    schema_name: str, schema_fields: Dict[str, Any]
) -> SchemaAnalyzer:
    """Get the appropriate analyzer for a schema."""
    return _registry.get_analyzer(schema_name, schema_fields)


def register_analyzer(analyzer: SchemaAnalyzer):
    """Register a custom analyzer."""
    _registry.register(analyzer)


def register_plugin(plugin_module: str):
    """Register a plugin module."""
    _registry.register_plugin(plugin_module)
