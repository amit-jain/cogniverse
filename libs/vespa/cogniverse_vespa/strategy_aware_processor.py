"""
Strategy-aware document processor for Vespa ingestion.
Formats documents and selects embedding fields based on ranking strategy requirements
extracted from schema JSON files.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StrategyAwareProcessor:
    """Process documents based on ranking strategy requirements"""

    def __init__(self, schema_loader):
        """
        Initialize strategy-aware processor.

        Args:
            schema_loader: SchemaLoader instance for loading schemas (REQUIRED)
        """
        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for StrategyAwareProcessor. "
                "Dependency injection is mandatory - pass SchemaLoader instance explicitly."
            )
        self.schema_loader = schema_loader
        self.ranking_strategies = self._load_ranking_strategies()

    def _load_ranking_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load ranking strategies from JSON file"""
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        if not isinstance(self.schema_loader, FilesystemSchemaLoader):
            raise ValueError(
                f"Unsupported schema_loader type: {type(self.schema_loader)}. "
                f"StrategyAwareProcessor requires FilesystemSchemaLoader."
            )

        schemas_dir = self.schema_loader.base_path
        strategies_file = schemas_dir / "ranking_strategies.json"

        if not strategies_file.exists():
            # Auto-generate if missing
            from .ranking_strategy_extractor import (
                extract_all_ranking_strategies,
                save_ranking_strategies,
            )

            strategies = extract_all_ranking_strategies(schemas_dir)
            save_ranking_strategies(strategies, strategies_file)
            logger.info("Generated ranking strategies file")

        with open(strategies_file, "r") as f:
            return json.load(f)

    def get_required_embeddings(self, schema_name: str) -> Dict[str, bool]:
        """Determine what embeddings are needed based on ranking strategies"""

        if schema_name not in self.ranking_strategies:
            raise ValueError(f"No ranking strategies found for schema '{schema_name}'")

        schema_strategies = self.ranking_strategies[schema_name]

        needs_float = any(
            s.get("needs_float_embeddings", False) for s in schema_strategies.values()
        )
        needs_binary = any(
            s.get("needs_binary_embeddings", False) for s in schema_strategies.values()
        )

        return {"needs_float": needs_float, "needs_binary": needs_binary}

    def get_embedding_field_names(self, schema_name: str) -> Dict[str, str]:
        """Get the actual embedding field names used by the schema"""

        if schema_name not in self.ranking_strategies:
            raise ValueError(f"No ranking strategies found for schema '{schema_name}'")

        schema_strategies = self.ranking_strategies[schema_name]

        # Collect unique field names from all strategies
        float_fields = set()
        binary_fields = set()

        for strategy in schema_strategies.values():
            if strategy.get("needs_float_embeddings", False):
                field = strategy.get("embedding_field")
                if field and "binary" not in field:
                    float_fields.add(field)

            if strategy.get("needs_binary_embeddings", False):
                field = strategy.get("embedding_field")
                if field and "binary" in field:
                    binary_fields.add(field)

        # Select the float field that pairs with the binary field. ColBERT/patch
        # embeddings always come in float+binary pairs (e.g., semantic_embedding and
        # semantic_embedding_binary). HNSW-only fields like acoustic_embedding have
        # no binary counterpart and must not be selected as the primary float field.
        result = {}
        if binary_fields:
            result["binary_field"] = next(iter(binary_fields))
            # Derive the float field from the binary field base name
            binary_base = result["binary_field"].replace("_binary", "")
            if binary_base in float_fields:
                result["float_field"] = binary_base
            elif float_fields:
                result["float_field"] = next(iter(float_fields))
        elif float_fields:
            result["float_field"] = next(iter(float_fields))

        return result
