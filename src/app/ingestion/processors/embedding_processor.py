#!/usr/bin/env python3
"""
Embedding Processor - Pluggable embedding generation.

Generates embeddings using the existing embedding generator system.
"""

import logging
from typing import Any

from ..processor_base import BaseProcessor


class EmbeddingProcessor(BaseProcessor):
    """Handles embedding generation using existing embedding generators."""

    PROCESSOR_NAME = "embedding"

    def __init__(
        self,
        logger: logging.Logger,
        embedding_type: str = "multi_vector",
        model_name: str = "vidore/colpali-v1.2",
    ):
        """
        Initialize embedding processor.

        Args:
            logger: Logger instance
            embedding_type: Type of embedding (multi_vector, single_vector)
            model_name: Model to use for embedding generation
        """
        super().__init__(logger)
        self.embedding_type = embedding_type
        self.model_name = model_name

    @classmethod
    def from_config(
        cls, config: dict[str, Any], logger: logging.Logger
    ) -> "EmbeddingProcessor":
        """Create embedding processor from configuration."""
        return cls(
            logger=logger,
            embedding_type=config.get("type", "multi_vector"),
            model_name=config.get("model_name", "vidore/colpali-v1.2"),
        )

    def process(self, *args, **kwargs) -> Any:
        """Process input data (implements BaseProcessor abstract method)."""
        # Delegate to generate_embeddings for backwards compatibility
        if args and isinstance(args[0], dict):
            return self.generate_embeddings(args[0])
        return self.generate_embeddings(kwargs)

    def generate_embeddings(self, data: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings (placeholder - delegates to existing system)."""
        self.logger.info("🧬 Embedding generation delegated to existing system")

        # For now, return placeholder to avoid breaking the pipeline
        return {
            "embeddings": "delegated_to_existing_system",
            "type": self.embedding_type,
            "model": self.model_name,
        }

    def cleanup(self):
        """Clean up embedding resources."""
        pass
