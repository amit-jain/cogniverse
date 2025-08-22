#!/usr/bin/env python3
"""
Factory for creating embedding generators based on backend type
"""

import logging
from typing import Any, Dict, Optional

from .backend_factory import BackendFactory
from .embedding_generator_impl import EmbeddingGeneratorImpl


class EmbeddingGeneratorFactory:
    """Factory for creating embedding generators"""

    @staticmethod
    def create(
        backend: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        profile_config: Dict[str, Any] = None,
    ):
        """
        Create an embedding generator for the specified backend

        Args:
            backend: Backend type ("vespa", "elasticsearch" in future, etc)
            config: Configuration dictionary
            logger: Logger instance
            profile_config: Profile configuration containing process_type and model info

        Returns:
            Embedding generator instance
        """

        # Get backend client (singleton)
        backend_client = BackendFactory.create(
            backend_type=backend, config=config, logger=logger
        )

        return EmbeddingGeneratorImpl(
            config=profile_config,  # This becomes self.profile_config
            logger=logger,
            backend_client=backend_client,
        )


def create_embedding_generator(
    config: Dict[str, Any], schema_name: str, logger: Optional[logging.Logger] = None
):
    """
    Creates an embedding generator for a specific schema.

    Args:
        config: Main configuration dictionary
        schema_name: Schema/profile name to use
        logger: Optional logger
    """

    # Get backend from config
    backend = config.get("embedding_backend", config.get("search_backend", "vespa"))

    # Get profile configuration using schema_name as key
    profiles = config.get("video_processing_profiles", {})
    if schema_name not in profiles:
        raise ValueError(
            f"Profile '{schema_name}' not found in video_processing_profiles"
        )

    profile_config = profiles[schema_name]
    # Add schema_name to profile_config so it's available
    profile_config["schema_name"] = schema_name

    # Create and return generator
    return EmbeddingGeneratorFactory.create(
        backend=backend, config=config, logger=logger, profile_config=profile_config
    )
