#!/usr/bin/env python3
"""
Factory for creating embedding generators based on backend type
"""

from typing import Dict, Any, Optional
import logging
from .embedding_generator_impl import EmbeddingGeneratorImpl
from .backend_factory import BackendFactory


class EmbeddingGeneratorFactory:
    """Factory for creating embedding generators"""
    
    @staticmethod
    def create(
        backend: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        profile_config: Dict[str, Any] = None
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
        
        # Backend needs both main config and profile config
        # Profile config contains schema_name and model-specific settings
        # Main config contains system-wide settings like vespa_url
        backend_config = {
            **config,  # System config (vespa_url, ports, etc.)
            **profile_config  # Profile config (schema_name, model, etc.) - overwrites any duplicates
        }
        
        # Delegate to backend factory
        backend_client = BackendFactory.create(backend, backend_config, logger)
        
        # Return embedding generator with backend client
        return EmbeddingGeneratorImpl(
            config=config,
            logger=logger,
            profile_config=profile_config,
            backend_client=backend_client
        )


def create_embedding_generator(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
):
    """
    Creates an embedding generator based on config settings.
    
    The active profile in config determines:
    - Backend type (from embedding_backend)
    - Processing type (from profile's process_type)
    - Model to use (from profile's embedding_model)
    """
    
    # Get backend from config
    backend = config.get("embedding_backend", "vespa")
    
    # Get active profile
    active_profile = config.get("active_profile")
    if not active_profile:
        raise ValueError("No active_profile specified in config")
    
    # Get profile configuration
    profiles = config.get("video_processing_profiles", {})
    if active_profile not in profiles:
        raise ValueError(f"Profile '{active_profile}' not found in video_processing_profiles")
    
    profile_config = profiles[active_profile]
    
    # Create and return generator
    return EmbeddingGeneratorFactory.create(
        backend=backend,
        config=config,
        logger=logger,
        profile_config=profile_config
    )