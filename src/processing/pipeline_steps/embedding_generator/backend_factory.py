#!/usr/bin/env python3
"""
Backend Factory - Creates backend clients
"""

from typing import Dict, Any, Optional
import logging
from .backend_client import BackendClient
from .vespa_pyvespa_client import VespaPyClient


class BackendFactory:
    """Factory for creating backend clients"""
    
    _backends = {
        "vespa": VespaPyClient,
        # "elasticsearch": ElasticsearchClient,  # Future
        # "weaviate": WeaviateClient,           # Future
    }
    
    @classmethod
    def create(
        cls,
        backend_type: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> BackendClient:
        """Create a backend client of the specified type"""
        
        backend_type = backend_type.lower()
        
        if backend_type not in cls._backends:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available backends: {list(cls._backends.keys())}"
            )
        
        backend_class = cls._backends[backend_type]
        # Each backend extracts its own schema from config
        return backend_class(config, logger)
    
    @classmethod
    def register_backend(cls, name: str, backend_class: type):
        """Register a new backend type"""
        cls._backends[name.lower()] = backend_class