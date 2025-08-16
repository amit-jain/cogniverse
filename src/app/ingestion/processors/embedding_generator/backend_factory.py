#!/usr/bin/env python3
"""
Backend Factory - Creates backend clients
"""

from typing import Dict, Any, Optional
import logging
from src.common.core.backend_registry import get_backend_registry
from src.common.core.interfaces import IngestionBackend


class BackendFactory:
    """Factory for creating backends using the backend registry"""
    
    @classmethod
    def create(
        cls,
        backend_type: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> IngestionBackend:
        """Create a backend of the specified type using the backend registry
        
        Returns:
            IngestionBackend: The backend that implements the ingestion interface
        """
        
        backend_type = backend_type.lower()
        
        # Get backend from registry
        registry = get_backend_registry()
        
        backend = registry.get_ingestion_backend(backend_type, config)
        return backend
    
