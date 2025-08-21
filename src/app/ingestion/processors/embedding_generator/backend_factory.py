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
        
        Args:
            backend_type: Type of backend (e.g., "vespa")
            config: Full application config
            logger: Optional logger
            
        Returns:
            IngestionBackend: The backend that implements the ingestion interface
        """
        
        backend_type = backend_type.lower()
        
        # CRITICAL: Log backend creation to track when new instances are made
        if not logger:
            logger = logging.getLogger(__name__)
        
        logger.warning(f"🏭 BACKEND FACTORY: Getting {backend_type} backend")
        logger.warning(f"   Config keys: {list(config.keys())[:10]}")
        
        # Get backend from registry (singleton)
        registry = get_backend_registry()
        
        backend = registry.get_ingestion_backend(backend_type, config)
        
        logger.warning(f"   Got backend instance: {id(backend)}")
        
        return backend
    
