"""
Backend Registry - Auto-discovery and registration of backend implementations.

This module provides a registry pattern for backends to self-register,
enabling a pluggable architecture where new backends can be added without
modifying core code.
"""

import importlib
import logging
from typing import Dict, Type, Optional, Any
from pathlib import Path

from .interfaces import IngestionBackend, SearchBackend, Backend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Central registry for backend auto-discovery and management.
    
    Backends self-register when imported, allowing the app layer
    to discover and use them without direct imports.
    """
    
    _instance = None
    _ingestion_backends: Dict[str, Type[IngestionBackend]] = {}
    _search_backends: Dict[str, Type[SearchBackend]] = {}
    _full_backends: Dict[str, Type[Backend]] = {}
    _backend_instances: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register_ingestion(cls, name: str, backend_class: Type[IngestionBackend]) -> None:
        """
        Register an ingestion backend.
        
        Args:
            name: Backend name (e.g., "vespa", "milvus")
            backend_class: Backend class implementing IngestionBackend
        """
        cls._ingestion_backends[name] = backend_class
        logger.info(f"Registered ingestion backend: {name}")
    
    @classmethod
    def register_search(cls, name: str, backend_class: Type[SearchBackend]) -> None:
        """
        Register a search backend.
        
        Args:
            name: Backend name
            backend_class: Backend class implementing SearchBackend
        """
        cls._search_backends[name] = backend_class
        logger.info(f"Registered search backend: {name}")
    
    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]) -> None:
        """
        Register a full backend (supports both ingestion and search).
        
        Args:
            name: Backend name
            backend_class: Backend class implementing Backend
        """
        cls._full_backends[name] = backend_class
        # Also register in individual registries
        cls._ingestion_backends[name] = backend_class
        cls._search_backends[name] = backend_class
        logger.info(f"Registered full backend: {name}")
    
    @classmethod
    def get_ingestion_backend(
        cls, 
        name: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> IngestionBackend:
        """
        Get an ingestion backend instance.
        
        Args:
            name: Backend name
            config: Optional configuration for initialization
            
        Returns:
            Initialized IngestionBackend instance
            
        Raises:
            ValueError: If backend not found
        """
        # For full backends, use unified cache key
        if name in cls._full_backends:
            instance_key = f"backend_{name}"
        else:
            instance_key = f"ingestion_{name}"
            
        # Check if instance already exists
        if instance_key in cls._backend_instances:
            return cls._backend_instances[instance_key]
        
        # Try to auto-import if not registered
        if name not in cls._ingestion_backends:
            cls._try_import_backend(name)
        
        if name not in cls._ingestion_backends:
            available = list(cls._ingestion_backends.keys())
            raise ValueError(
                f"Ingestion backend '{name}' not found. "
                f"Available backends: {available}"
            )
        
        # Create and initialize instance
        backend_class = cls._ingestion_backends[name]
        instance = backend_class()
        
        if config:
            instance.initialize(config)
        
        # Cache instance
        cls._backend_instances[instance_key] = instance
        
        return instance
    
    @classmethod
    def get_search_backend(
        cls, 
        name: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> SearchBackend:
        """
        Get a search backend instance.
        
        Args:
            name: Backend name
            config: Optional configuration for initialization
            
        Returns:
            Initialized SearchBackend instance
            
        Raises:
            ValueError: If backend not found
        """
        # For full backends, use unified cache key
        if name in cls._full_backends:
            instance_key = f"backend_{name}"
        else:
            instance_key = f"search_{name}"
            
        # Check if instance already exists
        if instance_key in cls._backend_instances:
            return cls._backend_instances[instance_key]
        
        # Try to auto-import if not registered
        if name not in cls._search_backends:
            cls._try_import_backend(name)
        
        if name not in cls._search_backends:
            available = list(cls._search_backends.keys())
            raise ValueError(
                f"Search backend '{name}' not found. "
                f"Available backends: {available}"
            )
        
        # Create and initialize instance
        backend_class = cls._search_backends[name]
        instance = backend_class()
        
        if config:
            instance.initialize(config)
        
        # Cache instance
        cls._backend_instances[instance_key] = instance
        
        return instance
    
    @classmethod
    def _try_import_backend(cls, name: str) -> None:
        """
        Try to import a backend module to trigger self-registration.
        
        Args:
            name: Backend name to import
        """
        try:
            # Try standard backend location
            module_path = f"src.backends.{name}"
            importlib.import_module(module_path)
            logger.info(f"Auto-imported backend module: {module_path}")
        except ImportError:
            # Try with underscore (e.g., "vespa_backend")
            try:
                module_path = f"src.backends.{name}_backend"
                importlib.import_module(module_path)
                logger.info(f"Auto-imported backend module: {module_path}")
            except ImportError:
                logger.debug(f"Could not auto-import backend: {name}")
    
    @classmethod
    def list_ingestion_backends(cls) -> list:
        """List all registered ingestion backends."""
        return list(cls._ingestion_backends.keys())
    
    @classmethod
    def list_search_backends(cls) -> list:
        """List all registered search backends."""
        return list(cls._search_backends.keys())
    
    @classmethod
    def list_full_backends(cls) -> list:
        """List all registered full backends."""
        return list(cls._full_backends.keys())
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached backend instances."""
        cls._backend_instances.clear()
        logger.info("Cleared all backend instances")
    
    @classmethod
    def is_registered(cls, name: str, backend_type: str = "any") -> bool:
        """
        Check if a backend is registered.
        
        Args:
            name: Backend name
            backend_type: Type to check ("ingestion", "search", "full", "any")
            
        Returns:
            True if registered, False otherwise
        """
        if backend_type == "ingestion":
            return name in cls._ingestion_backends
        elif backend_type == "search":
            return name in cls._search_backends
        elif backend_type == "full":
            return name in cls._full_backends
        else:  # "any"
            return (
                name in cls._ingestion_backends or 
                name in cls._search_backends or 
                name in cls._full_backends
            )


# Global registry instance
_backend_registry = BackendRegistry()


def get_backend_registry() -> BackendRegistry:
    """Get the global BackendRegistry instance."""
    return _backend_registry


def register_ingestion_backend(name: str, backend_class: Type[IngestionBackend]) -> None:
    """
    Convenience function to register an ingestion backend.
    
    Args:
        name: Backend name
        backend_class: Backend class implementing IngestionBackend
    """
    _backend_registry.register_ingestion(name, backend_class)


def register_search_backend(name: str, backend_class: Type[SearchBackend]) -> None:
    """
    Convenience function to register a search backend.
    
    Args:
        name: Backend name
        backend_class: Backend class implementing SearchBackend
    """
    _backend_registry.register_search(name, backend_class)


def register_backend(name: str, backend_class: Type[Backend]) -> None:
    """
    Convenience function to register a full backend.
    
    Args:
        name: Backend name
        backend_class: Backend class implementing Backend
    """
    _backend_registry.register_backend(name, backend_class)