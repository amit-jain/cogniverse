"""
Registry for cache backend plugins
"""

from typing import Dict, Type
from .base import CacheBackend, BackendConfig


class CacheBackendRegistry:
    """Registry for cache backend plugins"""
    
    _backends: Dict[str, Type[CacheBackend]] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: Type[CacheBackend]):
        """Register a new cache backend"""
        cls._backends[name] = backend_class
    
    @classmethod
    def create(cls, config: dict) -> CacheBackend:
        """Create backend instance from config"""
        backend_type = config.get('backend_type')
        backend_class = cls._backends.get(backend_type)
        if not backend_class:
            raise ValueError(f"Unknown backend type: {backend_type}")
        
        # Convert dict config to appropriate config class
        if backend_type == 'filesystem':
            from .backends.filesystem import FilesystemCacheConfig
            return backend_class(FilesystemCacheConfig(**config))
        elif backend_type == 'structured_filesystem':
            from .backends.structured_filesystem import StructuredFilesystemConfig
            return backend_class(StructuredFilesystemConfig(**config))
        else:
            # Try to pass config directly
            return backend_class(config)
    
    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backend types"""
        return list(cls._backends.keys())