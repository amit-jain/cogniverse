"""
Base classes for the caching system
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache with optional TTL in seconds"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass
    
    # Optional methods that backends can override
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a key (optional)"""
        return None
    
    async def list_keys(self, pattern: Optional[str] = None, include_metadata: bool = False) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """List keys matching pattern (optional)"""
        return []
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries (optional)"""
        return 0


@dataclass
class BackendConfig:
    """Base configuration for cache backends"""
    backend_type: str
    enabled: bool = True
    priority: int = 0  # Lower numbers = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class CacheConfig:
    """Main cache configuration"""
    backends: List[BackendConfig]
    default_ttl: int = 3600  # 1 hour
    enable_stats: bool = True
    enable_compression: bool = True
    serialization_format: str = "pickle"  # or "json", "msgpack"


class CacheManager:
    """Manages multiple cache backends with tiered caching"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.backends: List[CacheBackend] = []
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize cache backends from config"""
        from .registry import CacheBackendRegistry
        
        # Import backends to ensure they're registered
        from .backends import structured_filesystem
        
        # Sort by priority (lower number = higher priority)
        sorted_configs = sorted(
            [b for b in self.config.backends if b.get('enabled', True)],
            key=lambda x: x.get('priority', 0)
        )
        
        for backend_config in sorted_configs:
            try:
                backend = CacheBackendRegistry.create(backend_config)
                self.backends.append(backend)
                logger.info(f"Initialized {backend_config.get('backend_type', 'unknown')} cache backend")
            except Exception as e:
                logger.error(f"Failed to initialize {backend_config.get('backend_type', 'unknown')}: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache, checking tiers in order"""
        for i, backend in enumerate(self.backends):
            try:
                value = await backend.get(key)
                if value is not None:
                    self._stats['hits'] += 1
                    # Populate higher priority tiers
                    await self._populate_higher_tiers(key, value, i)
                    return value
            except Exception as e:
                logger.warning(f"Error getting from {backend.__class__.__name__}: {e}")
        
        self._stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in all configured cache tiers"""
        if ttl is None:
            ttl = self.config.default_ttl
        
        results = []
        for backend in self.backends:
            try:
                result = await backend.set(key, value, ttl)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error setting in {backend.__class__.__name__}: {e}")
                results.append(False)
        
        if any(results):
            self._stats['sets'] += 1
        
        return any(results)  # Success if at least one backend succeeded
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache tiers"""
        results = []
        for backend in self.backends:
            try:
                result = await backend.delete(key)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error deleting from {backend.__class__.__name__}: {e}")
                results.append(False)
        
        if any(results):
            self._stats['deletes'] += 1
        
        return any(results)
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear matching entries from all tiers"""
        total_cleared = 0
        for backend in self.backends:
            try:
                cleared = await backend.clear(pattern)
                total_cleared += cleared
            except Exception as e:
                logger.warning(f"Error clearing {backend.__class__.__name__}: {e}")
        
        return total_cleared
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        stats = {
            'manager': self._stats.copy(),
            'backends': {}
        }
        
        # Aggregate size from backends
        total_size = 0
        total_files = 0
        
        for backend in self.backends:
            backend_name = backend.__class__.__name__
            try:
                backend_stats = await backend.get_stats()
                stats['backends'][backend_name] = backend_stats
                # Aggregate size if available
                if 'size_bytes' in backend_stats:
                    total_size += backend_stats['size_bytes']
                if 'total_files' in backend_stats:
                    total_files += backend_stats.get('total_files', 0)
            except Exception as e:
                logger.warning(f"Error getting stats from {backend_name}: {e}")
                stats['backends'][backend_name] = {'error': str(e)}
        
        # Add aggregated size to manager stats
        stats['manager']['size_bytes'] = total_size
        stats['manager']['total_files'] = total_files
        
        # Calculate hit rate
        total_requests = self._stats['hits'] + self._stats['misses']
        if total_requests > 0:
            stats['manager']['hit_rate'] = self._stats['hits'] / total_requests
        else:
            stats['manager']['hit_rate'] = 0.0
        
        return stats
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries from all backends"""
        total_cleaned = 0
        
        for backend in self.backends:
            try:
                cleaned = await backend.cleanup_expired()
                total_cleaned += cleaned
            except Exception as e:
                logger.warning(f"Error cleaning expired from {backend.__class__.__name__}: {e}")
        
        return total_cleaned
    
    async def list_keys(self, pattern: Optional[str] = None, include_metadata: bool = False) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """List all keys from primary backend"""
        if self.backends:
            try:
                return await self.backends[0].list_keys(pattern, include_metadata)
            except Exception as e:
                logger.warning(f"Error listing keys: {e}")
        
        return []
    
    async def _populate_higher_tiers(self, key: str, value: Any, backend_index: int):
        """Populate higher priority cache tiers with the found value"""
        # Populate caches with higher priority (lower index)
        for i in range(backend_index):
            try:
                await self.backends[i].set(key, value, self.config.default_ttl)
            except Exception as e:
                logger.debug(f"Failed to populate tier {i}: {e}")