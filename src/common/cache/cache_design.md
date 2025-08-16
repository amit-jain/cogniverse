# Plugin-Based Caching Strategy Design

## Overview
This document outlines the design for a plugin-based caching system that supports multiple storage backends including local filesystem and object storage (S3, GCS, Azure).

## Architecture

### Core Components

#### 1. Cache Interface
```python
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from datetime import datetime

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
```

#### 2. Cache Manager
```python
class CacheManager:
    """Manages multiple cache backends with tiered caching"""
    
    def __init__(self, config: CacheConfig):
        self.backends: List[CacheBackend] = []
        self.config = config
        self._initialize_backends()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache, checking tiers in order"""
        for backend in self.backends:
            value = await backend.get(key)
            if value is not None:
                # Populate higher tiers
                await self._populate_higher_tiers(key, value, backend)
                return value
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in all configured cache tiers"""
        results = []
        for backend in self.backends:
            result = await backend.set(key, value, ttl)
            results.append(result)
        return all(results)
```

### Storage Backends

#### 1. Local Filesystem Backend
```python
class FilesystemCacheBackend(CacheBackend):
    """Local filesystem cache backend"""
    
    def __init__(self, config: FilesystemCacheConfig):
        self.base_path = Path(config.base_path)
        self.max_size_mb = config.max_size_mb
        self.eviction_policy = config.eviction_policy  # LRU, FIFO, etc.
        self._ensure_directory()
    
    async def get(self, key: str) -> Optional[Any]:
        file_path = self._get_file_path(key)
        if file_path.exists():
            # Check TTL
            if self._is_expired(file_path):
                await self.delete(key)
                return None
            
            # Update access time for LRU
            file_path.touch()
            
            # Read and deserialize
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None
```

#### 2. S3 Backend
```python
class S3CacheBackend(CacheBackend):
    """AWS S3 cache backend"""
    
    def __init__(self, config: S3CacheConfig):
        self.bucket = config.bucket
        self.prefix = config.prefix
        self.s3_client = boto3.client('s3', **config.aws_config)
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/{key}"
            )
            
            # Check metadata for TTL
            metadata = response.get('Metadata', {})
            if 'ttl' in metadata:
                ttl_timestamp = int(metadata['ttl'])
                if time.time() > ttl_timestamp:
                    await self.delete(key)
                    return None
            
            # Deserialize and return
            return pickle.loads(response['Body'].read())
        except self.s3_client.exceptions.NoSuchKey:
            return None
```

#### 3. Redis Backend (for comparison)
```python
class RedisCacheBackend(CacheBackend):
    """Redis cache backend for fast access"""
    
    def __init__(self, config: RedisCacheConfig):
        self.redis = aioredis.from_url(config.redis_url)
        self.prefix = config.prefix
    
    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(f"{self.prefix}:{key}")
        if value:
            return pickle.loads(value)
        return None
```

### Configuration

```python
@dataclass
class CacheConfig:
    """Main cache configuration"""
    backends: List[BackendConfig]
    default_ttl: int = 3600  # 1 hour
    enable_stats: bool = True
    enable_compression: bool = True
    serialization_format: str = "pickle"  # or "json", "msgpack"

@dataclass 
class FilesystemCacheConfig(BackendConfig):
    """Filesystem backend configuration"""
    backend_type: str = "filesystem"
    base_path: str = "~/.cache/cogniverse"
    max_size_mb: int = 1000
    eviction_policy: str = "lru"
    
@dataclass
class S3CacheConfig(BackendConfig):
    """S3 backend configuration"""
    backend_type: str = "s3"
    bucket: str
    prefix: str = "cache"
    aws_config: Dict[str, Any] = field(default_factory=dict)
```

### Use Cases

#### 1. Embedding Cache
```python
class EmbeddingCache:
    """Specialized cache for embeddings"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.stats = CacheStats()
    
    async def get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        key = self._generate_key(text, model)
        embedding = await self.cache.get(key)
        
        if embedding is not None:
            self.stats.record_hit()
            return np.frombuffer(embedding, dtype=np.float32)
        
        self.stats.record_miss()
        return None
    
    async def set_embedding(self, text: str, model: str, embedding: np.ndarray):
        key = self._generate_key(text, model)
        # Convert to bytes for efficient storage
        value = embedding.tobytes()
        await self.cache.set(key, value, ttl=86400)  # 24 hours
```

#### 2. Query Result Cache
```python
class QueryResultCache:
    """Cache for search query results"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    async def get_results(
        self, 
        query: str, 
        filters: Dict[str, Any],
        ranking_strategy: str
    ) -> Optional[List[SearchResult]]:
        key = self._generate_key(query, filters, ranking_strategy)
        return await self.cache.get(key)
```

### Plugin System

```python
class CacheBackendRegistry:
    """Registry for cache backend plugins"""
    
    _backends: Dict[str, Type[CacheBackend]] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: Type[CacheBackend]):
        """Register a new cache backend"""
        cls._backends[name] = backend_class
    
    @classmethod
    def create(cls, config: BackendConfig) -> CacheBackend:
        """Create backend instance from config"""
        backend_class = cls._backends.get(config.backend_type)
        if not backend_class:
            raise ValueError(f"Unknown backend type: {config.backend_type}")
        return backend_class(config)

# Register built-in backends
CacheBackendRegistry.register("filesystem", FilesystemCacheBackend)
CacheBackendRegistry.register("s3", S3CacheBackend)
CacheBackendRegistry.register("redis", RedisCacheBackend)
```

### Example Usage

```python
# Configuration
cache_config = CacheConfig(
    backends=[
        FilesystemCacheConfig(
            base_path="/tmp/cogniverse_cache",
            max_size_mb=500
        ),
        S3CacheConfig(
            bucket="cogniverse-cache",
            prefix="embeddings",
            aws_config={"region_name": "us-west-2"}
        )
    ],
    default_ttl=3600,
    enable_compression=True
)

# Initialize cache manager
cache_manager = CacheManager(cache_config)

# Use in VespaSearchBackend
class VespaSearchBackend(SearchBackend):
    def __init__(self, ...):
        self.embedding_cache = EmbeddingCache(cache_manager)
    
    async def search(self, ...):
        # Check cache first
        if query_embeddings is None:
            cached = await self.embedding_cache.get_embedding(
                query_text, 
                self.profile
            )
            if cached is not None:
                query_embeddings = cached
            else:
                # Generate and cache
                query_embeddings = encoder.encode(query_text)
                await self.embedding_cache.set_embedding(
                    query_text,
                    self.profile, 
                    query_embeddings
                )
```

## Benefits

1. **Flexibility**: Easy to add new storage backends
2. **Performance**: Tiered caching with fast local cache backed by durable storage
3. **Cost-effective**: Use cheaper object storage for less frequently accessed data
4. **Scalability**: Distributed caching across multiple nodes
5. **Fault tolerance**: Fallback to other tiers if one fails

## Implementation Plan

1. **Phase 1**: Core interfaces and filesystem backend
2. **Phase 2**: S3 backend and compression support
3. **Phase 3**: Redis backend and distributed cache coordination
4. **Phase 4**: Cache warming and predictive prefetching