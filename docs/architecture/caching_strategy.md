# Production-Ready Caching Strategy

## Overview
Plugin-based caching architecture for video processing pipeline with support for multiple backends.

## Plugin Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Retrieve data from cache"""
        pass
    
    @abstractmethod
    async def put(self, key: str, data: bytes, metadata: Dict[str, Any] = None) -> bool:
        """Store data in cache with optional metadata"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove data from cache"""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str) -> List[str]:
        """List all keys with given prefix"""
        pass
```

## Backend Implementations

### 1. Local Filesystem Backend
```python
class LocalFSBackend(CacheBackend):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        
    async def exists(self, key: str) -> bool:
        return (self.base_path / key).exists()
```

### 2. S3 Backend
```python
class S3Backend(CacheBackend):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self.client = boto3.client('s3')
```

### 3. GCS Backend
```python
class GCSBackend(CacheBackend):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self.client = storage.Client()
```

### 4. Redis Backend (for small artifacts)
```python
class RedisBackend(CacheBackend):
    def __init__(self, redis_url: str, ttl: int = 86400):
        self.redis = aioredis.from_url(redis_url)
        self.ttl = ttl
```

## Cache Manager

```python
class CacheManager:
    """Manages multiple cache backends with fallback support"""
    
    def __init__(self, primary: CacheBackend, fallback: Optional[CacheBackend] = None):
        self.primary = primary
        self.fallback = fallback
        
    async def get_or_compute(self, key: str, compute_fn: Callable) -> Any:
        """Get from cache or compute and store"""
        # Check primary
        data = await self.primary.get(key)
        if data:
            return data
            
        # Check fallback
        if self.fallback:
            data = await self.fallback.get(key)
            if data:
                # Promote to primary
                await self.primary.put(key, data)
                return data
        
        # Compute and store
        data = await compute_fn()
        await self.primary.put(key, data)
        return data
```

## Key Structure

```
profile_{profile_name}/
├── videos/
│   └── {video_id}/
│       ├── metadata.json
│       ├── keyframes/
│       │   └── frame_{timestamp}.jpg
│       ├── audio/
│       │   └── transcription.json
│       ├── descriptions/
│       │   └── frame_{timestamp}.json
│       └── embeddings/
│           └── {model_name}.npz
```

## Configuration

```yaml
caching:
  enabled: true
  backends:
    primary:
      type: "s3"
      config:
        bucket: "video-processing-cache"
        prefix: "cogniverse/v1"
        region: "us-east-1"
    fallback:
      type: "local"
      config:
        base_path: "/tmp/cogniverse-cache"
  policies:
    ttl_days: 30
    max_size_gb: 1000
    eviction: "lru"
```

## Usage in Pipeline

```python
class VideoProcessor:
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache = cache_manager
        
    async def process_video(self, video_path: Path):
        video_id = extract_video_id(video_path)
        
        # Check if already processed
        if self.cache:
            key = f"profile_{self.profile}/videos/{video_id}/metadata.json"
            if await self.cache.exists(key):
                logger.info(f"Skipping {video_id} - already processed")
                return
        
        # Process video
        results = await self._process(video_path)
        
        # Cache results
        if self.cache:
            await self._cache_results(video_id, results)
```

## Benefits

1. **Pluggable**: Easy to add new backends
2. **Testable**: Mock backends for testing
3. **Scalable**: Distributed caching with Redis/S3
4. **Fault-tolerant**: Fallback support
5. **Cost-effective**: Tiered storage (hot/cold)
6. **Production-ready**: Lifecycle policies, monitoring

## Implementation Timeline

1. Phase 1: Define interfaces and local backend
2. Phase 2: Add S3/GCS backends
3. Phase 3: Add Redis for metadata
4. Phase 4: Implement lifecycle policies
5. Phase 5: Add monitoring and metrics