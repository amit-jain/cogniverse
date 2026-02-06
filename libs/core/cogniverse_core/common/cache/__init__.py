"""
Cache module for Cogniverse - Plugin-based caching with multiple backends
"""

from .backends.structured_filesystem import StructuredFilesystemBackend
from .base import BackendConfig, CacheBackend, CacheConfig, CacheManager
from .embedding_cache import EmbeddingCache
from .pipeline_cache import PipelineArtifactCache, VideoArtifacts
from .registry import CacheBackendRegistry

__all__ = [
    "CacheBackend",
    "CacheConfig",
    "CacheManager",
    "BackendConfig",
    "EmbeddingCache",
    "StructuredFilesystemBackend",
    "PipelineArtifactCache",
    "VideoArtifacts",
    "CacheBackendRegistry",
]
