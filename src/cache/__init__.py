"""
Cache module for Cogniverse - Plugin-based caching with multiple backends
"""

from .base import CacheBackend, CacheConfig, CacheManager, BackendConfig
from .backends.structured_filesystem import StructuredFilesystemBackend
from .pipeline_cache import PipelineArtifactCache
from .registry import CacheBackendRegistry

__all__ = [
    'CacheBackend',
    'CacheConfig',
    'CacheManager',
    'BackendConfig',
    'StructuredFilesystemBackend',
    'PipelineArtifactCache',
    'CacheBackendRegistry'
]