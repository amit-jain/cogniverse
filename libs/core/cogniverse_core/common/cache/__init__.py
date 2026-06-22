"""
Cache module for Cogniverse - Plugin-based caching with multiple backends
"""

from .backends.s3 import S3CacheBackend, S3CacheBackendConfig
from .backends.structured_filesystem import StructuredFilesystemBackend
from .base import BackendConfig, CacheBackend, CacheConfig, CacheManager
from .pipeline_cache import PipelineArtifactCache, VideoArtifacts
from .registry import CacheBackendRegistry

__all__ = [
    "CacheBackend",
    "CacheConfig",
    "CacheManager",
    "BackendConfig",
    "StructuredFilesystemBackend",
    "S3CacheBackend",
    "S3CacheBackendConfig",
    "PipelineArtifactCache",
    "VideoArtifacts",
    "CacheBackendRegistry",
]
