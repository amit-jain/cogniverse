"""
Cache module for Cogniverse - Plugin-based caching with multiple backends
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .backends.s3 import (
    S3CacheBackend,
    S3CacheBackendConfig,
    configured_s3_backend_defaults,
)
from .backends.structured_filesystem import StructuredFilesystemBackend
from .base import BackendConfig, CacheBackend, CacheConfig, CacheManager
from .pipeline_cache import PipelineArtifactCache
from .registry import CacheBackendRegistry


def require_s3_cache_backend_defaults(
    backends: Sequence[Mapping[str, Any]],
) -> None:
    """Raise when an enabled S3 cache backend lacks runtime MinIO defaults."""
    defaults = configured_s3_backend_defaults()
    missing_vars: set[str] = set()
    for backend in backends:
        if backend.get("backend_type") != "s3" or not backend.get("enabled", True):
            continue
        if not (backend.get("endpoint") or defaults.endpoint):
            missing_vars.add("MINIO_ENDPOINT")
        if not (backend.get("access_key") or defaults.access_key):
            missing_vars.add("MINIO_ACCESS_KEY")
        if not (backend.get("secret_key") or defaults.secret_key):
            missing_vars.add("MINIO_SECRET_KEY")
    if missing_vars:
        missing = ", ".join(sorted(missing_vars))
        raise RuntimeError(
            "S3 cache backend needs MinIO settings at startup: " + missing
        )


__all__ = [
    "CacheBackend",
    "CacheConfig",
    "CacheManager",
    "BackendConfig",
    "StructuredFilesystemBackend",
    "S3CacheBackend",
    "S3CacheBackendConfig",
    "PipelineArtifactCache",
    "CacheBackendRegistry",
    "require_s3_cache_backend_defaults",
]
