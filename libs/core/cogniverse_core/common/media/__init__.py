"""Media access abstraction.

URI -> local Path resolver, fsspec-backed and tenant-scoped. Used by both
ingestion (write side, populating ``source_url``) and evaluation (read side,
fetching frames for the visual judge).
"""

from .cache import MediaCache
from .config import (
    HttpBackendConfig,
    MediaCacheConfig,
    MediaConfig,
    S3BackendConfig,
)
from .locator import (
    DEFAULT_VIDEO_EXTENSIONS,
    LOCAL_SCHEMES,
    NETWORK_SCHEMES,
    PVC_SCHEME,
    MediaLocator,
    MediaStat,
)

__all__ = [
    "DEFAULT_VIDEO_EXTENSIONS",
    "HttpBackendConfig",
    "LOCAL_SCHEMES",
    "MediaCache",
    "MediaCacheConfig",
    "MediaConfig",
    "MediaLocator",
    "MediaStat",
    "NETWORK_SCHEMES",
    "PVC_SCHEME",
    "S3BackendConfig",
]
