"""Configuration for the media access subsystem.

Loaded from the ``media`` section of ``configs/config.json``. All fields are
optional — defaults give a working ``file://``-only setup for local development.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class S3BackendConfig:
    """Settings for the s3:// scheme (works with AWS S3 and S3-compatible stores like MinIO)."""

    endpoint_url: Optional[str] = None
    region: str = "us-east-1"
    anon: bool = False


@dataclass(frozen=True)
class HttpBackendConfig:
    """Settings for the http:// and https:// schemes."""

    timeout_s: int = 60


@dataclass(frozen=True)
class MediaCacheConfig:
    """Local cache settings. ``base_dir=None`` resolves to a tenant-scoped tempdir."""

    base_dir: Optional[str] = None
    max_bytes_gb: int = 50
    ttl_days: int = 7


@dataclass(frozen=True)
class MediaConfig:
    default_uri_scheme: str = "file"
    uri_prefix: str = ""
    pvc_mount_root: str = "/mnt"
    cache: MediaCacheConfig = field(default_factory=MediaCacheConfig)
    s3: S3BackendConfig = field(default_factory=S3BackendConfig)
    http: HttpBackendConfig = field(default_factory=HttpBackendConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaConfig":
        cache = MediaCacheConfig(**data.get("cache", {}))
        backends = data.get("backends", {})
        s3 = S3BackendConfig(**backends.get("s3", {}))
        http = HttpBackendConfig(**backends.get("http", {}))
        return cls(
            default_uri_scheme=data.get("default_uri_scheme", "file"),
            uri_prefix=data.get("uri_prefix", ""),
            pvc_mount_root=data.get("pvc_mount_root", "/mnt"),
            cache=cache,
            s3=s3,
            http=http,
        )
