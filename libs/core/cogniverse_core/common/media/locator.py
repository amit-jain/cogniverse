"""URI -> local Path resolver, fsspec-backed and tenant-scoped.

The :class:`MediaLocator` is the single abstraction both ingestion (write side,
populating Vespa ``source_url``) and evaluation (read side, fetching frames for
the visual judge) consume. It dispatches by URI scheme:

- ``file://<path>`` and bare paths: short-circuit to the local path with no copy.
- ``pvc://<volume>/<rest>``: translates to ``<config.pvc_mount_root>/<volume>/<rest>``.
- ``s3://``, ``http(s)://``: fetched via fsspec, cached in a tenant-scoped local
  directory under :class:`~.cache.MediaCache`.

cv2/ffmpeg/whisper consumers want a real local path string; ``localize()`` always
returns one.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterator, Optional
from urllib.parse import urlparse

from cogniverse_core.common.tenant_utils import get_tenant_storage_path

from .cache import MediaCache
from .config import MediaConfig

logger = logging.getLogger(__name__)

LOCAL_SCHEMES = frozenset({"file", ""})
PVC_SCHEME = "pvc"
NETWORK_SCHEMES = frozenset({"s3", "gs", "az", "http", "https"})
DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
)


@dataclass(frozen=True)
class MediaStat:
    size: int
    etag: Optional[str] = None
    last_modified: Optional[float] = None
    content_type: Optional[str] = None


class MediaLocator:
    def __init__(
        self,
        tenant_id: str,
        config: MediaConfig,
        cache_root: Optional[Path] = None,
    ):
        self.tenant_id = tenant_id
        self.config = config

        configured = config.cache.base_dir
        if cache_root is not None:
            base = Path(cache_root)
        elif configured:
            base = Path(configured)
        else:
            base = Path(tempfile.gettempdir()) / "cogniverse-media-cache"

        tenant_cache = get_tenant_storage_path(base, tenant_id) / "media"
        max_bytes = int(config.cache.max_bytes_gb) * 1024**3
        self.cache = MediaCache(tenant_cache, max_bytes=max_bytes)

    @staticmethod
    def _scheme(uri: str) -> str:
        return urlparse(uri).scheme

    def to_canonical_uri(self, raw: str) -> str:
        """Normalize a path or URI to a canonical URI string.

        - Already a URI (contains ``://``): returned unchanged.
        - ``default_uri_scheme=="file"`` and no ``uri_prefix``: returns
          ``file://<absolute>``.
        - ``uri_prefix`` set: prepended to the path. Absolute local paths reduce
          to their basename; relative paths preserve their structure.
        - Otherwise: ``<default_uri_scheme>://<basename>``.
        """
        if "://" in raw:
            return raw

        prefix = self.config.uri_prefix
        if prefix:
            prefix_normalized = prefix.rstrip("/") + "/"
            path = Path(raw)
            if path.is_absolute():
                return prefix_normalized + path.name
            return prefix_normalized + str(path).lstrip("/")

        scheme = self.config.default_uri_scheme
        if scheme == "file":
            return f"file://{Path(raw).resolve()}"
        return f"{scheme}://{Path(raw).name}"

    def localize(self, uri: str) -> Path:
        scheme = self._scheme(uri)
        if scheme in LOCAL_SCHEMES:
            return self._localize_file(uri)
        if scheme == PVC_SCHEME:
            return self._localize_pvc(uri)
        if scheme in NETWORK_SCHEMES:
            return self._localize_network(uri)
        raise ValueError(f"Unsupported URI scheme: {scheme!r} (uri={uri!r})")

    def _localize_file(self, uri: str) -> Path:
        path_str = uri[len("file://") :] if uri.startswith("file://") else uri
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Local media not found: {p}")
        return p

    def _localize_pvc(self, uri: str) -> Path:
        parsed = urlparse(uri)
        volume = parsed.netloc
        rest = parsed.path.lstrip("/")
        if not volume:
            raise ValueError(f"pvc URI missing volume name: {uri!r}")
        p = Path(self.config.pvc_mount_root) / volume / rest
        if not p.exists():
            raise FileNotFoundError(f"PVC-mounted media not found: {p}")
        return p

    def _localize_network(self, uri: str) -> Path:
        basename = Path(urlparse(uri).path).name or "blob"

        stat = self._stat_remote(uri)
        key = MediaCache.make_key(uri, etag=stat.etag if stat else None)

        cached = self.cache.get(key, basename)
        if cached is not None:
            logger.debug("media cache hit: %s", uri)
            return cached

        staging = self.cache.staging_path()
        try:
            self._download_to(uri, staging)
            return self.cache.put(key, basename, staging)
        except Exception:
            if staging.exists():
                try:
                    staging.unlink()
                except OSError:
                    pass
            raise

    def _download_to(self, uri: str, dest: Path) -> None:
        import fsspec

        fs_kwargs = self._fs_kwargs_for(uri)
        fs, path = fsspec.core.url_to_fs(uri, **fs_kwargs)
        try:
            fs.get_file(path, str(dest))
            return
        except (NotImplementedError, AttributeError):
            with (
                fsspec.open(uri, mode="rb", **fs_kwargs) as src,
                open(dest, "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)

    def _fs_kwargs_for(self, uri: str) -> dict[str, Any]:
        scheme = self._scheme(uri)
        if scheme == "s3":
            kwargs: dict[str, Any] = {}
            s3 = self.config.s3
            client_kwargs: dict[str, Any] = {}
            if s3.endpoint_url:
                client_kwargs["endpoint_url"] = s3.endpoint_url
            if s3.region:
                client_kwargs["region_name"] = s3.region
            if client_kwargs:
                kwargs["client_kwargs"] = client_kwargs
            if s3.anon:
                kwargs["anon"] = True
            return kwargs
        return {}

    def _stat_remote(self, uri: str) -> Optional[MediaStat]:
        import fsspec

        try:
            fs_kwargs = self._fs_kwargs_for(uri)
            fs, path = fsspec.core.url_to_fs(uri, **fs_kwargs)
            info = fs.info(path)
        except Exception as exc:
            logger.debug("stat failed for %s: %s", uri, exc)
            return None

        size = int(info.get("size", 0))
        etag = info.get("ETag") or info.get("etag")
        last_modified = info.get("LastModified") or info.get("mtime")
        if hasattr(last_modified, "timestamp"):
            last_modified = last_modified.timestamp()
        return MediaStat(
            size=size,
            etag=str(etag) if etag else None,
            last_modified=float(last_modified) if last_modified is not None else None,
            content_type=info.get("ContentType"),
        )

    def stat(self, uri: str) -> MediaStat:
        scheme = self._scheme(uri)
        if scheme in LOCAL_SCHEMES or scheme == PVC_SCHEME:
            p = self.localize(uri)
            st = p.stat()
            return MediaStat(size=st.st_size, last_modified=st.st_mtime)
        s = self._stat_remote(uri)
        if s is None:
            raise FileNotFoundError(uri)
        return s

    def exists(self, uri: str) -> bool:
        try:
            self.stat(uri)
            return True
        except (FileNotFoundError, OSError, ValueError):
            return False

    def open(self, uri: str, mode: str = "rb") -> IO[Any]:
        return open(self.localize(uri), mode)

    def list(
        self,
        prefix_uri: str,
        extensions: Optional[tuple[str, ...]] = None,
    ) -> Iterator[str]:
        exts = extensions if extensions is not None else DEFAULT_VIDEO_EXTENSIONS
        scheme = self._scheme(prefix_uri)

        if scheme in LOCAL_SCHEMES:
            base_str = (
                prefix_uri[len("file://") :]
                if prefix_uri.startswith("file://")
                else prefix_uri
            )
            base = Path(base_str)
            for ext in exts:
                for p in sorted(base.rglob(f"*{ext}")):
                    if p.is_file():
                        yield f"file://{p.resolve()}"
            return

        if scheme == PVC_SCHEME:
            parsed = urlparse(prefix_uri)
            mount_root = Path(self.config.pvc_mount_root) / parsed.netloc
            base = mount_root / parsed.path.lstrip("/")
            for ext in exts:
                for p in sorted(base.rglob(f"*{ext}")):
                    if p.is_file():
                        rel = p.relative_to(mount_root)
                        yield f"pvc://{parsed.netloc}/{rel}"
            return

        if scheme in NETWORK_SCHEMES:
            import fsspec

            fs_kwargs = self._fs_kwargs_for(prefix_uri)
            fs, path = fsspec.core.url_to_fs(prefix_uri, **fs_kwargs)
            for entry in fs.find(path):
                if any(entry.endswith(ext) for ext in exts):
                    yield f"{scheme}://{entry}"
            return

        raise ValueError(f"Unsupported URI scheme for list: {scheme!r}")
