"""Content-addressed local cache for fetched media bytes."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MediaCache:
    """Tenant-scoped, content-addressed local cache.

    Layout: ``<base_dir>/<key[:2]>/<key>/<basename>``. Original filename is preserved
    so consumers that sniff codec by extension (cv2, ffmpeg) work correctly.

    Atomicity: writes go to ``<base_dir>/.staging/<uuid>`` and are promoted into
    place via ``os.replace``.

    Eviction: entries older than ``ttl_seconds`` (by ``atime``) are dropped
    first, then LRU by ``atime`` while total bytes exceed ``max_bytes``.
    """

    STAGING_DIR_NAME = ".staging"

    def __init__(
        self,
        base_dir: Path,
        max_bytes: int = 50 * 1024**3,
        ttl_seconds: Optional[float] = None,
    ):
        self.base_dir = Path(base_dir)
        self.staging_dir = self.base_dir / self.STAGING_DIR_NAME
        self.max_bytes = int(max_bytes)
        # None or <= 0 disables age eviction (size-only).
        self.ttl_seconds = (
            float(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else None
        )
        self._lock = threading.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(uri: str, etag: Optional[str] = None) -> str:
        h = hashlib.sha256()
        h.update(uri.encode("utf-8"))
        if etag:
            h.update(b"\0")
            h.update(etag.encode("utf-8"))
        return h.hexdigest()

    def _key_to_path(self, key: str, basename: str) -> Path:
        return self.base_dir / key[:2] / key / basename

    def get(self, key: str, basename: str) -> Optional[Path]:
        p = self._key_to_path(key, basename)
        if p.exists():
            try:
                st = p.stat()
                os.utime(p, (time.time(), st.st_mtime))
            except OSError as exc:
                logger.debug("utime failed for %s: %s", p, exc)
            return p
        return None

    def staging_path(self) -> Path:
        return self.staging_dir / uuid.uuid4().hex

    def put(self, key: str, basename: str, src: Path) -> Path:
        dest = self._key_to_path(key, basename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dest)
        with self._lock:
            self._evict_if_needed()
        return dest

    def _iter_cached_files(self) -> list[Path]:
        files: list[Path] = []
        for p in self.base_dir.rglob("*"):
            if not p.is_file():
                continue
            if self.staging_dir in p.parents:
                continue
            files.append(p)
        return files

    def total_bytes(self) -> int:
        total = 0
        for p in self._iter_cached_files():
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def _unlink(self, p: Path) -> bool:
        try:
            p.unlink()
        except OSError as exc:
            logger.debug("eviction unlink failed for %s: %s", p, exc)
            return False
        try:
            p.parent.rmdir()
        except OSError:
            pass
        return True

    def _evict_if_needed(self) -> None:
        files: list[tuple[float, int, Path]] = []
        total = 0
        cutoff = (time.time() - self.ttl_seconds) if self.ttl_seconds else None
        for p in self._iter_cached_files():
            try:
                st = p.stat()
            except OSError:
                continue
            # Age eviction: drop entries not accessed within the TTL window.
            if cutoff is not None and st.st_atime < cutoff:
                if self._unlink(p):
                    continue
            files.append((st.st_atime, st.st_size, p))
            total += st.st_size

        if total <= self.max_bytes:
            return

        files.sort(key=lambda t: t[0])
        for _, size, p in files:
            if total <= self.max_bytes:
                break
            if self._unlink(p):
                total -= size
