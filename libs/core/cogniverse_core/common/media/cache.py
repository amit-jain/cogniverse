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

# Staging files older than this are orphans from a hard-killed process (the
# unlink-on-failure path never ran) — in-flight downloads are seconds old, so
# an hour gate never races a live write.
_STAGING_ORPHAN_MAX_AGE_S = 3600.0


class MediaCache:
    """Tenant-scoped, content-addressed local cache.

    Layout: ``<base_dir>/<key[:2]>/<key>/<basename>``. Original filename is preserved
    so consumers that sniff codec by extension (cv2, ffmpeg) work correctly.

    Atomicity: writes go to ``<base_dir>/.staging/<uuid>`` and are promoted into
    place via ``os.replace``.

    Eviction: entries older than ``ttl_seconds`` (by ``atime``) are dropped
    first, then LRU by ``atime`` while total bytes exceed ``max_bytes``.
    A running byte total keeps under-budget puts walk-free; the tree is
    walked only on the first put, when over budget, or when a TTL sweep is
    due (at most once per TTL period — an expired entry lingers at most one
    extra period).
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
        # Running byte total so an under-budget put never walks the tree
        # (the walk is O(cached files) under the lock). None until the
        # first put initializes it; eviction passes resync it exactly.
        self._total_bytes: Optional[int] = None
        self._last_ttl_sweep = 0.0
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        # A killed process orphans its staging file (invisible to eviction
        # and the byte budget); the next construction over the same base_dir
        # reaps aged orphans so they can't accumulate across restarts.
        self._reap_staging_orphans()

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
        displaced = 0
        if dest.exists():
            try:
                displaced = dest.stat().st_size
            except OSError:
                displaced = 0
        os.replace(src, dest)
        try:
            added = dest.stat().st_size
        except OSError:
            added = 0
        with self._lock:
            if self._total_bytes is None:
                self._total_bytes = self.total_bytes()
            else:
                self._total_bytes += added - displaced
            ttl_due = (
                self.ttl_seconds is not None
                and (time.time() - self._last_ttl_sweep) >= self.ttl_seconds
            )
            if ttl_due or self._total_bytes > self.max_bytes:
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

    def _reap_staging_orphans(self) -> None:
        """Unlink staging files older than the orphan gate.

        Age-gated so an in-flight download's staging file is never touched.
        """
        cutoff = time.time() - _STAGING_ORPHAN_MAX_AGE_S
        try:
            entries = list(self.staging_dir.iterdir())
        except OSError:
            return
        for p in entries:
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
                    logger.info("reaped orphaned staging file %s", p)
            except OSError as exc:
                logger.debug("staging reap failed for %s: %s", p, exc)

    def _evict_if_needed(self) -> None:
        """Full-walk pass: reap aged staging orphans, expire TTL-stale
        entries, then LRU-evict while over budget. Resyncs the running byte
        total from the walk and stamps the sweep time so put() amortizes TTL
        walks to once per TTL period.
        """
        self._reap_staging_orphans()
        files: list[tuple[float, int, Path]] = []
        total = 0
        now = time.time()
        cutoff = (now - self.ttl_seconds) if self.ttl_seconds else None
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

        if self.ttl_seconds is not None:
            self._last_ttl_sweep = now

        if total <= self.max_bytes:
            self._total_bytes = total
            return

        files.sort(key=lambda t: t[0])
        for _, size, p in files:
            if total <= self.max_bytes:
                break
            if self._unlink(p):
                total -= size
        self._total_bytes = total
