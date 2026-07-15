"""
Structured filesystem cache backend that preserves human-readable paths
"""

import asyncio
import gzip
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import msgpack

from ..base import CacheBackend

logger = logging.getLogger(__name__)

# Expiry is encoded in each cache file's mtime (set() os.utime's it to
# expires_at) instead of a per-entry ``.meta`` sidecar — that sidecar doubled
# the filesystem writes on every cached frame during ingestion. A no-ttl entry
# gets this far-future mtime so it never expires. Entries written before this
# change still carry a legacy ``.meta`` sidecar and are read via it (see
# ``_read_expiry``), so an upgrade does not invalidate a warm cache.
_NEVER_EXPIRES = 4102444800.0  # 2100-01-01 UTC


@dataclass
class StructuredFilesystemConfig:
    """Configuration for structured filesystem cache backend"""

    backend_type: str = "structured_filesystem"
    base_path: str = "~/.cache/cogniverse/pipeline"
    serialization_format: str = "pickle"  # or "json", "msgpack"
    enable_compression: bool = True
    enabled: bool = True
    priority: int = 0
    enable_ttl: bool = True  # Whether to enforce TTL
    cleanup_on_startup: bool = True  # Clean expired items on startup
    metadata_format: str = "json"  # Metadata always in JSON for portability


class StructuredFilesystemBackend(CacheBackend):
    """
    Filesystem cache that preserves human-readable directory structure.
    Instead of hash-based paths, uses actual keys as paths.
    """

    CONFIG_CLASS = StructuredFilesystemConfig

    def __init__(self, config: StructuredFilesystemConfig):
        self.config = config
        self.base_path = Path(config.base_path).expanduser()
        self.format = config.serialization_format
        self.enable_compression = config.enable_compression

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "size_bytes": 0,
            "total_files": 0,
        }

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Cleanup expired items on startup if enabled
        if config.cleanup_on_startup and config.enable_ttl:
            # Schedule cleanup to run later when event loop is available
            self._needs_cleanup = True
        else:
            self._needs_cleanup = False

    def _sanitize_path_component(self, component: str) -> str:
        """Sanitize a path component to be filesystem-safe"""
        # Replace problematic characters
        replacements = {
            ":": "_",
            "/": "_",
            "\\": "_",
            "|": "_",
            "?": "_",
            "*": "_",
            "<": "_",
            ">": "_",
            '"': "_",
            "\n": "_",
            "\r": "_",
            "\t": "_",
        }

        for old, new in replacements.items():
            component = component.replace(old, new)

        # Limit length to avoid filesystem limits
        if len(component) > 200:
            component = component[:200]

        return component

    def _key_to_path(self, key: str) -> Path:
        """
        Convert cache key to filesystem path.

        Examples:
        - profile:video:abc123:keyframes -> profile/keyframes/abc123/metadata.pkl
        - profile:video:abc123:keyframes:frame_001 -> profile/keyframes/abc123/frame_001.jpg
        - profile:video:abc123:transcript -> profile/transcripts/abc123.pkl
        - profile:video:abc123:descriptions -> profile/descriptions/abc123.pkl
        """
        parts = key.split(":")

        # Handle different key patterns
        if len(parts) >= 4:
            profile = self._sanitize_path_component(parts[0])
            _ = parts[1]  # Should be "video" - validated elsewhere
            video_id = self._sanitize_path_component(parts[2])
            artifact_type = parts[3]

            # Remove parameters from artifact type
            if "keyframes" in artifact_type:
                artifact_type = "keyframes"

            # Build path based on artifact type
            if artifact_type == "keyframes":
                if len(parts) > 4 and parts[-1].startswith("frame_"):
                    # Individual frame: profile/keyframes/video_id/frame_XXX.jpg
                    frame_name = parts[-1]
                    return (
                        self.base_path
                        / profile
                        / "keyframes"
                        / video_id
                        / f"{frame_name}.jpg"
                    )
                else:
                    # Keyframe metadata: profile/keyframes/video_id/metadata.pkl
                    return (
                        self.base_path
                        / profile
                        / "keyframes"
                        / video_id
                        / f"metadata.{self._get_extension()}"
                    )

            elif artifact_type == "transcript":
                # Transcript: profile/transcripts/video_id.pkl
                return (
                    self.base_path
                    / profile
                    / "transcripts"
                    / f"{video_id}.{self._get_extension()}"
                )

            elif artifact_type == "descriptions":
                # Descriptions: profile/descriptions/video_id.pkl
                return (
                    self.base_path
                    / profile
                    / "descriptions"
                    / f"{video_id}.{self._get_extension()}"
                )

            elif artifact_type == "segment_frames":
                if len(parts) > 4 and parts[-1].startswith("frame_"):
                    # Individual frame: profile/segments/video_id/segment_X/frame_Y.jpg
                    segment_info = "_".join(
                        parts[3:-1]
                    )  # Everything except the last frame_X part
                    safe_segment = self._sanitize_path_component(segment_info)
                    frame_name = parts[-1]
                    return (
                        self.base_path
                        / profile
                        / "segments"
                        / video_id
                        / safe_segment
                        / f"{frame_name}.jpg"
                    )
                else:
                    # Segment metadata: profile/segments/video_id/segment_X/metadata.pkl
                    safe_key = self._sanitize_path_component("_".join(parts[3:]))
                    return (
                        self.base_path
                        / profile
                        / "segments"
                        / video_id
                        / safe_key
                        / f"metadata.{self._get_extension()}"
                    )

            else:
                # Other artifacts: profile/other/video_id/artifact.pkl
                safe_key = self._sanitize_path_component("_".join(parts[3:]))
                return (
                    self.base_path
                    / profile
                    / "other"
                    / video_id
                    / f"{safe_key}.{self._get_extension()}"
                )

        # Fallback for other key patterns
        safe_key = self._sanitize_path_component(key)
        return self.base_path / "misc" / f"{safe_key}.{self._get_extension()}"

    def _get_extension(self) -> str:
        """Get file extension based on serialization format"""
        return {"pickle": "pkl", "json": "json", "msgpack": "msgpack"}.get(
            self.format, "dat"
        )

    def _serialize(self, data: Any) -> bytes:
        """Serialize data based on format, optionally gzip-compressed."""
        if self.format == "pickle":
            raw = pickle.dumps(data)
        elif self.format == "json":
            raw = json.dumps(data).encode("utf-8")
        elif self.format == "msgpack":
            raw = msgpack.packb(data)
        else:
            raise ValueError(f"Unknown serialization format: {self.format}")
        return gzip.compress(raw) if self.enable_compression else raw

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data based on format. Entries written before compression
        was enabled still read back (gzip magic 0x1f 0x8b is detected)."""
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        if self.format == "pickle":
            return pickle.loads(data)
        elif self.format == "json":
            return json.loads(data.decode("utf-8"))
        elif self.format == "msgpack":
            return msgpack.unpackb(data)
        else:
            raise ValueError(f"Unknown serialization format: {self.format}")

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)

        if not file_path.exists():
            self._stats["misses"] += 1
            return None

        # Check expiration (encoded in the file mtime, or a legacy sidecar).
        if self.config.enable_ttl:
            expires_at = await self._read_expiry(file_path)
            if expires_at is not None and time.time() > expires_at:
                await self.delete(key)
                self._stats["misses"] += 1
                return None

        try:
            # Special handling for image files
            if file_path.suffix == ".jpg":
                async with aiofiles.open(file_path, "rb") as f:
                    data = await f.read()
                self._stats["hits"] += 1
                return data

            # Regular serialized data
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()

            self._stats["hits"] += 1
            # gunzip + unpickle is CPU-bound (tens of ms on a large artifact);
            # run it off the event loop so concurrent requests keep flowing.
            return await asyncio.to_thread(self._deserialize, data)

        except Exception as e:
            logger.error(f"Error reading cache file {file_path}: {e}")
            self._stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)

        try:
            # Create parent directory
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write data
            if isinstance(value, bytes) and file_path.suffix == ".jpg":
                # This is image data
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(value)
            else:
                # Regular data - serialize it. pickle + gzip is CPU-bound; run
                # it off the event loop so concurrent requests keep flowing.
                data = await asyncio.to_thread(self._serialize, value)
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(data)

            # Encode expiry in the file mtime instead of a sidecar file — one
            # fewer write per entry. A ttl sets mtime to expires_at; no ttl uses
            # the never-expires sentinel.
            now = time.time()
            expires_at = now + ttl if (ttl is not None and ttl > 0) else _NEVER_EXPIRES
            os.utime(file_path, (now, expires_at))

            self._stats["sets"] += 1
            return True

        except Exception as e:
            logger.error(f"Error writing cache file {file_path}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)
        meta_path = self._get_metadata_path(file_path)

        deleted = False
        if file_path.exists():
            try:
                file_path.unlink()
                deleted = True
                self._stats["deletes"] += 1
            except Exception as e:
                logger.error(f"Error deleting cache file {file_path}: {e}")

        # Delete metadata
        if meta_path.exists():
            try:
                meta_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting metadata file {meta_path}: {e}")

        # Clean up empty directories
        if deleted:
            parent = file_path.parent
            while parent != self.base_path:
                try:
                    if not any(parent.iterdir()):
                        parent.rmdir()
                    parent = parent.parent
                except (OSError, PermissionError):
                    break

        return deleted

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache and is not expired"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)

        if not file_path.exists():
            return False

        # Check expiration if TTL is enabled (mtime, or a legacy sidecar).
        if self.config.enable_ttl:
            expires_at = await self._read_expiry(file_path)
            if expires_at is not None and time.time() > expires_at:
                return False

        return True

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern"""
        await self._run_startup_cleanup_if_needed()
        cleared = 0

        if pattern and pattern.endswith("*"):
            # Pattern-based clearing
            prefix = pattern[:-1]

            # Find all files that match the pattern
            for path in self.base_path.rglob("*"):
                if path.is_file():
                    # Try to reconstruct the key from path
                    # This is approximate but should work for most cases
                    if prefix in str(path):
                        try:
                            path.unlink()
                            cleared += 1
                        except (OSError, PermissionError):
                            pass
        else:
            # Clear everything
            import shutil

            try:
                shutil.rmtree(self.base_path)
                self.base_path.mkdir(parents=True, exist_ok=True)

                # Count cleared files (approximate)
                cleared = self._stats.get("total_files", 0)
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

        return cleared

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        await self._run_startup_cleanup_if_needed()
        self._update_stats()
        return self._stats.copy()

    def _update_stats(self):
        """Update cache statistics"""
        total_size = 0
        total_files = 0
        total_metadata_files = 0

        try:
            for path in self.base_path.rglob("*"):
                if path.is_file():
                    if path.suffix == ".meta":
                        total_metadata_files += 1
                    else:
                        total_size += path.stat().st_size
                        total_files += 1
        except Exception as e:
            logger.error(f"Error calculating cache stats: {e}")

        self._stats["size_bytes"] = total_size
        self._stats["total_files"] = total_files
        self._stats["metadata_files"] = total_metadata_files

    def _get_metadata_path(self, file_path: Path) -> Path:
        """Get metadata file path for a cache file"""
        return file_path.parent / f"{file_path.name}.meta"

    async def _read_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read the legacy ``.meta`` sidecar for a cache file, if one exists.

        New entries no longer write a sidecar (expiry is in the mtime); this
        only returns data for entries written before that change.
        """
        meta_path = self._get_metadata_path(file_path)

        if not meta_path.exists():
            return None

        try:
            async with aiofiles.open(meta_path, "r") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error reading metadata {meta_path}: {e}")
            return None

    async def _read_expiry(self, file_path: Path) -> Optional[float]:
        """Expiry timestamp for a cache entry, or None if the file is gone.

        A legacy ``.meta`` sidecar (older entries) wins so an upgrade doesn't
        invalidate a warm cache: its ``expires_at`` is used, or the never-expires
        sentinel when it recorded no ttl. Otherwise expiry is the file mtime,
        which ``set`` stamps to ``expires_at`` (or the sentinel for no ttl).
        """
        meta = await self._read_metadata(file_path)
        if meta is not None:
            return meta.get("expires_at", _NEVER_EXPIRES)
        try:
            return file_path.stat().st_mtime
        except OSError:
            return None

    async def _run_startup_cleanup_if_needed(self) -> None:
        """Kick off the one-time startup cleanup on the first async operation.

        ``__init__`` is sync and cannot await, so when ``cleanup_on_startup``
        is enabled it sets a flag that this guard consumes once an event loop
        is running. The flag is cleared before scheduling so concurrent first
        calls don't each trigger a sweep. The sweep itself runs as a
        background task — it rglobs the whole cache tree (tens of thousands
        of frame files on a warm video cache), and running it inline made the
        first cache get of the process pay the full walk before answering.
        """
        if self._needs_cleanup:
            self._needs_cleanup = False
            self._startup_cleanup_task = asyncio.create_task(self._cleanup_expired())

    async def _cleanup_expired(self):
        """Clean up expired cache entries"""
        """This runs on startup and can be called manually"""
        logger.info("Starting cleanup of expired cache entries")

        expired_count = 0
        checked_count = 0

        try:
            # Walk the cache's data files (expiry is in their mtime; legacy
            # .meta sidecars are read via _read_expiry). The tree walk is sync
            # (pathlib), so yield periodically to keep the loop responsive.
            for path in self.base_path.rglob("*"):
                if not path.is_file() or path.suffix == ".meta":
                    continue
                checked_count += 1
                if checked_count % 100 == 0:
                    await asyncio.sleep(0)

                try:
                    expires_at = await self._read_expiry(path)
                    if expires_at is not None and time.time() > expires_at:
                        path.unlink()
                        # Drop any legacy sidecar alongside it.
                        meta_path = self._get_metadata_path(path)
                        if meta_path.exists():
                            meta_path.unlink()

                        expired_count += 1
                        self._stats["evictions"] += 1

                except Exception as e:
                    logger.error(f"Error processing cache file {path}: {e}")

            logger.info(
                f"Cleanup complete: checked {checked_count} files, removed {expired_count} expired entries"
            )

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cache key (useful for debugging/inspection)"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)
        return await self._read_metadata(file_path)

    async def list_keys(
        self, pattern: Optional[str] = None, include_metadata: bool = False
    ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """List all keys in cache with optional pattern matching"""
        """Useful for debugging and cache inspection"""
        await self._run_startup_cleanup_if_needed()
        keys = []

        # This is approximate reconstruction of keys from paths
        for meta_path in self.base_path.rglob("*.meta"):
            try:
                metadata = await self._read_metadata(meta_path.parent / meta_path.stem)
                if metadata and "key" in metadata:
                    key = metadata["key"]

                    # Apply pattern filter if provided
                    if pattern:
                        if pattern.endswith("*") and key.startswith(pattern[:-1]):
                            pass  # Match
                        elif pattern in key:
                            pass  # Match
                        else:
                            continue  # No match

                    if include_metadata:
                        keys.append((key, metadata))
                    else:
                        keys.append((key, None))
            except Exception as e:
                logger.debug(f"Error reading metadata for listing: {e}")

        return keys


# Register the backend
from ..registry import CacheBackendRegistry

CacheBackendRegistry.register("structured_filesystem", StructuredFilesystemBackend)
