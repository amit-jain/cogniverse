"""
Structured filesystem cache backend that preserves human-readable paths
"""

import asyncio
import base64
import gzip
import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiofiles
import msgpack

from ..base import CacheBackend

logger = logging.getLogger(__name__)

# Expiry is encoded in each cache file's mtime. A no-ttl entry gets this
# far-future mtime so it never expires.
_NEVER_EXPIRES = 4102444800.0  # 2100-01-01 UTC
_KEY_CHUNK_SIZE = 120

# A .tmp file older than this is an orphan from a crash between the atomic
# write and os.replace — reaped by the cleanup sweep. Comfortably longer than
# any real serialize+write so a live writer's temp is never removed.
_TMP_ORPHAN_MAX_AGE_S = 3600.0


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
    """Filesystem cache with readable namespaces and reversible key leaves."""

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
            "\x00": "_",
        }

        for old, new in replacements.items():
            component = component.replace(old, new)

        # A bare "."/".." component becomes a literal directory segment and
        # walks OUT of base_path; an empty component collapses the path.
        if component in (".", "..", ""):
            component = "_"

        # Limit length to avoid filesystem limits
        if len(component) > 200:
            component = component[:200]

        return component

    @staticmethod
    def _key_namespace_parts(key: str) -> Optional[Tuple[str, str, str]]:
        parts = key.split(":")
        if len(parts) >= 4 and parts[1] == "video":
            return parts[0], parts[2], parts[3]
        if len(parts) >= 3 and parts[0] == "video":
            return "_default", parts[1], parts[2]
        return None

    @classmethod
    def _is_image_key(cls, key: str) -> bool:
        namespace = cls._key_namespace_parts(key)
        return bool(
            namespace
            and namespace[2] in {"keyframes", "segment_frames"}
            and key.split(":")[-1].startswith("frame_")
        )

    def _namespace_path(self, key: str) -> Path:
        """Return the human-readable directory that owns a canonical key."""
        namespace = self._key_namespace_parts(key)
        if namespace:
            profile, video_id, artifact_type = namespace
            profile = self._sanitize_path_component(profile)
            video_id = self._sanitize_path_component(video_id)
            artifact_dir = {
                "keyframes": "keyframes",
                "transcript": "transcripts",
                "descriptions": "descriptions",
                "segment_frames": "segments",
                "segmentation": "segments",
            }.get(artifact_type, "other")
            return self.base_path / profile / artifact_dir / video_id
        return self.base_path / "misc"

    @staticmethod
    def _encode_key(key: str) -> str:
        encoded = base64.urlsafe_b64encode(key.encode("utf-8")).decode("ascii")
        return "k" + encoded.rstrip("=")

    @staticmethod
    def _decode_key(encoded: str) -> str:
        if not encoded.startswith("k"):
            raise ValueError("canonical key encoding is missing its marker")
        payload = encoded[1:]
        padding = "=" * (-len(payload) % 4)
        decoded = base64.b64decode(payload + padding, altchars=b"-_", validate=True)
        return decoded.decode("utf-8")

    def _key_to_path(self, key: str) -> Path:
        """Map a key to its canonical, collision-free ``.keys`` path."""
        encoded = self._encode_key(key)
        chunks = [
            encoded[start : start + _KEY_CHUNK_SIZE]
            for start in range(0, len(encoded), _KEY_CHUNK_SIZE)
        ]
        extension = "jpg" if self._is_image_key(key) else self._get_extension()
        return (
            self._namespace_path(key)
            / ".keys"
            / Path(*chunks[:-1])
            / f"{chunks[-1]}.{extension}"
        )

    def _path_to_key(self, path: Path) -> str:
        """Reverse a canonical ``.keys`` path to the exact original key."""
        relative = path.relative_to(self.base_path)
        markers = [
            index for index, part in enumerate(relative.parts) if part == ".keys"
        ]
        if not markers:
            raise ValueError(f"not a canonical cache path: {path}")
        marker = markers[-1]
        encoded_parts = list(relative.parts[marker + 1 :])
        if not encoded_parts:
            raise ValueError(f"canonical cache path has no encoded key: {path}")
        encoded_parts[-1] = Path(encoded_parts[-1]).stem
        key = self._decode_key("".join(encoded_parts))
        if self._key_to_path(key) != path:
            raise ValueError(f"cache path is not canonical for decoded key: {path}")
        return key

    def _iter_entry_paths(self):
        """Yield only valid files from the canonical layout."""
        for path in self.base_path.rglob("*"):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            try:
                self._path_to_key(path)
            except (UnicodeDecodeError, ValueError):
                continue
            yield path

    @staticmethod
    def _matches_pattern(key: str, pattern: Optional[str]) -> bool:
        if pattern is None or pattern == "*":
            return True
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return key == pattern

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

        # Expiry is encoded in the canonical file's mtime.
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
        tmp_path = file_path.with_name(f"{file_path.name}.{uuid4().hex[:8]}.tmp")

        try:
            # Create parent directory
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to a temp file, stamp the expiry into its mtime, then
            # atomically replace the destination. The destination therefore
            # always carries its true expiry — a concurrent read or cleanup
            # sweep can never observe a fresh entry with a write-time mtime,
            # judge it expired, and destroy it. A partial write is never
            # visible under the final name either.
            if isinstance(value, bytes) and file_path.suffix == ".jpg":
                # This is image data
                async with aiofiles.open(tmp_path, "wb") as f:
                    await f.write(value)
            else:
                # Regular data - serialize it. pickle + gzip is CPU-bound; run
                # it off the event loop so concurrent requests keep flowing.
                data = await asyncio.to_thread(self._serialize, value)
                async with aiofiles.open(tmp_path, "wb") as f:
                    await f.write(data)

            # Expiry lives in the file mtime — one fewer write per entry. A
            # ttl sets mtime to expires_at; no ttl uses the never-expires
            # sentinel.
            now = time.time()
            expires_at = now + ttl if (ttl is not None and ttl > 0) else _NEVER_EXPIRES
            os.utime(tmp_path, (now, expires_at))

            os.replace(tmp_path, file_path)

            self._stats["sets"] += 1
            return True

        except Exception as e:
            logger.error(f"Error writing cache file {file_path}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except (OSError, ValueError):
                # Error cleanup must not turn set()'s bool contract into a raise.
                pass
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)

        deleted = False
        if file_path.exists():
            try:
                file_path.unlink()
                deleted = True
                self._stats["deletes"] += 1
            except Exception as e:
                logger.error(f"Error deleting cache file {file_path}: {e}")

        if deleted:
            self._prune_empty_parents(file_path.parent)

        return deleted

    def _prune_empty_parents(self, parent: Path) -> None:
        while parent != self.base_path:
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
                parent = parent.parent
            except (OSError, PermissionError):
                break

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache and is not expired"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)

        if not file_path.exists():
            return False

        if self.config.enable_ttl:
            expires_at = await self._read_expiry(file_path)
            if expires_at is not None and time.time() > expires_at:
                return False

        return True

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries.

        - ``None`` or ``"*"`` clears the whole cache.
        - ``"<key-prefix>:*"`` clears entries whose path matches the prefix.
        - any other value clears the single entry at that exact key.
        """
        await self._run_startup_cleanup_if_needed()

        if pattern is None or pattern == "*":
            try:
                cleared = sum(1 for _ in self._iter_entry_paths())
                shutil.rmtree(self.base_path)
                self.base_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return 0
            return cleared

        cleared = 0
        for path in list(self._iter_entry_paths()):
            key = self._path_to_key(path)
            if self._matches_pattern(key, pattern):
                try:
                    path.unlink()
                    cleared += 1
                    self._prune_empty_parents(path.parent)
                except (OSError, PermissionError):
                    pass

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

        try:
            for path in self._iter_entry_paths():
                total_size += path.stat().st_size
                total_files += 1
        except Exception as e:
            logger.error(f"Error calculating cache stats: {e}")

        self._stats["size_bytes"] = total_size
        self._stats["total_files"] = total_files

    async def _read_expiry(self, file_path: Path) -> Optional[float]:
        """Return the canonical entry's mtime-encoded expiry."""
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
        """Remove expired canonical entries and stale atomic-write files."""
        logger.info("Starting cleanup of expired cache entries")

        expired_count = 0
        checked_count = 0

        try:
            for path in self.base_path.rglob("*.tmp"):
                try:
                    if time.time() - path.stat().st_mtime > _TMP_ORPHAN_MAX_AGE_S:
                        path.unlink()
                        expired_count += 1
                except OSError:
                    pass

            for path in list(self._iter_entry_paths()):
                checked_count += 1
                if checked_count % 100 == 0:
                    await asyncio.sleep(0)

                try:
                    expires_at = await self._read_expiry(path)
                    if expires_at is not None and time.time() > expires_at:
                        path.unlink()
                        self._prune_empty_parents(path.parent)
                        expired_count += 1
                        self._stats["evictions"] += 1

                except Exception as e:
                    logger.error(f"Error processing cache file {path}: {e}")

            logger.info(
                f"Cleanup complete: checked {checked_count} files, removed {expired_count} expired entries"
            )

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
        return expired_count

    async def cleanup_expired(self) -> int:
        """Run an expiry sweep and return the number of removed files."""
        return await self._cleanup_expired()

    def _metadata_for_path(self, key: str, file_path: Path) -> Dict[str, Any]:
        stat = file_path.stat()
        expires_at = stat.st_mtime
        return {
            "key": key,
            "relative_path": file_path.relative_to(self.base_path).as_posix(),
            "size_bytes": stat.st_size,
            "expires_at": (None if expires_at >= _NEVER_EXPIRES else expires_at),
            "serialization_format": (
                "raw" if file_path.suffix == ".jpg" else self.format
            ),
        }

    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cache key (useful for debugging/inspection)"""
        await self._run_startup_cleanup_if_needed()
        file_path = self._key_to_path(key)
        if not file_path.is_file():
            return None
        expires_at = await self._read_expiry(file_path)
        if (
            self.config.enable_ttl
            and expires_at is not None
            and time.time() > expires_at
        ):
            return None
        try:
            return self._metadata_for_path(key, file_path)
        except OSError:
            return None

    async def list_keys(
        self, pattern: Optional[str] = None, include_metadata: bool = False
    ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """List exact keys decoded from canonical paths."""
        await self._run_startup_cleanup_if_needed()
        keys: List[Tuple[str, Optional[Dict[str, Any]]]] = []

        for path in self._iter_entry_paths():
            try:
                key = self._path_to_key(path)
                if not self._matches_pattern(key, pattern):
                    continue
                expires_at = await self._read_expiry(path)
                if (
                    self.config.enable_ttl
                    and expires_at is not None
                    and time.time() > expires_at
                ):
                    continue
                metadata = (
                    self._metadata_for_path(key, path) if include_metadata else None
                )
                keys.append((key, metadata))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                logger.debug(f"Error decoding canonical cache path {path}: {exc}")

        return sorted(keys, key=lambda item: item[0])


# Register the backend
from ..registry import CacheBackendRegistry

CacheBackendRegistry.register("structured_filesystem", StructuredFilesystemBackend)
