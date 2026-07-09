"""enable_compression must actually gzip cached payloads.

The flag was forwarded into CacheConfig but no backend read it, so cached bytes
were never compressed. These pin: compression round-trips, the on-disk bytes
are gzip when enabled, mixed compressed/uncompressed entries still read back,
and the manager-level flag reaches the backends.
"""

from __future__ import annotations

import gzip

import pytest

from cogniverse_core.common.cache.backends.structured_filesystem import (
    StructuredFilesystemBackend,
    StructuredFilesystemConfig,
)


def _backend(base_path, enable_compression):
    cfg = StructuredFilesystemConfig(
        base_path=str(base_path),
        enable_compression=enable_compression,
        cleanup_on_startup=False,
    )
    return StructuredFilesystemBackend(cfg)


@pytest.mark.asyncio
async def test_round_trip_with_compression(tmp_path):
    backend = _backend(tmp_path, True)
    value = {"frames": list(range(1000)), "label": "x" * 500}
    await backend.set("k1", value)
    assert await backend.get("k1") == value


@pytest.mark.asyncio
async def test_bytes_are_gzip_when_enabled(tmp_path):
    backend = _backend(tmp_path, True)
    payload = backend._serialize({"a": "b" * 1000})
    assert payload[:2] == b"\x1f\x8b"  # gzip magic
    # and it round-trips through deserialize
    assert backend._deserialize(payload) == {"a": "b" * 1000}


@pytest.mark.asyncio
async def test_bytes_not_gzip_when_disabled(tmp_path):
    backend = _backend(tmp_path, False)
    payload = backend._serialize({"a": "b"})
    assert payload[:2] != b"\x1f\x8b"


@pytest.mark.asyncio
async def test_uncompressed_entry_still_readable_after_enabling(tmp_path):
    # Write with compression off, read with a backend that has it on: the
    # magic-byte check means the old plain entry still deserializes.
    off = _backend(tmp_path, False)
    await off.set("legacy", {"v": 1})

    on = _backend(tmp_path, True)
    assert await on.get("legacy") == {"v": 1}


@pytest.mark.asyncio
async def test_compression_reduces_size(tmp_path):
    on = _backend(tmp_path, True)
    off = _backend(tmp_path, False)
    value = {"repetitive": "cogniverse " * 2000}
    assert len(on._serialize(value)) < len(off._serialize(value))


def test_manager_propagates_compression_flag_to_backends(tmp_path):
    from cogniverse_core.common.cache.base import CacheConfig, CacheManager

    config = CacheConfig(
        backends=[
            {
                "backend_type": "structured_filesystem",
                "base_path": str(tmp_path / "c"),
                "cleanup_on_startup": False,
            }
        ],
        enable_compression=False,
    )
    manager = CacheManager(config)
    assert manager.backends, "backend must initialize"
    # The manager-level flag reached the backend even though the backend dict
    # did not set it.
    assert manager.backends[0].enable_compression is False


def test_backend_override_wins_over_manager_flag(tmp_path):
    from cogniverse_core.common.cache.base import CacheConfig, CacheManager

    config = CacheConfig(
        backends=[
            {
                "backend_type": "structured_filesystem",
                "base_path": str(tmp_path / "c"),
                "cleanup_on_startup": False,
                "enable_compression": True,  # explicit backend override
            }
        ],
        enable_compression=False,  # manager default
    )
    manager = CacheManager(config)
    assert manager.backends[0].enable_compression is True


def test_gzip_magic_matches_stdlib(tmp_path):
    # Guard the magic-byte constant against drift.
    assert gzip.compress(b"x")[:2] == b"\x1f\x8b"
