"""StructuredFilesystemBackend runs cleanup_on_startup on first async use.

`__init__` is sync so it sets `_needs_cleanup` and defers; the flag was never
read, so `cleanup_on_startup` did nothing and expired entries from a previous
run were never purged.
"""

from __future__ import annotations

import json
import time

import pytest

from cogniverse_core.common.cache.backends.structured_filesystem import (
    StructuredFilesystemBackend,
    StructuredFilesystemConfig,
)


async def _seed_expired_entry(base_path: str, key: str):
    """Write a cache entry then back-date its metadata so it is expired."""
    seeder = StructuredFilesystemConfig(
        base_path=base_path, cleanup_on_startup=False, enable_ttl=True
    )
    backend = StructuredFilesystemBackend(seeder)
    await backend.set(key, "stale-data", ttl=1000)

    cache_path = backend._key_to_path(key)
    meta_path = backend._get_metadata_path(cache_path)
    meta = json.loads(meta_path.read_text())
    meta["expires_at"] = time.time() - 100
    meta_path.write_text(json.dumps(meta))
    return cache_path


@pytest.mark.asyncio
async def test_startup_cleanup_purges_expired_entry_on_first_op(tmp_path):
    base = str(tmp_path)
    cache_path = await _seed_expired_entry(base, "old_key")
    assert cache_path.exists()

    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=base, cleanup_on_startup=True, enable_ttl=True
        )
    )
    # Sync __init__ cannot await — the expired file is still present.
    assert cache_path.exists()
    assert backend._needs_cleanup is True

    # First async op triggers the deferred startup sweep.
    await backend.get("unrelated_key")

    assert cache_path.exists() is False
    assert backend._needs_cleanup is False


@pytest.mark.asyncio
async def test_no_startup_cleanup_leaves_expired_file_until_accessed(tmp_path):
    base = str(tmp_path)
    cache_path = await _seed_expired_entry(base, "old_key")

    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=base, cleanup_on_startup=False, enable_ttl=True
        )
    )
    await backend.get("unrelated_key")

    # No startup sweep — the expired file remains until its key is accessed.
    assert cache_path.exists() is True
    assert backend._needs_cleanup is False
