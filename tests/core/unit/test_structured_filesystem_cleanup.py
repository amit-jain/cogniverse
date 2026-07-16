"""StructuredFilesystemBackend runs cleanup_on_startup on first async use.

`__init__` is sync so it sets `_needs_cleanup` and defers; the flag was never
read, so `cleanup_on_startup` did nothing and expired entries from a previous
run were never purged.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from cogniverse_core.common.cache.backends.structured_filesystem import (
    StructuredFilesystemBackend,
    StructuredFilesystemConfig,
)


async def _seed_expired_entry(base_path: str, key: str):
    """Write a cache entry then back-date its mtime so it is expired.

    Expiry is encoded in the file mtime (no .meta sidecar), so aging an entry
    means back-dating that mtime.
    """
    seeder = StructuredFilesystemConfig(
        base_path=base_path, cleanup_on_startup=False, enable_ttl=True
    )
    backend = StructuredFilesystemBackend(seeder)
    await backend.set(key, "stale-data", ttl=1000)

    cache_path = backend._key_to_path(key)
    past = time.time() - 100
    os.utime(cache_path, (past, past))
    return cache_path


@pytest.mark.unit
@pytest.mark.ci_fast
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

    # First async op schedules the deferred startup sweep as a background
    # task — the op itself must not pay the full cache-tree walk.
    await backend.get("unrelated_key")
    assert backend._needs_cleanup is False

    # The sweep still purges the expired entry once it completes.
    await backend._startup_cleanup_task
    assert cache_path.exists() is False


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


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_expiry_encoded_in_mtime_without_sidecar(tmp_path):
    """set() records expiry in the file mtime and writes NO .meta sidecar —
    one fewer filesystem write per cached entry."""
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    await backend.set("k", "data", ttl=1000)
    cache_path = backend._key_to_path("k")

    # No sidecar written; expiry lives in the mtime (~now + ttl).
    assert not backend._get_metadata_path(cache_path).exists()
    assert cache_path.stat().st_mtime == pytest.approx(time.time() + 1000, abs=5)
    assert await backend.get("k") == "data"

    # Back-dating the mtime expires the entry — get purges it.
    past = time.time() - 10
    os.utime(cache_path, (past, past))
    assert await backend.get("k") is None
    assert cache_path.exists() is False


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_no_ttl_entry_never_expires(tmp_path):
    """A set() with no ttl gets the never-expires mtime sentinel."""
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    await backend.set("k", "data")  # no ttl
    cache_path = backend._key_to_path("k")
    assert cache_path.stat().st_mtime > time.time() + 10**9  # far future
    assert await backend.get("k") == "data"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_reads_never_destroy_a_live_entry(tmp_path):
    """A reader (or the cleanup sweep) racing a writer must NEVER see a fresh
    entry as expired. The original set() wrote the data file (mtime =
    write-time) and only then stamped the expiry via os.utime — in that window
    a concurrent get() read mtime as the expiry, judged the entry expired, and
    DELETED it (including never-expiring ttl=None entries); the writer's utime
    then failed. The write must be atomic: temp file, stamp, os.replace."""
    import asyncio
    import time as _time

    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    key = "profile:video:racevid:transcript"
    assert await backend.set(key, "v0") is True  # ttl=None → never expires

    stop = _time.monotonic() + 1.0
    false_miss = 0
    set_failures = 0

    async def writer():
        nonlocal set_failures
        i = 0
        while _time.monotonic() < stop:
            if not await backend.set(key, f"v{i}"):
                set_failures += 1
            i += 1

    async def reader():
        nonlocal false_miss
        while _time.monotonic() < stop:
            if await backend.get(key) is None:
                false_miss += 1

    await asyncio.gather(writer(), reader(), reader())

    assert false_miss == 0, f"{false_miss} false expiries destroyed a live entry"
    assert set_failures == 0, f"{set_failures} set() calls failed mid-race"
    assert await backend.get(key) is not None


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_rewrite_clears_legacy_sidecar(tmp_path):
    """Re-writing a key that still carries a legacy .meta sidecar must clear
    the sidecar — otherwise the STALE sidecar expiry governs the fresh write
    (a past expires_at deleted freshly-written data on the next read)."""
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    key = "profile:video:vid9:transcript"
    await backend.set(key, "old-data", ttl=1000)
    cache_path = backend._key_to_path(key)
    meta_path = backend._get_metadata_path(cache_path)
    # Upgrade-transition state: a legacy sidecar with a stale (past) expiry.
    meta_path.write_text(json.dumps({"expires_at": time.time() - 10}))

    assert await backend.set(key, "fresh-data", ttl=3600) is True

    assert not meta_path.exists(), "stale legacy sidecar must be cleared on rewrite"
    assert await backend.get(key) == "fresh-data"
    assert cache_path.exists()


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_legacy_meta_sidecar_still_honored(tmp_path):
    """An entry written before mtime-encoding (a .meta sidecar) is read via the
    sidecar, so an upgrade does not invalidate a warm cache: a future sidecar
    expiry wins over a past mtime."""
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    await backend.set("k", "data", ttl=1000)
    cache_path = backend._key_to_path("k")

    # Simulate a legacy entry: a sidecar with a FUTURE expiry over a PAST mtime.
    backend._get_metadata_path(cache_path).write_text(
        json.dumps({"expires_at": time.time() + 1000})
    )
    past = time.time() - 500
    os.utime(cache_path, (past, past))

    # The sidecar (future) wins over the mtime (past) → still readable.
    assert await backend.get("k") == "data"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_keyframe_image_round_trips_raw_for_any_frame_id(tmp_path):
    """A keyframe image must survive the cache as raw bytes regardless of frame_id.

    Regression: set() wrote raw bytes only for frame_0..9999 (range(10000)) and
    pickled higher indices, while get() returns raw for any .jpg path — so frames
    past ~5 min of 30fps video came back as pickled bytes and failed to decode.
    """
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(base_path=str(tmp_path), cleanup_on_startup=False)
    )
    raw = b"\xff\xd8\xff\xe0opaque-image-bytes\xff\xd9"
    for frame_id in (5000, 15000, 123456):
        key = f"prof:video:vid123:keyframes:frame_{frame_id}"
        assert await backend.set(key, raw) is True
        assert await backend.get(key) == raw, f"frame_{frame_id} did not round-trip"
        path = backend._key_to_path(key)
        assert path.suffix == ".jpg"
        assert path.read_bytes() == raw  # raw on disk, not a pickle envelope


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_cleanup_drops_legacy_sidecar_alongside_expired_entry(tmp_path):
    """The startup sweep must remove a legacy .meta sidecar together with its
    expired data file — leaving it would orphan sidecars forever (and a later
    same-key write already clears them, but the sweep path never ran in a
    test)."""
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    key = "profile:video:legacy9:transcript"
    await backend.set(key, "old", ttl=1000)
    cache_path = backend._key_to_path(key)
    meta_path = backend._get_metadata_path(cache_path)
    # Legacy entry shape: sidecar with a PAST expiry governs the file.
    meta_path.write_text(json.dumps({"expires_at": time.time() - 50}))

    await backend._cleanup_expired()

    assert not cache_path.exists(), "expired entry must be swept"
    assert not meta_path.exists(), "its legacy sidecar must be dropped too"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_cleanup_reaps_old_tmp_orphans_but_spares_fresh_ones(tmp_path):
    """A .tmp left by a crash between write and os.replace is reaped once it's
    older than the safety window; a fresh in-flight .tmp is spared."""
    import cogniverse_core.common.cache.backends.structured_filesystem as sfs

    backend = sfs.StructuredFilesystemBackend(
        sfs.StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    d = tmp_path / "misc"
    d.mkdir(parents=True, exist_ok=True)
    old = d / "k.abc123.tmp"
    fresh = d / "k.def456.tmp"
    old.write_bytes(b"orphan")
    fresh.write_bytes(b"in-flight")
    past = time.time() - (sfs._TMP_ORPHAN_MAX_AGE_S + 60)
    os.utime(old, (past, past))

    await backend._cleanup_expired()

    assert not old.exists(), "stale .tmp orphan must be reaped"
    assert fresh.exists(), "a fresh in-flight .tmp must be spared"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_set_failure_returns_false_and_leaves_no_tmp(tmp_path, monkeypatch):
    """A serialize failure mid-set must return False and leave no orphaned
    .tmp behind (the except-branch cleanup)."""
    from cogniverse_core.common.cache.backends.structured_filesystem import (
        StructuredFilesystemBackend,
        StructuredFilesystemConfig,
    )

    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(base_path=str(tmp_path), cleanup_on_startup=False)
    )
    monkeypatch.setattr(
        backend, "_serialize", lambda v: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    ok = await backend.set("prof:video:v1:transcript", {"a": 1})

    assert ok is False
    assert list(tmp_path.rglob("*.tmp")) == [], "failed set left an orphaned .tmp"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_failed_overwrite_leaves_legacy_entry_intact(tmp_path, monkeypatch):
    """A failed os.replace (disk full) mid-set must not destroy the OLD entry.

    The legacy .meta sidecar was unlinked BEFORE the replace, so a failed
    replace left the old data file without its (future-expiry) sidecar — its
    past mtime then read as expired and the next get() deleted a live entry.
    """
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    await backend.set("k", "old-data", ttl=1000)
    cache_path = backend._key_to_path("k")
    # Legacy entry shape: a FUTURE-expiry sidecar governing a PAST mtime.
    backend._get_metadata_path(cache_path).write_text(
        json.dumps({"expires_at": time.time() + 1000})
    )
    past = time.time() - 500
    os.utime(cache_path, (past, past))

    real_replace = os.replace

    def enospc(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", enospc)
    assert await backend.set("k", "new-data", ttl=3600) is False
    monkeypatch.setattr(os, "replace", real_replace)

    assert await backend.get("k") == "old-data", (
        "failed overwrite destroyed the legacy entry"
    )
