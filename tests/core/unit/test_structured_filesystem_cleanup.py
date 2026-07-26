"""StructuredFilesystemBackend runs cleanup_on_startup on first async use.

`__init__` is sync so it sets `_needs_cleanup` and defers; the flag was never
read, so `cleanup_on_startup` did nothing and expired entries from a previous
run were never purged.
"""

from __future__ import annotations

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
    assert ".keys" in cache_path.parts
    assert list(tmp_path.rglob("*.meta")) == []
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

    from cogniverse_core.common.cache.base import CacheConfig, CacheManager
    from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache

    manager = CacheManager(
        CacheConfig(
            backends=[
                {
                    "backend_type": "structured_filesystem",
                    "base_path": str(tmp_path),
                    "cleanup_on_startup": False,
                    "enable_ttl": True,
                }
            ]
        )
    )
    cache = PipelineArtifactCache(manager, ttl=3600, profile="profile")
    video = "/videos/racevid.mp4"
    assert await cache.set_transcript(video, {"value": "v0"}) is True

    stop = _time.monotonic() + 1.0
    false_miss = 0
    set_failures = 0

    async def writer():
        nonlocal set_failures
        i = 0
        while _time.monotonic() < stop:
            if not await cache.set_transcript(video, {"value": f"v{i}"}):
                set_failures += 1
            i += 1

    async def reader():
        nonlocal false_miss
        while _time.monotonic() < stop:
            if await cache.get_transcript(video) is None:
                false_miss += 1

    await asyncio.gather(writer(), reader(), reader())

    assert false_miss == 0, f"{false_miss} false expiries destroyed a live entry"
    assert set_failures == 0, f"{set_failures} set() calls failed mid-race"
    assert await cache.get_transcript(video) is not None


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_old_layout_is_not_read_listed_or_counted(tmp_path):
    """Only canonical .keys files are cache entries."""
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    key = "profile:video:vid9:transcript:lang=en:model=base"
    old_path = tmp_path / "profile" / "transcripts" / "vid9.pkl"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(backend._serialize({"text": "old"}))
    future = time.time() + 1000
    os.utime(old_path, (time.time(), future))
    old_path.with_name(f"{old_path.name}.meta").write_text(
        '{"key": "profile:video:vid9:transcript:lang=en:model=base"}'
    )

    assert await backend.get(key) is None
    assert await backend.exists(key) is False
    assert await backend.list_keys() == []
    stats = await backend.get_stats()
    assert stats["total_files"] == 0
    assert stats["size_bytes"] == 0


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
    keys = [
        "prof:video:vid123:keyframes:frame_5000",
        "prof:video:vid123:keyframes:frame_15000",
        "prof:video:vid123:keyframes:frame_123456",
        "video:vid123:keyframes:frame_7",
    ]
    for key in keys:
        assert await backend.set(key, raw) is True
        assert await backend.get(key) == raw, f"{key} did not round-trip"
        path = backend._key_to_path(key)
        assert path.suffix == ".jpg"
        assert path.read_bytes() == raw  # raw on disk, not a pickle envelope


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_cleanup_removes_expired_canonical_entry(tmp_path):
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    key = "profile:video:expired9:transcript"
    await backend.set(key, "old", ttl=1000)
    cache_path = backend._key_to_path(key)
    past = time.time() - 50
    os.utime(cache_path, (past, past))

    assert await backend.cleanup_expired() == 1

    assert not cache_path.exists(), "expired entry must be swept"


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
async def test_failed_overwrite_preserves_prior_public_cache_value(
    tmp_path, monkeypatch
):
    from cogniverse_core.common.cache.base import CacheConfig, CacheManager
    from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache

    manager = CacheManager(
        CacheConfig(
            backends=[
                {
                    "backend_type": "structured_filesystem",
                    "base_path": str(tmp_path),
                    "cleanup_on_startup": False,
                    "enable_ttl": True,
                }
            ]
        )
    )
    cache = PipelineArtifactCache(manager, ttl=3600, profile="profile")
    video = "/videos/overwrite.mp4"
    old = {"text": "old-data", "segments": [{"text": "old"}]}
    new = {"text": "new-data", "segments": [{"text": "new"}]}
    assert await cache.set_transcript(video, old) is True

    real_replace = os.replace

    def enospc(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", enospc)
    assert await cache.set_transcript(video, new) is False
    monkeypatch.setattr(os, "replace", real_replace)

    assert await cache.get_transcript(video) == old
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_dotdot_key_component_cannot_escape_base_path(tmp_path):
    """A '..' key component became a literal directory segment — the sanitizer
    replaced slashes and metachars but not dot-dot, so a crafted key wrote
    OUTSIDE base_path. The backend is registered generically; keys are not
    guaranteed hex-digest-shaped forever."""
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    key = "..:video:vid9:transcript"

    resolved = backend._key_to_path(key).resolve()
    assert resolved.is_relative_to(tmp_path.resolve()), (
        f"key escaped base_path: {resolved}"
    )

    assert await backend.set(key, "payload", ttl=60) is True
    assert await backend.get(key) == "payload"
    escaped = tmp_path.parent / "transcripts"
    assert not escaped.exists(), f"cache wrote outside base_path: {escaped}"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_nul_byte_key_round_trips_through_encoded_path(tmp_path):
    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path), cleanup_on_startup=False, enable_ttl=True
        )
    )
    key = "pro\x00file:video:vid:transcript"

    assert await backend.set(key, "x", ttl=60) is True
    path = backend._key_to_path(key)
    assert "\x00" not in str(path)
    assert backend._path_to_key(path) == key
    assert await backend.get(key) == "x"
    assert await backend.exists(key) is True
