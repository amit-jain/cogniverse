"""Unit tests for the content-addressed MediaCache."""

import os
import time

import pytest

from cogniverse_core.common.media import MediaCache


@pytest.fixture
def cache(tmp_path):
    return MediaCache(tmp_path / "cache", max_bytes=1_000_000)


def _write(path, content: bytes = b"x"):
    path.write_bytes(content)
    return path


class TestMakeKey:
    def test_deterministic(self):
        assert MediaCache.make_key("s3://b/k.mp4") == MediaCache.make_key(
            "s3://b/k.mp4"
        )

    def test_etag_changes_key(self):
        assert MediaCache.make_key("s3://b/k.mp4", etag="abc") != MediaCache.make_key(
            "s3://b/k.mp4"
        )

    def test_different_uris_different_keys(self):
        assert MediaCache.make_key("s3://b/a.mp4") != MediaCache.make_key(
            "s3://b/b.mp4"
        )


class TestPutGet:
    def test_round_trip(self, cache, tmp_path):
        src = _write(tmp_path / "src", b"hello")
        key = MediaCache.make_key("s3://b/v.mp4")

        dest = cache.put(key, "v.mp4", src)

        assert dest.exists()
        assert dest.read_bytes() == b"hello"
        assert cache.get(key, "v.mp4") == dest

    def test_get_missing_returns_none(self, cache):
        assert cache.get("nonexistent", "v.mp4") is None

    def test_basename_preserved_in_path(self, cache, tmp_path):
        src = _write(tmp_path / "src", b"data")
        key = MediaCache.make_key("https://example.com/clip.mkv")

        dest = cache.put(key, "clip.mkv", src)

        assert dest.name == "clip.mkv"

    def test_put_moves_source_into_cache(self, cache, tmp_path):
        src = _write(tmp_path / "src", b"v1")
        key = MediaCache.make_key("s3://b/v.mp4")

        dest = cache.put(key, "v.mp4", src)

        assert dest.exists()
        assert not src.exists()
        assert dest.read_bytes() == b"v1"


class TestEviction:
    def test_no_eviction_under_budget(self, tmp_path):
        cache = MediaCache(tmp_path / "cache", max_bytes=1000)

        for i in range(3):
            src = tmp_path / f"src_{i}"
            src.write_bytes(b"x" * 100)
            cache.put(MediaCache.make_key(f"s3://b/{i}"), f"{i}.mp4", src)

        assert cache.total_bytes() == 300
        for i in range(3):
            assert cache.get(MediaCache.make_key(f"s3://b/{i}"), f"{i}.mp4") is not None

    def test_lru_eviction_when_over_budget(self, tmp_path):
        cache = MediaCache(tmp_path / "cache", max_bytes=250)

        for i in range(3):
            src = tmp_path / f"src_{i}"
            src.write_bytes(b"x" * 100)
            cache.put(MediaCache.make_key(f"s3://b/{i}"), f"{i}.mp4", src)
            time.sleep(0.01)

        keys_present = [
            cache.get(MediaCache.make_key(f"s3://b/{i}"), f"{i}.mp4") is not None
            for i in range(3)
        ]
        assert keys_present.count(True) == 2
        assert cache.get(MediaCache.make_key("s3://b/0"), "0.mp4") is None

    def test_get_bumps_atime_keeping_entry_alive(self, tmp_path):
        cache = MediaCache(tmp_path / "cache", max_bytes=250)

        for i in range(2):
            src = tmp_path / f"src_{i}"
            src.write_bytes(b"x" * 100)
            cache.put(MediaCache.make_key(f"s3://b/{i}"), f"{i}.mp4", src)
            time.sleep(0.05)

        cache.get(MediaCache.make_key("s3://b/0"), "0.mp4")
        time.sleep(0.05)

        src2 = tmp_path / "src_2"
        src2.write_bytes(b"x" * 100)
        cache.put(MediaCache.make_key("s3://b/2"), "2.mp4", src2)

        assert cache.get(MediaCache.make_key("s3://b/0"), "0.mp4") is not None
        assert cache.get(MediaCache.make_key("s3://b/1"), "1.mp4") is None


class TestStaging:
    def test_staging_dir_created(self, cache):
        assert cache.staging_dir.exists()

    def test_staging_path_unique(self, cache):
        a = cache.staging_path()
        b = cache.staging_path()
        assert a != b
        assert a.parent == cache.staging_dir

    def test_total_bytes_excludes_staging(self, cache):
        staging = cache.staging_path()
        staging.write_bytes(b"x" * 999)

        assert cache.total_bytes() == 0


class TestAtimeBump:
    def test_get_updates_atime(self, cache, tmp_path):
        src = _write(tmp_path / "src", b"hi")
        key = MediaCache.make_key("s3://b/v.mp4")
        cache.put(key, "v.mp4", src)

        dest = cache._key_to_path(key, "v.mp4")
        old_atime = os.path.getatime(dest)
        os.utime(dest, (old_atime - 100, os.path.getmtime(dest)))

        cache.get(key, "v.mp4")
        new_atime = os.path.getatime(dest)
        assert new_atime > old_atime - 100
