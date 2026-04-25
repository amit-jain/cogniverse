"""Regression tests for the URI-hash cache-key change in PipelineArtifactCache.

Phase 2 of the unified MediaLocator rollout switched
``PipelineArtifactCache._generate_video_key`` from ``Path.stem`` (which collides
across roots) to ``sha256(canonical_uri)[:16]``. These tests would have failed
on the pre-change code.
"""

from unittest.mock import Mock

import pytest

from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache


@pytest.fixture
def cache():
    return PipelineArtifactCache(cache_manager=Mock(), profile=None)


@pytest.fixture
def profiled_cache():
    return PipelineArtifactCache(cache_manager=Mock(), profile="frame_based_colpali")


class TestSameStemDifferentRoots:
    def test_distinct_keys_for_same_basename_in_different_dirs(self, cache, tmp_path):
        a = tmp_path / "root_a"
        b = tmp_path / "root_b"
        a.mkdir()
        b.mkdir()
        (a / "v_clip.mp4").write_bytes(b"")
        (b / "v_clip.mp4").write_bytes(b"")

        key_a = cache._generate_video_key(str(a / "v_clip.mp4"))
        key_b = cache._generate_video_key(str(b / "v_clip.mp4"))

        assert key_a != key_b


class TestStability:
    def test_same_path_same_key(self, cache):
        key_1 = cache._generate_video_key("/abs/path/v.mp4")
        key_2 = cache._generate_video_key("/abs/path/v.mp4")
        assert key_1 == key_2

    def test_uri_input_passes_through_without_double_canonicalization(self, cache):
        uri_key = cache._generate_video_key("s3://corpus/v.mp4")

        assert uri_key.startswith("video:")
        assert len(uri_key.split(":", 1)[1]) == 16

    def test_uri_and_equivalent_file_uri_produce_same_key(self, cache):
        path_key = cache._generate_video_key("/abs/v.mp4")
        uri_key = cache._generate_video_key("file:///abs/v.mp4")
        assert path_key == uri_key


class TestProfileNamespace:
    def test_profile_prefix_preserved(self, profiled_cache):
        key = profiled_cache._generate_video_key("/abs/v.mp4")
        assert key.startswith("frame_based_colpali:video:")

    def test_no_profile_no_prefix(self, cache):
        key = cache._generate_video_key("/abs/v.mp4")
        assert key.startswith("video:")
        assert not key.startswith("frame_based_colpali:")


class TestKeyShape:
    def test_key_format_is_video_colon_sha256_prefix(self, cache):
        key = cache._generate_video_key("/abs/v.mp4")
        prefix, digest = key.split(":", 1)
        assert prefix == "video"
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)
