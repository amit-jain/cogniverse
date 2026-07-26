"""Pipeline-cache reserved surfaces: segment frames, invalidation, stats.

Kept as supported API (segment-frame caching for chunked profiles, ops
invalidation, stats) — none had a single caller or test, so their contracts
are pinned here against the real filesystem-backed CacheManager.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_core.common.cache.base import CacheConfig, CacheManager
from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture()
def cache(tmp_path):
    manager = CacheManager(
        CacheConfig(
            backends=[
                {
                    "backend_type": "structured_filesystem",
                    "base_path": str(tmp_path / "cache"),
                    "cleanup_on_startup": False,
                }
            ]
        )
    )
    return PipelineArtifactCache(manager, ttl=3600, profile="test_profile")


@pytest.mark.asyncio
async def test_segment_frames_round_trip(cache):
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)]
    ok = await cache.set_segment_frames(
        video_path="/videos/clip.mp4",
        segment_id=0,
        start_time=0.0,
        end_time=6.0,
        frames=frames,
        timestamps=[0.0, 2.0, 4.0],
    )
    assert ok is True

    result = await cache.get_segment_frames(
        video_path="/videos/clip.mp4",
        segment_id=0,
        start_time=0.0,
        end_time=6.0,
        load_images=True,
    )
    assert result is not None
    metadata, images = result if isinstance(result, tuple) else (result, [])
    assert metadata["segment_id"] == 0
    assert metadata["timestamps"] == [0.0, 2.0, 4.0]
    assert len(images) == 3
    assert (images[1] == frames[1]).all(), "frame bytes must survive the cache"


@pytest.mark.asyncio
async def test_segment_frames_miss_on_different_segment(cache):
    await cache.set_segment_frames(
        video_path="/videos/clip.mp4",
        segment_id=0,
        start_time=0.0,
        end_time=6.0,
        frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
        timestamps=[0.0],
    )
    assert (
        await cache.get_segment_frames(
            video_path="/videos/clip.mp4",
            segment_id=7,
            start_time=42.0,
            end_time=48.0,
        )
        is None
    )


@pytest.mark.asyncio
async def test_invalidate_video_clears_its_entries(cache):
    await cache.set_transcript("/videos/clip.mp4", {"text": "hello", "segments": []})
    assert await cache.get_transcript("/videos/clip.mp4") is not None

    cleared = await cache.invalidate_video("/videos/clip.mp4")

    assert cleared >= 1
    assert await cache.get_transcript("/videos/clip.mp4") is None


@pytest.mark.asyncio
async def test_get_cache_stats_shape(cache):
    await cache.set_transcript("/videos/clip.mp4", {"text": "x", "segments": []})

    stats = await cache.get_cache_stats()

    assert set(stats.keys()) == {"overall", "artifacts"}
    assert isinstance(stats["overall"], dict) and stats["overall"]
    # Per-artifact stats are an acknowledged placeholder.
    assert stats["artifacts"] == {
        "keyframes": "Not implemented",
        "transcripts": "Not implemented",
        "descriptions": "Not implemented",
    }


@pytest.mark.asyncio
async def test_invalidate_video_does_not_wipe_prefix_sibling_profiles(tmp_path):
    """invalidate_video on profile 'direct_video_global' must NOT also delete
    the same video's cache under 'direct_video_global_large' — the token match
    is by path SEGMENT, not substring."""
    from cogniverse_core.common.cache.base import CacheConfig, CacheManager
    from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache

    def _cache(profile):
        mgr = CacheManager(
            CacheConfig(
                backends=[
                    {
                        "backend_type": "structured_filesystem",
                        "base_path": str(tmp_path / "cache"),
                        "cleanup_on_startup": False,
                    }
                ]
            )
        )
        return PipelineArtifactCache(mgr, ttl=3600, profile=profile)

    base = _cache("direct_video_global")
    large = _cache("direct_video_global_large")
    await base.set_transcript(
        "/videos/clip.mp4",
        {"text": "base-en", "segments": []},
        model_size="base",
        language="en",
    )
    await base.set_transcript(
        "/videos/clip.mp4",
        {"text": "large-fr", "segments": []},
        model_size="large",
        language="fr",
    )
    await base.set_transcript(
        "/videos/other.mp4",
        {"text": "other", "segments": []},
        model_size="base",
        language="en",
    )
    await large.set_transcript("/videos/clip.mp4", {"text": "large", "segments": []})

    cleared = await base.invalidate_video("/videos/clip.mp4")

    assert cleared == 2
    assert (
        await base.get_transcript("/videos/clip.mp4", model_size="base", language="en")
        is None
    )
    assert (
        await base.get_transcript("/videos/clip.mp4", model_size="large", language="fr")
        is None
    )
    assert await base.get_transcript(
        "/videos/other.mp4", model_size="base", language="en"
    ) == {"text": "other", "segments": []}
    large_survivor = await large.get_transcript("/videos/clip.mp4")
    assert large_survivor is not None, "sibling _large profile was wrongly wiped"
    assert large_survivor["text"] == "large"


@pytest.mark.asyncio
async def test_parameter_variants_use_distinct_reversible_canonical_paths(cache):
    video = "/videos/clip.mp4"
    variants = {
        ("base", "en"): {"text": "base-en", "segments": []},
        ("base", "fr"): {"text": "base-fr", "segments": []},
        ("large-v3", "en"): {"text": "large-en", "segments": []},
        ("vision/" + ("x" * 320), "日本語"): {
            "text": "long-unicode",
            "segments": [],
        },
    }

    for (model, language), payload in variants.items():
        assert (
            await cache.set_transcript(
                video, payload, model_size=model, language=language
            )
            is True
        )

    for (model, language), payload in variants.items():
        assert (
            await cache.get_transcript(video, model_size=model, language=language)
            == payload
        )

    backend = cache.cache.backends[0]
    video_key = cache._generate_video_key(video)
    keys = [
        cache._generate_artifact_key(
            video_key, "transcript", lang=language, model=model
        )
        for model, language in variants
    ]
    paths = [backend._key_to_path(key) for key in keys]

    assert len(set(paths)) == len(variants)
    assert all(".keys" in path.parts for path in paths)
    assert all(max(map(len, path.parts)) <= 200 for path in paths)
    assert [backend._path_to_key(path) for path in paths] == keys


@pytest.mark.asyncio
async def test_list_keys_and_stats_are_exact_for_canonical_entries(cache):
    video = "/videos/exact.mp4"
    await cache.set_transcript(
        video,
        {"text": "english", "segments": []},
        model_size="base",
        language="en",
    )
    await cache.set_transcript(
        video,
        {"text": "french", "segments": []},
        model_size="large",
        language="fr",
    )
    await cache.set_descriptions(
        video,
        {"descriptions": [{"frame_id": 0, "text": "goal"}]},
        model_name="vision/model:v2",
        batch_size=7,
    )

    backend = cache.cache.backends[0]
    video_key = cache._generate_video_key(video)
    expected_keys = sorted(
        [
            cache._generate_artifact_key(
                video_key, "transcript", lang="en", model="base"
            ),
            cache._generate_artifact_key(
                video_key, "transcript", lang="fr", model="large"
            ),
            cache._generate_artifact_key(
                video_key,
                "descriptions",
                batch_size=7,
                model="vision/model:v2",
            ),
        ]
    )
    expected_paths = [backend._key_to_path(key) for key in expected_keys]

    assert await backend.list_keys() == [(key, None) for key in expected_keys]
    assert await backend.list_keys(f"{video_key}:transcript:*") == [
        (key, None) for key in expected_keys if ":transcript:" in key
    ]

    stats = await backend.get_stats()
    assert stats["total_files"] == 3
    assert stats["size_bytes"] == sum(path.stat().st_size for path in expected_paths)


@pytest.mark.asyncio
async def test_clear_star_wipes_all_and_nonwildcard_clears_one_key(tmp_path):
    """clear('*') == clear-all; a non-wildcard pattern clears only that key
    (never a full wipe)."""
    from cogniverse_core.common.cache.backends.structured_filesystem import (
        StructuredFilesystemBackend,
        StructuredFilesystemConfig,
    )

    backend = StructuredFilesystemBackend(
        StructuredFilesystemConfig(
            base_path=str(tmp_path / "c"), cleanup_on_startup=False
        )
    )
    await backend.set("prof:video:v1:transcript", {"a": 1})
    await backend.set("prof:video:v2:transcript", {"b": 2})

    # Non-wildcard clears exactly one entry, leaving the other.
    n = await backend.clear("prof:video:v1:transcript")
    assert n == 1
    assert await backend.get("prof:video:v1:transcript") is None
    assert await backend.get("prof:video:v2:transcript") == {"b": 2}

    # '*' clears everything that remains.
    await backend.clear("*")
    assert await backend.get("prof:video:v2:transcript") is None


@pytest.mark.asyncio
async def test_segmentation_round_trip(cache):
    payload = {
        "segments": [
            {
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 1.0,
                "frame_timestamps": [0.0, 0.5],
                "transcript_segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
                "transcript_text": "hi",
                "metadata": {"duration": 1.0, "frame_count": 2},
            }
        ],
        "metadata": {"video_id": "clip", "num_segments": 1},
        "full_transcript": "hi",
        "document_structure": {"type": "single_doc"},
    }
    ok = await cache.set_segmentation(
        "/videos/clip.mp4",
        payload,
        strategy="chunks",
        segment_duration=6.0,
        segment_overlap=1.0,
        sampling_fps=2.0,
        max_frames=12,
        transcript_fingerprint="abc123",
    )
    assert ok is True

    got = await cache.get_segmentation(
        "/videos/clip.mp4",
        strategy="chunks",
        segment_duration=6.0,
        segment_overlap=1.0,
        sampling_fps=2.0,
        max_frames=12,
        transcript_fingerprint="abc123",
    )
    assert got == payload


@pytest.mark.asyncio
async def test_segmentation_misses_on_any_param_change(cache):
    payload = {"segments": [], "metadata": {}, "full_transcript": ""}
    await cache.set_segmentation(
        "/videos/clip.mp4",
        payload,
        strategy="chunks",
        segment_duration=6.0,
        segment_overlap=1.0,
        sampling_fps=2.0,
        max_frames=12,
        transcript_fingerprint="abc123",
    )

    for change in (
        {"strategy": "windows"},
        {"segment_duration": 30.0},
        {"segment_overlap": 0.0},
        {"sampling_fps": 1.0},
        {"max_frames": 6},
        {"transcript_fingerprint": "other"},
    ):
        params = {
            "strategy": "chunks",
            "segment_duration": 6.0,
            "segment_overlap": 1.0,
            "sampling_fps": 2.0,
            "max_frames": 12,
            "transcript_fingerprint": "abc123",
            **change,
        }
        assert await cache.get_segmentation("/videos/clip.mp4", **params) is None
