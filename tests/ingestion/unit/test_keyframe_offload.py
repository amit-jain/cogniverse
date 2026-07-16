"""Keyframe image caching + MinIO upload run off the event loop.

set_cached_keyframes decodes every extracted frame from disk (cv2.imread),
get_cached_keyframes re-encodes cached frames back to disk (cv2.imwrite), and
the segmentation strategy uploads the frames to MinIO over HTTP. All three ran
synchronously on the async ingestion path, freezing the loop for the whole
batch. These pin the offload with a concurrent ticker.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit]


async def _ticks_during(coro_factory) -> int:
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    await coro_factory()
    stop.set()
    await t
    return ticks


@pytest.mark.asyncio
async def test_set_cached_keyframes_offloads_image_decode(monkeypatch):
    from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

    pipe = object.__new__(VideoIngestionPipeline)

    async def _set_keyframes(*a, **k):
        return None

    pipe.cache = SimpleNamespace(set_keyframes=_set_keyframes)
    pipe._keyframe_cache_kwargs = lambda: {}

    def _blocking_load(meta):
        time.sleep(0.3)  # cv2.imread decode of every frame
        return {}

    monkeypatch.setattr(pipe, "_load_keyframe_images", _blocking_load)

    meta = {"keyframes": [{"frame_id": 0, "path": "/x.jpg"}]}
    ticks = await _ticks_during(lambda: pipe.set_cached_keyframes(Path("/v.mp4"), meta))
    assert ticks >= 10, f"only {ticks} ticks — cv2 frame decode ran on the loop"


@pytest.mark.asyncio
async def test_get_cached_keyframes_offloads_rehydrate(monkeypatch):
    from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

    pipe = object.__new__(VideoIngestionPipeline)

    async def _get_keyframes(*a, **k):
        return ({"keyframes": [{"frame_id": 0}]}, {"0": object()})

    pipe.cache = SimpleNamespace(get_keyframes=_get_keyframes)
    pipe._keyframe_cache_kwargs = lambda: {}

    def _blocking_rehydrate(video_path, metadata, images):
        time.sleep(0.3)  # cv2.imwrite re-encode of every frame back to disk

    monkeypatch.setattr(pipe, "_rehydrate_keyframe_images", _blocking_rehydrate)

    ticks = await _ticks_during(lambda: pipe.get_cached_keyframes(Path("/v.mp4")))
    assert ticks >= 10, f"only {ticks} ticks — cv2 frame re-encode ran on the loop"


@pytest.mark.asyncio
async def test_segmentation_offloads_keyframe_upload(monkeypatch):
    from cogniverse_runtime.ingestion.processing_strategy_set import (
        ProcessingStrategySet,
    )

    pss = object.__new__(ProcessingStrategySet)

    def _blocking_upload(video_path, metadata):
        time.sleep(0.3)  # MinIO HTTP upload of every keyframe

    async def _get_cached(video_path):
        return None

    async def _set_cached(video_path, result):
        return None

    processor = SimpleNamespace(
        extract_keyframes=lambda vp, out: {"keyframes": [{"frame_id": 0}]}
    )
    processor_manager = SimpleNamespace(get_processor=lambda name: processor)
    pipeline_context = SimpleNamespace(
        config=SimpleNamespace(extract_keyframes=True),
        get_cached_keyframes=_get_cached,
        set_cached_keyframes=_set_cached,
        upload_keyframes_to_object_store=_blocking_upload,
        profile_output_dir=Path("/tmp"),
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )
    strategy = SimpleNamespace(get_required_processors=lambda: ["keyframe"])

    ticks = await _ticks_during(
        lambda: pss._process_segmentation(
            strategy, Path("/v.mp4"), processor_manager, pipeline_context
        )
    )
    assert ticks >= 10, f"only {ticks} ticks — MinIO keyframe upload ran on the loop"


@pytest.mark.asyncio
async def test_segmentation_skips_keyframes_when_disabled():
    """extract_keyframes=False in pipeline_config skips keyframe extraction,
    matching the transcribe_audio/generate_descriptions/generate_embeddings
    gates — the flag was silently ignored, extracting frames regardless."""
    from cogniverse_runtime.ingestion.processing_strategy_set import (
        ProcessingStrategySet,
    )

    pss = object.__new__(ProcessingStrategySet)

    extracted = []

    def _extract(video_path, out_dir):
        extracted.append(video_path)
        return {"keyframes": [{"frame_id": 0}]}

    async def _get_cached(video_path):
        return None

    async def _set_cached(video_path, result):
        return None

    def _upload(video_path, result):
        return None

    processor = SimpleNamespace(extract_keyframes=_extract)
    processor_manager = SimpleNamespace(get_processor=lambda name: processor)
    pipeline_context = SimpleNamespace(
        config=SimpleNamespace(extract_keyframes=False),
        get_cached_keyframes=_get_cached,
        set_cached_keyframes=_set_cached,
        upload_keyframes_to_object_store=_upload,
        profile_output_dir=Path("/tmp"),
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )
    strategy = SimpleNamespace(get_required_processors=lambda: ["keyframe"])

    result = await pss._process_segmentation(
        strategy, Path("/v.mp4"), processor_manager, pipeline_context
    )
    assert result == {}, f"expected no keyframes when disabled, got {result}"
    assert extracted == [], "keyframe extraction ran despite extract_keyframes=False"
