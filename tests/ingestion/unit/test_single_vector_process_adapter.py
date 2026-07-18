"""SingleVectorVideoProcessor.process forwards transcript_data/metadata.

The adapter passed its 2nd positional (output_dir) into process_video's
transcript_data slot and dropped **kwargs, so transcript_data/metadata never
reached process_video.
"""

from __future__ import annotations

from pathlib import Path

from cogniverse_runtime.ingestion.processors.single_vector_processor import (
    SingleVectorVideoProcessor,
)


def test_process_forwards_transcript_and_metadata():
    proc = object.__new__(SingleVectorVideoProcessor)
    captured = {}

    def fake_process_video(video_path, transcript_data=None, metadata=None):
        captured.update(vp=video_path, td=transcript_data, md=metadata)
        return {}

    proc.process_video = fake_process_video

    proc.process(
        Path("v.mp4"),
        output_dir=Path("/tmp/out"),
        transcript_data={"full_text": "hi"},
        metadata={"a": 1},
    )

    assert captured["td"] == {"full_text": "hi"}
    assert captured["md"] == {"a": 1}
    assert captured["vp"] == Path("v.mp4")


import pytest


@pytest.mark.asyncio
async def test_segment_strategy_hands_the_pipeline_cache_to_the_processor():
    """The processor's cache knob is wired from the pipeline context, so an
    enabled pipeline cache reaches segmentation without every construction
    site threading it through."""
    from types import SimpleNamespace

    from cogniverse_runtime.ingestion.strategies import (
        SingleVectorSegmentationStrategy,
    )

    class _RecordingCache:
        def __init__(self):
            self.get_calls = 0

        async def get_segmentation(self, video_path, **kwargs):
            self.get_calls += 1
            return None

        async def set_segmentation(self, video_path, result, **kwargs):
            return True

    proc = object.__new__(SingleVectorVideoProcessor)
    proc.cache = None
    seen = {}

    def fake_process_video(video_path, transcript_data=None):
        seen["cache_at_call"] = proc.cache
        return {"segments": [], "metadata": {}, "full_transcript": ""}

    proc.process_video = fake_process_video

    cache = _RecordingCache()
    context = SimpleNamespace(
        processor_manager=SimpleNamespace(get_processor=lambda name: proc),
        cache=cache,
    )

    strategy = SingleVectorSegmentationStrategy()
    out = await strategy.segment(Path("v.mp4"), context)

    assert seen["cache_at_call"] is cache
    assert proc.cache is cache
    assert out["single_vector_processing"]["segments"] == []
