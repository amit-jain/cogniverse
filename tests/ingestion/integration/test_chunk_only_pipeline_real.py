"""Real-pipeline regression: a chunk-only profile (no description strategy)
processes a real video end-to-end.

Chunk-based profiles pair chunk segmentation with ``NoDescriptionStrategy`` —
they embed the video chunks directly and never run keyframe VLM descriptions.
This drives the real ``ProcessingStrategySet.process`` with a real
``ChunkProcessor`` (ffmpeg) over a real video, proving the chunk path produces
``video_chunks`` and no descriptions, cleanly — the path the chunk-vs-VLM
construction guard leaves intact.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.processors.chunk_processor import ChunkProcessor
from cogniverse_runtime.ingestion.strategies import ChunkSegmentationStrategy

pytestmark = pytest.mark.integration


def _make_video(path: Path, seconds: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size=64x64:rate=5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=True,
    )


@pytest.mark.asyncio
async def test_chunk_only_profile_processes_cleanly_end_to_end(tmp_path, monkeypatch):
    # No-op telemetry so the stage spans don't need BACKEND_URL wiring.
    class _Span:
        def set_attribute(self, *a, **k):
            pass

    @contextmanager
    def _span(*a, **k):
        yield _Span()

    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: SimpleNamespace(span=_span),
    )

    video = tmp_path / "clip.mp4"
    _make_video(video, seconds=3)

    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    pm = ProcessorManager(logger)
    # Real ffmpeg chunk processor, 1s chunks over a 3s video -> 3 real chunks.
    pm._processors["chunk"] = ChunkProcessor(
        logger, chunk_duration=1.0, chunk_overlap=0.0, cache_chunks=False
    )

    ctx = SimpleNamespace(
        tenant_id="acme:acme",
        schema_name="video_colqwen_chunks",
        logger=logger,
        profile_output_dir=tmp_path,
        processor_manager=pm,
        config=SimpleNamespace(generate_descriptions=True, extract_keyframes=True),
    )

    pss = ProcessingStrategySet(
        segmentation=ChunkSegmentationStrategy(chunk_duration=1.0, cache_chunks=False),
    )

    result = await pss.process(video, pm, ctx)

    # Real chunk segmentation produced the three chunk files under video_chunks.
    assert "video_chunks" in result
    chunks = result["video_chunks"]["chunks"]
    assert len(chunks) == 3
    for chunk in chunks:
        chunk_path = Path(chunk["path"])
        assert chunk_path.exists() and chunk_path.stat().st_size > 0

    # No description strategy configured, so no descriptions are attempted and
    # the run completes without touching the keyframe VLM.
    assert "descriptions" not in result
