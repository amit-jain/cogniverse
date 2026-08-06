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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.processors.chunk_processor import ChunkProcessor
from cogniverse_runtime.ingestion.strategies import ChunkSegmentationStrategy

pytestmark = pytest.mark.integration
TRACKED_VIDEO = Path("tests/system/resources/videos/v_-D1gdv_gQyw.mp4")


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


def _make_repeated_real_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-stream_loop",
            "-1",
            "-i",
            str(TRACKED_VIDEO),
            "-t",
            "61",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ],
        capture_output=True,
        check=True,
    )


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return int(result.stdout.strip())


def test_each_real_video_chunk_contains_the_exact_frame_span(tmp_path):
    source = tmp_path / "repeated-real-video.mp4"
    _make_repeated_real_video(source)
    processor = ChunkProcessor(
        logger=SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
        chunk_duration=30.0,
        chunk_overlap=0.0,
        cache_chunks=False,
    )

    result = processor.extract_chunks(source, output_dir=tmp_path / "output")

    assert [chunk["duration"] for chunk in result["chunks"]] == [30.0, 30.0, 1.0]
    assert [_video_frame_count(Path(chunk["path"])) for chunk in result["chunks"]] == [
        900,
        900,
        30,
    ]


def test_concurrent_real_chunk_extraction_keeps_segment_frames_isolated(tmp_path):
    source = tmp_path / "repeated-real-video.mp4"
    _make_repeated_real_video(source)
    processor = ChunkProcessor(
        logger=SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
        cache_chunks=False,
    )
    segment_paths = [tmp_path / "first.mp4", tmp_path / "tail.mp4"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda args: processor._extract_chunk(source, *args),
                [(segment_paths[0], 0.0, 30.0), (segment_paths[1], 60.0, 1.0)],
            )
        )

    assert results == [True, True]
    assert [_video_frame_count(path) for path in segment_paths] == [900, 30]


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
