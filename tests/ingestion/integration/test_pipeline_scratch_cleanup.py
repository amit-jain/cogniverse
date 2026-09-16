"""Job-owned media survives its consumers and is released on every exit."""

import asyncio
import hashlib
import logging
import threading
from pathlib import Path

import pytest
from PIL import Image

from cogniverse_runtime.ingestion.pipeline import PipelineConfig, VideoIngestionPipeline
from tests.ingestion.integration.test_required_transcription import make_video

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def scratch_pipeline(tmp_path, *, pages=False):
    strategy = (
        {"class": "DocumentVisualSegmentationStrategy", "params": {"dpi": 72}}
        if pages
        else {
            "class": "ChunkSegmentationStrategy",
            "params": {
                "chunk_duration": 0.5,
                "chunk_overlap": 0.0,
                "cache_chunks": False,
            },
        }
    )
    pipeline = VideoIngestionPipeline(
        tenant_id="prodfixingestion:scratch",
        schema_name="scratch",
        config=PipelineConfig(
            generate_embeddings=False,
            transcribe_audio=False,
            generate_descriptions=False,
        ),
        app_config={
            "backend": {
                "profiles": {
                    "scratch": {
                        "strategies": {"segmentation": strategy},
                    }
                }
            }
        },
    )
    pipeline.profile_output_dir = tmp_path / "processing"
    pipeline.profile_output_dir.mkdir()
    return pipeline


@pytest.mark.parametrize("pages", [False, True])
async def test_job_removes_generated_media_and_preserves_source(tmp_path, pages):
    source = tmp_path / ("source.pdf" if pages else "source.mp4")
    if pages:
        Image.new("RGB", (32, 32), "blue").save(source, "PDF")
    else:
        make_video(source, audio=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    pipeline = scratch_pipeline(tmp_path, pages=pages)
    result = await pipeline.process_video_async_with_strategies(source)
    assert result["status"] == "completed"
    if pages:
        paths = [Path(page["path"]) for page in result["results"]["document_pages"]]
        assert len(paths) == 1
    else:
        chunks = result["results"]["video_chunks"]["chunks"]
        assert [(c["start_time"], c["end_time"]) for c in chunks] == [
            (0.0, 0.5),
            (0.5, 1.0),
        ]
        paths = [Path(chunk["path"]) for chunk in chunks]
    assert [path.exists() for path in paths] == [False] * len(paths)
    assert list(pipeline.profile_output_dir.rglob("*")) == []
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


async def test_ffmpeg_failure_removes_already_written_chunks(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, audio=False)
    pipeline = scratch_pipeline(tmp_path)
    processor = pipeline.processor_manager.get_processor("chunk")
    extract = processor._extract_chunk
    written = []

    def fail_second(video_path, chunk_path, start, duration):
        if start == 0.5:
            # The decoder sees an absent input after the first output was written.
            return extract(tmp_path / "missing.mp4", chunk_path, start, duration)
        value = extract(video_path, chunk_path, start, duration)
        written.append(chunk_path)
        return value

    monkeypatch.setattr(processor, "_extract_chunk", fail_second)
    result = await pipeline.process_video_async_with_strategies(source)
    assert result["status"] == "failed"
    assert len(written) == 1
    assert written[0].exists() is False
    assert list(pipeline.profile_output_dir.rglob("*")) == []
    assert source.exists() is True


async def test_cancelled_job_waits_for_decoder_and_preserves_concurrent_job(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.mp4"
    make_video(source, audio=False)
    pipeline = scratch_pipeline(tmp_path)
    processor = pipeline.processor_manager.get_processor("chunk")
    extract = processor._extract_chunk
    entered = threading.Barrier(3, timeout=10)
    release = threading.Event()
    paths = []
    guard = threading.Lock()

    def paused_extract(video_path, chunk_path, start, duration):
        if start == 0.0:
            with guard:
                paths.append(chunk_path)
            entered.wait()
            assert release.wait(10) is True
        return extract(video_path, chunk_path, start, duration)

    monkeypatch.setattr(processor, "_extract_chunk", paused_extract)
    first = asyncio.create_task(pipeline.process_video_async_with_strategies(source))
    second = asyncio.create_task(pipeline.process_video_async_with_strategies(source))
    try:
        await asyncio.to_thread(entered.wait)
        first.cancel()
        await asyncio.sleep(0.05)
        assert first.done() is False
        assert len(set(paths)) == 2
        assert [path.parent.exists() for path in paths] == [True, True]
    finally:
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
    assert type(results[0]) is asyncio.CancelledError
    assert results[1]["status"] == "completed"
    assert list(pipeline.profile_output_dir.rglob("*")) == []
    assert source.exists() is True


def keyframe_pipeline(tmp_path, cache_dir):
    """A frame-profile pipeline sharing one cache tier across runs."""
    pipeline = VideoIngestionPipeline(
        tenant_id="prodfixingestion:scratch",
        schema_name="scratchframes",
        config=PipelineConfig(
            generate_embeddings=False,
            transcribe_audio=False,
            generate_descriptions=False,
            extract_keyframes=True,
        ),
        app_config={
            "pipeline_cache": {
                "enabled": True,
                "backends": [
                    {
                        "backend_type": "structured_filesystem",
                        "base_path": str(cache_dir),
                        "serialization_format": "pickle",
                        "priority": 0,
                        "enable_ttl": False,
                        "cleanup_on_startup": False,
                    }
                ],
                "default_ttl": 0,
                "serialization_format": "pickle",
            },
            "backend": {
                "profiles": {
                    "scratchframes": {
                        "strategies": {
                            "segmentation": {
                                "class": "FrameSegmentationStrategy",
                                "params": {"fps": 2.0, "threshold": 0.999},
                            }
                        },
                    }
                }
            },
        },
    )
    pipeline.profile_output_dir = tmp_path
    return pipeline


async def test_keyframe_cache_hit_releases_the_frames_it_rehydrates(tmp_path):
    source = tmp_path / "source.mp4"
    make_video(source, audio=False)
    cache_dir = tmp_path / "cache"
    output = tmp_path / "processing"
    output.mkdir()

    extracting = keyframe_pipeline(output, cache_dir)
    first = await extracting.process_video_async_with_strategies(source)
    assert first["status"] == "completed"
    extracted = [kf["path"] for kf in first["results"]["keyframes"]["keyframes"]]
    assert extracted != []
    assert list(output.rglob("*")) == []

    rehydrating = keyframe_pipeline(output, cache_dir)
    second = await rehydrating.process_video_async_with_strategies(source)
    assert second["status"] == "completed"
    rehydrated = [kf["path"] for kf in second["results"]["keyframes"]["keyframes"]]
    assert [Path(path).name for path in rehydrated] == [
        Path(path).name for path in extracted
    ]
    assert rehydrated != extracted
    assert list(output.rglob("*")) == []


async def test_scratch_that_cannot_be_released_is_reported(tmp_path, caplog):
    source = tmp_path / "source.mp4"
    make_video(source, audio=False)
    pipeline = scratch_pipeline(tmp_path)
    held = []

    original = pipeline._release_job_scratch

    def block_then_release(scratch_dir):
        held.append(scratch_dir)
        scratch_dir.chmod(0o500)
        try:
            original(scratch_dir)
        finally:
            scratch_dir.chmod(0o700)

    pipeline._release_job_scratch = block_then_release
    with caplog.at_level("WARNING", logger=pipeline.logger.name):
        result = await pipeline.process_video_async_with_strategies(source)
    assert result["status"] == "completed"
    assert len(held) == 1
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ] == [
        f"Job scratch {held[0]} still holds "
        f"{len(list(held[0].rglob('*')))} path(s) after release"
    ]
    original(held[0])
    assert list(pipeline.profile_output_dir.rglob("*")) == []
