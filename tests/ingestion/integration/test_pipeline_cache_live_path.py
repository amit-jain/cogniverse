"""Integration tests for the live-path pipeline cache wiring (Piece B).

Drives the real ``ProcessingStrategySet`` strategy methods against a real
``PipelineArtifactCache`` + filesystem backend on disk. The "fresh pod"
scenario shares the cache backend dir but uses a new ``profile_output_dir`` —
proving a second pod skips extraction (cache hit) and rehydrates keyframe
image files to its own disk so downstream steps can open them.
"""

from __future__ import annotations

import logging
import subprocess
import types
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from cogniverse_core.common.cache.base import CacheConfig, CacheManager
from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from tests.system.minio_test_manager import MinIOTestManager


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def minio():
    if not _docker_available():
        pytest.skip("Docker not available")
    pytest.importorskip("boto3")
    manager = MinIOTestManager()
    instance = manager.start()
    try:
        yield instance
    finally:
        manager.stop()


def _s3_backend(instance, bucket):
    return {
        "backend_type": "s3",
        "endpoint": instance.endpoint,
        "access_key": instance.access_key,
        "secret_key": instance.secret_key,
        "bucket": bucket,
        "key_prefix": "pipeline/",
        "serialization_format": "pickle",
        "priority": 1,
        "enabled": True,
    }


TRANSCRIPT = {
    "text": "hello world",
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
}
DESCRIPTIONS = {"frame_descriptions": {"0": "a frame"}, "model": "vlm"}


def _make_pipeline(
    cache_dir: Path, output_dir: Path, extra_backends: list | None = None
) -> VideoIngestionPipeline:
    """A real pipeline instance wired to a shared filesystem cache backend.

    Two pipelines built with the same ``cache_dir`` but different
    ``output_dir`` model two pods sharing one cache tier. ``extra_backends``
    appends lower-priority tiers (e.g. an S3/MinIO L2) shared across pods.
    """
    manager = CacheManager(
        CacheConfig(
            backends=[
                {
                    "backend_type": "structured_filesystem",
                    "base_path": str(cache_dir),
                    "serialization_format": "pickle",
                    "priority": 0,
                    "enable_ttl": False,
                    "cleanup_on_startup": False,
                }
            ]
            + (extra_backends or []),
            default_ttl=0,
        )
    )
    pipe = VideoIngestionPipeline.__new__(VideoIngestionPipeline)
    pipe.cache = PipelineArtifactCache(manager, ttl=0, profile="testprof")
    pipe.schema_name = "testprof"
    pipe.profile_output_dir = output_dir
    pipe.logger = logging.getLogger("test_pipeline_cache")
    pipe.config = types.SimpleNamespace(
        keyframe_threshold=0.999,
        max_frames_per_video=3000,
        vlm_batch_size=500,
        extract_keyframes=True,
        transcribe_audio=True,
        generate_descriptions=True,
    )
    pipe.app_config = {
        "backend": {
            "profiles": {
                "testprof": {
                    "pipeline_config": {"keyframe_strategy": "fps", "keyframe_fps": 1.0}
                }
            }
        }
    }
    # NOTE: deliberately do NOT set ``audio_transcriber`` — the real
    # VideoIngestionPipeline never sets that attribute, so the cache-kwargs
    # helpers must resolve the audio model from the processor manager. Setting
    # it here would mask the AttributeError the live path hit in e2e.
    pipe.processor_manager = _FakeProcessorManager(
        vlm=types.SimpleNamespace(vlm_endpoint="http://vlm"),
        audio=types.SimpleNamespace(model="base"),
    )
    return pipe


class _FakeProcessorManager:
    def __init__(self, **procs):
        self._procs = procs

    def get_processor(self, name):
        return self._procs.get(name)


class _FakeKeyframeProcessor:
    def __init__(self):
        self.calls = 0

    def extract_keyframes(self, video_path, output_dir):
        self.calls += 1
        kf_dir = Path(output_dir) / "keyframes" / Path(video_path).stem
        kf_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{Path(video_path).stem}_keyframe_0000.jpg"
        fpath = kf_dir / filename
        cv2.imwrite(str(fpath), np.full((4, 4, 3), 9, dtype=np.uint8))
        return {
            "keyframes": [
                {
                    "frame_number": 0,
                    "timestamp": 0.0,
                    "filename": filename,
                    "path": str(fpath),
                }
            ],
            "metadata": {"total_keyframes": 1},
        }


class _FakeAudioProcessor:
    def __init__(self):
        self.calls = 0

    def transcribe_audio(self, video_path, output_dir, _language):
        self.calls += 1
        return dict(TRANSCRIPT)


class _SegStrategy:
    def get_required_processors(self):
        return {"keyframe": {}}


class _TranscriptionStrategy:
    def get_required_processors(self):
        return {"audio": {}}


class _DescriptionStrategy:
    def __init__(self):
        self.calls = 0

    def get_required_processors(self):
        return {"vlm": {}}

    async def generate_descriptions(self, frames_metadata, video_path, ctx, opts):
        self.calls += 1
        return dict(DESCRIPTIONS)


@pytest.fixture
def video(tmp_path):
    v = tmp_path / "v1.mp4"
    v.write_bytes(b"\x00")
    return v


@pytest.mark.asyncio
async def test_keyframes_hit_skips_extraction_and_rehydrates_on_fresh_pod(
    tmp_path, video
):
    cache_dir = tmp_path / "cache"
    pss = ProcessingStrategySet()
    seg = _SegStrategy()
    kf_proc = _FakeKeyframeProcessor()
    pm = _FakeProcessorManager(keyframe=kf_proc)

    # Pod 1 — extracts and caches
    pipe1 = _make_pipeline(cache_dir, tmp_path / "pod1")
    r1 = await pss._process_segmentation(seg, video, pm, pipe1)
    assert kf_proc.calls == 1
    assert len(r1["keyframes"]["keyframes"]) == 1
    assert Path(r1["keyframes"]["keyframes"][0]["path"]).exists()

    # Pod 2 — fresh output dir, shared cache: must hit, not re-extract
    pipe2 = _make_pipeline(cache_dir, tmp_path / "pod2")
    r2 = await pss._process_segmentation(seg, video, pm, pipe2)
    assert kf_proc.calls == 1  # extraction NOT repeated — served from cache
    kf = r2["keyframes"]["keyframes"][0]
    # frame file rehydrated under pod2's dir and path repointed there
    assert Path(kf["path"]).exists()
    assert str(tmp_path / "pod2") in kf["path"]


@pytest.mark.asyncio
async def test_transcript_hit_skips_transcription_on_fresh_pod(tmp_path, video):
    cache_dir = tmp_path / "cache"
    pss = ProcessingStrategySet()
    strat = _TranscriptionStrategy()
    audio_proc = _FakeAudioProcessor()
    pm = _FakeProcessorManager(audio=audio_proc)

    pipe1 = _make_pipeline(cache_dir, tmp_path / "pod1")
    r1 = await pss._process_transcription(strat, video, pm, pipe1, {})
    assert audio_proc.calls == 1
    assert r1["transcript"] == TRANSCRIPT

    pipe2 = _make_pipeline(cache_dir, tmp_path / "pod2")
    r2 = await pss._process_transcription(strat, video, pm, pipe2, {})
    assert audio_proc.calls == 1  # cache hit, no re-transcription
    assert r2["transcript"] == TRANSCRIPT


@pytest.mark.asyncio
async def test_failed_transcript_is_not_cached(tmp_path, video):
    pipe = _make_pipeline(tmp_path / "cache", tmp_path / "pod1")

    # transcribe_audio returns an ``error`` dict when ASR is unreachable;
    # it must not be cached, so a later ingest re-transcribes instead of
    # serving the stale failure.
    failed = {"error": "asr down", "full_text": "", "segments": []}
    await pipe.set_cached_transcript(video, failed)
    assert await pipe.get_cached_transcript(video) is None

    # a successful transcript is cached and served
    await pipe.set_cached_transcript(video, TRANSCRIPT)
    assert await pipe.get_cached_transcript(video) == TRANSCRIPT


@pytest.mark.asyncio
async def test_descriptions_hit_skips_vlm_on_fresh_pod(tmp_path, video):
    cache_dir = tmp_path / "cache"
    pss = ProcessingStrategySet()
    strat = _DescriptionStrategy()
    pm = _FakeProcessorManager(vlm=types.SimpleNamespace(vlm_endpoint="http://vlm"))

    pipe1 = _make_pipeline(cache_dir, tmp_path / "pod1")
    r1 = await pss._process_description(strat, video, pm, pipe1, {"keyframes": {}})
    assert strat.calls == 1
    assert r1["descriptions"] == DESCRIPTIONS

    pipe2 = _make_pipeline(cache_dir, tmp_path / "pod2")
    r2 = await pss._process_description(strat, video, pm, pipe2, {"keyframes": {}})
    assert strat.calls == 1  # cache hit, VLM not re-run
    assert r2["descriptions"] == DESCRIPTIONS


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_keyframes_wiring_hits_shared_l2_on_fresh_pod(minio, tmp_path, video):
    bucket = f"cache-{uuid.uuid4().hex[:8]}"
    pss = ProcessingStrategySet()
    seg = _SegStrategy()
    kf_proc = _FakeKeyframeProcessor()
    pm = _FakeProcessorManager(keyframe=kf_proc)

    # Pod 1: extract + cache through L1(local) + L2(shared MinIO)
    pipe1 = _make_pipeline(
        tmp_path / "pod1",
        tmp_path / "pod1out",
        extra_backends=[_s3_backend(minio, bucket)],
    )
    r1 = await pss._process_segmentation(seg, video, pm, pipe1)
    assert kf_proc.calls == 1
    assert len(r1["keyframes"]["keyframes"]) == 1

    # Pod 2: fresh empty L1, SAME shared MinIO L2 bucket
    pipe2 = _make_pipeline(
        tmp_path / "pod2_empty",
        tmp_path / "pod2out",
        extra_backends=[_s3_backend(minio, bucket)],
    )
    r2 = await pss._process_segmentation(seg, video, pm, pipe2)

    assert kf_proc.calls == 1  # NOT re-extracted — served from shared MinIO L2
    kf = r2["keyframes"]["keyframes"][0]
    assert Path(kf["path"]).exists()  # frame rehydrated to pod2's own disk
    assert str(tmp_path / "pod2out") in kf["path"]
    # the hit was served by the S3/MinIO (L2) backend, not the empty L1
    s3 = pipe2.cache.cache.backends[1]
    assert s3.__class__.__name__ == "S3CacheBackend"
    assert (await s3.get_stats())["hits"] >= 1
