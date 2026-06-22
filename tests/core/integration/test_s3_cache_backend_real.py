"""Real-MinIO integration tests for the S3 cache backend.

Spins up a MinIO container via :class:`MinIOTestManager` and exercises the
full ``PipelineArtifactCache`` → ``CacheManager`` → ``S3CacheBackend`` path
against a real S3 API. The pod-restart test is the one that proves the
multi-pod fix: a fresh pod (empty L1) still serves cached artifacts from the
shared L2 bucket.

Requires Docker. Skips cleanly when Docker is unavailable.
"""

from __future__ import annotations

import subprocess
import uuid

import numpy as np
import pytest

from cogniverse_core.common.cache.base import CacheConfig, CacheManager
from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache
from tests.system.minio_test_manager import MinIOTestManager

VIDEO = "s3://corpus/v_cache.mp4"


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


def _s3_backend_dict(instance, bucket):
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


def _fs_backend_dict(base_path):
    return {
        "backend_type": "structured_filesystem",
        "base_path": str(base_path),
        "serialization_format": "pickle",
        "priority": 0,
        "enable_ttl": True,
        "cleanup_on_startup": False,
    }


def _manager(backends):
    return CacheManager(CacheConfig(backends=backends, default_ttl=3600))


@pytest.mark.requires_docker
class TestS3CacheBackendReal:
    async def test_transcript_round_trips_through_s3(self, minio):
        bucket = f"cache-{uuid.uuid4().hex[:8]}"
        cache = PipelineArtifactCache(
            _manager([_s3_backend_dict(minio, bucket)]), ttl=3600, profile="prof"
        )
        transcript = {
            "segments": [{"text": "hello world", "start": 0.0, "end": 1.5}],
            "language": "en",
        }

        assert await cache.set_transcript(VIDEO, transcript, model_size="base") is True
        assert await cache.get_transcript(VIDEO, model_size="base") == transcript

    async def test_keyframes_with_image_round_trip(self, minio):
        bucket = f"cache-{uuid.uuid4().hex[:8]}"
        cache = PipelineArtifactCache(
            _manager([_s3_backend_dict(minio, bucket)]), ttl=3600, profile="prof"
        )
        meta = {"keyframes": [{"frame_id": 0, "timestamp": 0.0}]}
        img = np.full((4, 4, 3), 7, dtype=np.uint8)

        assert (
            await cache.set_keyframes(
                VIDEO, meta, keyframe_images={"0": img}, strategy="fps", fps=1.0
            )
            is True
        )
        got = await cache.get_keyframes(
            VIDEO, strategy="fps", fps=1.0, load_images=True
        )

        assert isinstance(got, tuple)
        got_meta, images = got
        assert got_meta == meta
        assert "0" in images
        assert images["0"].shape == (4, 4, 3)

    async def test_pod_restart_serves_from_shared_l2(self, minio, tmp_path):
        bucket = f"cache-{uuid.uuid4().hex[:8]}"
        meta = {"keyframes": [{"frame_id": 0, "timestamp": 0.0}]}

        # Pod 1: write through L1(local fs) + L2(shared s3)
        cache1 = PipelineArtifactCache(
            _manager(
                [
                    _fs_backend_dict(tmp_path / "pod1"),
                    _s3_backend_dict(minio, bucket),
                ]
            ),
            ttl=3600,
            profile="prof",
        )
        assert await cache1.set_keyframes(VIDEO, meta, strategy="fps", fps=1.0) is True

        # Pod 2: a *fresh* pod — empty L1, same shared L2 bucket
        manager2 = _manager(
            [
                _fs_backend_dict(tmp_path / "pod2_empty"),
                _s3_backend_dict(minio, bucket),
            ]
        )
        cache2 = PipelineArtifactCache(manager2, ttl=3600, profile="prof")

        got = await cache2.get_keyframes(VIDEO, strategy="fps", fps=1.0)

        assert got == meta
        # the hit was served by the shared S3 (L2) backend, not L1
        s3_backend = manager2.backends[1]
        assert s3_backend.__class__.__name__ == "S3CacheBackend"
        assert (await s3_backend.get_stats())["hits"] >= 1
