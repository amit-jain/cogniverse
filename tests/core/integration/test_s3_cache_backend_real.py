"""Real-MinIO integration tests for the S3 cache backend.

Spins up a MinIO container via :class:`MinIOTestManager` and exercises the
full ``PipelineArtifactCache`` → ``CacheManager`` → ``S3CacheBackend`` path
against a real S3 API. The pod-restart test is the one that proves the
multi-pod fix: a fresh pod (empty L1) still serves cached artifacts from the
shared L2 bucket.

Requires Docker and boto3; missing test infrastructure is a test failure.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from cogniverse_core.common.cache.base import CacheConfig, CacheManager
from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache
from tests.system.minio_test_manager import MinIOTestManager

pytestmark = pytest.mark.integration

VIDEO = "s3://corpus/v_cache.mp4"


@pytest.fixture(scope="module")
def minio():
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
        meta = {
            "video_id": "v_cache",
            "keyframes": [
                {"frame_id": 0, "timestamp": 0.0, "filename": "frame_0000.jpg"}
            ],
            "strategy": "fps",
            "fps": 1.25,
        }

        # Pod 1: write through L1(local fs) + L2(shared s3)
        manager1 = _manager(
            [
                _fs_backend_dict(tmp_path / "pod1"),
                _s3_backend_dict(minio, bucket),
            ]
        )
        cache1 = PipelineArtifactCache(manager1, ttl=3600, profile="prof")
        assert (
            await cache1.set_keyframes(
                VIDEO, meta, strategy="fps", fps=1.25, max_frames=17
            )
            is True
        )
        video_key = cache1._generate_video_key(VIDEO)
        artifact_key = cache1._generate_artifact_key(
            video_key,
            "keyframes",
            strategy="fps",
            fps=1.25,
            max_frames=17,
        )
        assert await manager1.backends[0].get(artifact_key) == meta
        assert await manager1.backends[1].get(artifact_key) == meta

        # Pod 2: a *fresh* pod — empty L1, same shared L2 bucket
        manager2 = _manager(
            [
                _fs_backend_dict(tmp_path / "pod2_empty"),
                _s3_backend_dict(minio, bucket),
            ]
        )
        cache2 = PipelineArtifactCache(manager2, ttl=3600, profile="prof")
        fresh_l1 = manager2.backends[0]
        shared_l2 = manager2.backends[1]
        assert await fresh_l1.exists(artifact_key) is False
        assert (await fresh_l1.get_stats())["total_files"] == 0
        l2_hits_before = (await shared_l2.get_stats())["hits"]

        got = await cache2.get_keyframes(VIDEO, strategy="fps", fps=1.25, max_frames=17)

        assert got == meta
        assert shared_l2.__class__.__name__ == "S3CacheBackend"
        assert (await shared_l2.get_stats())["hits"] == l2_hits_before + 1
        assert await fresh_l1.exists(artifact_key) is True
        assert await fresh_l1.get(artifact_key) == meta
        assert (await fresh_l1.get_stats())["total_files"] == 1

    async def test_bucket_lifecycle_expiration_applied(self, minio):
        from cogniverse_core.common.cache.backends.s3 import (
            S3CacheBackend,
            S3CacheBackendConfig,
        )

        bucket = f"cache-{uuid.uuid4().hex[:8]}"
        backend = S3CacheBackend(
            S3CacheBackendConfig(
                endpoint=minio.endpoint,
                access_key=minio.access_key,
                secret_key=minio.secret_key,
                bucket=bucket,
                key_prefix="pipeline/",
                lifecycle_expiration_days=7,
            )
        )
        # first op triggers _s3() -> _ensure_bucket -> _apply_lifecycle
        await backend.set("p:video:abc:transcript", {"x": 1})

        rules = minio.boto3_client().get_bucket_lifecycle_configuration(Bucket=bucket)[
            "Rules"
        ]
        assert len(rules) == 1
        assert rules[0]["Status"] == "Enabled"
        assert rules[0]["Expiration"]["Days"] == 7
        assert rules[0]["Filter"]["Prefix"] == "pipeline/"
