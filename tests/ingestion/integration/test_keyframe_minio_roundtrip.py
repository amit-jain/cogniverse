"""Real-MinIO round trip for the multimodal keyframe path.

The ingestion write side (``minio_client.upload_keyframes``) saves keyframes to
object storage; the answer-time read side (``KeyframeImageResolver`` ->
``MediaLocator`` -> ``dspy.Image``) retrieves them from a search hit. This
exercises BOTH against a real MinIO container — data is saved, then retrieved
as the exact bytes and as fetchable ``dspy.Image`` inputs — not a mocked
boundary.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import dspy
import pytest
from PIL import Image

from cogniverse_agents.multimodal import KeyframeImageResolver
from cogniverse_core.common.media import (
    MediaCacheConfig,
    MediaConfig,
    MediaLocator,
    S3BackendConfig,
)
from cogniverse_runtime.ingestion_worker import minio_client
from tests.system.minio_test_manager import MinIOTestManager

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture(scope="module")
def minio():
    pytest.importorskip("boto3")
    manager = MinIOTestManager()
    instance = manager.start()
    try:
        yield instance
    finally:
        manager.stop()


@pytest.fixture
def minio_env(monkeypatch, minio):
    # Write side: upload_keyframes -> minio_client._client() / _default_bucket().
    monkeypatch.setenv("MINIO_ENDPOINT", minio.endpoint)
    monkeypatch.setenv("MINIO_ACCESS_KEY", minio.access_key)
    monkeypatch.setenv("MINIO_SECRET_KEY", minio.secret_key)
    # Read side: MediaLocator's fsspec/s3fs credentials.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", minio.access_key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", minio.secret_key)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def test_keyframes_saved_by_ingestion_then_retrieved_by_agent(
    minio_env, minio, tmp_path
):
    bucket = f"media-{uuid.uuid4().hex[:8]}"
    minio.boto3_client().create_bucket(Bucket=bucket)
    tenant_id, video_id = "acme:acme", "vid1"

    # Three distinct real keyframe jpgs on disk (as the extractor writes them).
    paths = []
    for i in range(3):
        p = tmp_path / f"{video_id}_keyframe_{i:04d}.jpg"
        Image.new("RGB", (8, 8), (30 + i * 60, 10, 10)).save(p)
        paths.append(str(p))

    # --- SAVE: ingestion write path uploads to the real MinIO ---
    uris = minio_client.upload_keyframes(
        tenant_id=tenant_id,
        video_id=video_id,
        keyframe_paths=paths,
        bucket=bucket,
    )
    assert uris == [
        f"s3://{bucket}/{tenant_id}/keyframes/{video_id}/{i:04d}.jpg" for i in range(3)
    ]

    # --- RETRIEVE: agent read path fetches from the real MinIO ---
    locator = MediaLocator(
        tenant_id=tenant_id,
        config=MediaConfig(
            cache=MediaCacheConfig(max_bytes_gb=1),
            s3=S3BackendConfig(
                endpoint_url=minio.endpoint, region="us-east-1", anon=False
            ),
        ),
        cache_root=tmp_path / "media-cache",
    )
    resolver = KeyframeImageResolver(locator)
    hits = [
        {
            "source_url": f"s3://{bucket}/{tenant_id}/{video_id}.mp4",
            "video_id": video_id,
            "segment_id": i,
        }
        for i in range(3)
    ]
    images = resolver.collect(hits, max_images=4)

    # Every keyframe came back as a fetchable dspy.Image.
    assert len(images) == 3
    assert all(isinstance(im, dspy.Image) for im in images)

    # And the exact bytes round-tripped: the object the resolver's URI points at
    # is byte-identical to the file ingestion uploaded.
    fetched = locator.localize(uris[1])
    assert fetched.read_bytes() == Path(paths[1]).read_bytes()


def test_missing_keyframe_is_skipped_not_raised(minio_env, minio, tmp_path):
    """A hit whose keyframe was never uploaded degrades to no image, not an
    error — the read path's designed pre-backfill behavior against real MinIO."""
    bucket = f"media-{uuid.uuid4().hex[:8]}"
    minio.boto3_client().create_bucket(Bucket=bucket)

    locator = MediaLocator(
        tenant_id="t:t",
        config=MediaConfig(
            cache=MediaCacheConfig(max_bytes_gb=1),
            s3=S3BackendConfig(
                endpoint_url=minio.endpoint, region="us-east-1", anon=False
            ),
        ),
        cache_root=tmp_path / "media-cache",
    )
    resolver = KeyframeImageResolver(locator)
    hit = {
        "source_url": f"s3://{bucket}/t:t/never-uploaded.mp4",
        "video_id": "never-uploaded",
        "segment_id": 0,
    }
    assert resolver.collect([hit], max_images=4) == []
