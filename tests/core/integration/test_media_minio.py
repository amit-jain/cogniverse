"""Integration test for MediaLocator against a real MinIO instance.

Spins up a single-node MinIO container via :class:`MinIOTestManager` and
exercises the full locator path: ``s3://`` URI → fsspec/s3fs fetch →
tenant-scoped local cache → second call serves from cache without
re-downloading.

Requires Docker. Skips cleanly when Docker is unavailable.
"""

from __future__ import annotations

import uuid

import pytest

from cogniverse_core.common.media import (
    MediaCacheConfig,
    MediaConfig,
    MediaLocator,
    S3BackendConfig,
)
from tests.system.minio_test_manager import MinIOTestManager


@pytest.fixture(scope="module")
def minio_container():
    manager = MinIOTestManager()
    instance = manager.start()
    try:
        yield {
            "endpoint": instance.endpoint,
            "access_key": instance.access_key,
            "secret_key": instance.secret_key,
            "instance": instance,
        }
    finally:
        manager.stop()


@pytest.fixture
def minio_s3_client(minio_container):
    pytest.importorskip("boto3")
    return minio_container["instance"].boto3_client()


@pytest.fixture
def populated_bucket(minio_s3_client, tmp_path_factory):
    bucket = f"corpus-{uuid.uuid4().hex[:8]}"
    minio_s3_client.create_bucket(Bucket=bucket)

    payload_dir = tmp_path_factory.mktemp("payloads")
    clip = payload_dir / "v_minio.mp4"
    clip.write_bytes(b"\x00" * 1024)
    minio_s3_client.upload_file(str(clip), bucket, "videos/v_minio.mp4")

    sub_clip = payload_dir / "v_other.mkv"
    sub_clip.write_bytes(b"\x01" * 1024)
    minio_s3_client.upload_file(str(sub_clip), bucket, "videos/sub/v_other.mkv")
    return bucket


@pytest.fixture
def locator(tmp_path, minio_container):
    config = MediaConfig(
        cache=MediaCacheConfig(max_bytes_gb=1),
        s3=S3BackendConfig(
            endpoint_url=minio_container["endpoint"],
            region="us-east-1",
            anon=False,
        ),
    )
    return MediaLocator(
        tenant_id="test", config=config, cache_root=tmp_path / "media-cache"
    )


@pytest.fixture(autouse=True)
def _minio_credentials(monkeypatch, minio_container):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", minio_container["access_key"])
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", minio_container["secret_key"])
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.mark.requires_docker
class TestMinioLocator:
    def test_localize_downloads_and_caches(
        self, locator, populated_bucket, minio_s3_client
    ):
        uri = f"s3://{populated_bucket}/videos/v_minio.mp4"

        local = locator.localize(uri)

        assert local.exists()
        assert local.stat().st_size == 1024
        assert local.name == "v_minio.mp4"

    def test_second_localize_unchanged_upstream_is_cache_hit(
        self, locator, populated_bucket
    ):
        uri = f"s3://{populated_bucket}/videos/v_minio.mp4"

        first = locator.localize(uri)
        first_mtime = first.stat().st_mtime

        second = locator.localize(uri)

        assert second == first
        assert second.stat().st_mtime == first_mtime

    def test_etag_change_triggers_refetch(
        self, locator, populated_bucket, minio_s3_client
    ):
        uri = f"s3://{populated_bucket}/videos/v_minio.mp4"

        first = locator.localize(uri)
        assert first.stat().st_size == 1024

        new_body = b"DIFFERENT" * 256
        minio_s3_client.put_object(
            Bucket=populated_bucket,
            Key="videos/v_minio.mp4",
            Body=new_body,
        )

        second = locator.localize(uri)

        assert second.stat().st_size == len(new_body)
        assert first.exists()  # old key kept; LRU eviction handles cleanup eventually

    def test_list_yields_video_uris(self, locator, populated_bucket):
        results = list(
            locator.list(f"s3://{populated_bucket}/videos", extensions=(".mp4", ".mkv"))
        )

        assert len(results) == 2
        assert all(r.startswith("s3://") for r in results)
        assert any(r.endswith("v_minio.mp4") for r in results)
        assert any(r.endswith("v_other.mkv") for r in results)

    def test_stat_returns_size(self, locator, populated_bucket):
        stat = locator.stat(f"s3://{populated_bucket}/videos/v_minio.mp4")
        assert stat.size == 1024
