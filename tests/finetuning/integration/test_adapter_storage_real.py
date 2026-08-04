"""Real-MinIO integration tests for adapter storage backends."""

from __future__ import annotations

import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cogniverse_finetuning.registry.storage import S3Storage, S3StorageConfig
from tests.system.minio_test_manager import MinIOTestManager

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def minio():
    manager = MinIOTestManager()
    instance = manager.start(name_prefix="finetuning-adapter-storage")
    try:
        yield instance
    finally:
        manager.stop()


def _build_adapter_tree(root: Path, index: int) -> tuple[Path, dict[str, bytes]]:
    adapter_dir = root / f"adapter_{index}"
    (adapter_dir / "nested").mkdir(parents=True)

    config = f'{{"adapter": {index}, "kind": "routing"}}'.encode("utf-8")
    weights = f"weights-{index}".encode("utf-8")
    readme = f"adapter {index}\n".encode("utf-8")

    (adapter_dir / "config.json").write_bytes(config)
    (adapter_dir / "nested" / "weights.bin").write_bytes(weights)
    (adapter_dir / "README.txt").write_bytes(readme)

    return adapter_dir, {
        "config.json": config,
        "nested/weights.bin": weights,
        "README.txt": readme,
    }


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _build_s3_storage(minio):
    return S3Storage(
        S3StorageConfig(
            endpoint_url=minio.endpoint,
            access_key=minio.access_key,
            secret_key=minio.secret_key,
        )
    )


@pytest.mark.requires_docker
class TestS3AdapterStorageReal:
    def test_directory_round_trip_preserves_nested_files(self, minio, tmp_path):
        bucket = f"adapter-{uuid.uuid4().hex[:8]}"
        client = minio.boto3_client()
        client.create_bucket(Bucket=bucket)

        source, expected = _build_adapter_tree(tmp_path, 0)
        stale_source, stale_expected = _build_adapter_tree(tmp_path / "stale", 9)

        storage = _build_s3_storage(minio)
        destination_uri = f"s3://{bucket}/adapters/routing_sft_v1.0.0"
        downloaded = tmp_path / "downloaded"
        shutil.copytree(stale_source, downloaded)
        assert _snapshot_tree(downloaded) == stale_expected

        uploaded_uri = storage.upload(str(source), destination_uri)
        assert uploaded_uri == destination_uri

        listed = client.list_objects_v2(
            Bucket=bucket, Prefix="adapters/routing_sft_v1.0.0/"
        )
        keys = sorted(obj["Key"] for obj in listed.get("Contents", []) or [])
        assert keys == [
            "adapters/routing_sft_v1.0.0/README.txt",
            "adapters/routing_sft_v1.0.0/config.json",
            "adapters/routing_sft_v1.0.0/nested/weights.bin",
        ]

        result_path = Path(storage.download(destination_uri, str(downloaded)))
        assert result_path == downloaded
        assert _snapshot_tree(downloaded) == expected

    def test_download_preserves_existing_content_when_source_missing(
        self, minio, tmp_path
    ):
        bucket = f"adapter-{uuid.uuid4().hex[:8]}"
        client = minio.boto3_client()
        client.create_bucket(Bucket=bucket)

        existing_source, existing_expected = _build_adapter_tree(tmp_path, 1)
        downloaded = tmp_path / "downloaded"
        shutil.copytree(existing_source, downloaded)
        before = _snapshot_tree(downloaded)

        storage = _build_s3_storage(minio)

        with pytest.raises(
            FileNotFoundError,
            match="Source adapter not found: s3://",
        ):
            storage.download(
                f"s3://{bucket}/adapters/missing_adapter",
                str(downloaded),
            )

        assert _snapshot_tree(downloaded) == before == existing_expected

    def test_concurrent_uploads_do_not_cross_contaminate(self, minio, tmp_path):
        bucket = f"adapter-{uuid.uuid4().hex[:8]}"
        client = minio.boto3_client()
        client.create_bucket(Bucket=bucket)

        storage = _build_s3_storage(minio)

        barrier = threading.Barrier(4)
        expected: dict[str, bytes] = {}
        jobs = []
        for index in range(4):
            source, payloads = _build_adapter_tree(tmp_path / "sources", index)

            destination_uri = f"s3://{bucket}/adapters/adapter_{index}"
            expected[f"adapters/adapter_{index}/README.txt"] = payloads["README.txt"]
            expected[f"adapters/adapter_{index}/config.json"] = payloads["config.json"]
            expected[f"adapters/adapter_{index}/nested/weights.bin"] = payloads[
                "nested/weights.bin"
            ]
            jobs.append((source, destination_uri))

        def _upload(item):
            source_path, destination_uri = item
            barrier.wait()
            return storage.upload(str(source_path), destination_uri)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_upload, jobs))

        assert sorted(results) == sorted(destination for _, destination in jobs)

        listed = client.list_objects_v2(Bucket=bucket, Prefix="adapters/")
        keys = sorted(obj["Key"] for obj in listed.get("Contents", []) or [])
        assert keys == sorted(expected)

        for key, body in expected.items():
            response = client.get_object(Bucket=bucket, Key=key)
            assert response["Body"].read() == body

    def test_download_raises_when_endpoint_is_unreachable(self, tmp_path):
        storage = S3Storage(
            S3StorageConfig(
                endpoint_url="http://127.0.0.1:65535",
                access_key="dummy",
                secret_key="dummy",
            )
        )

        with pytest.raises(
            RuntimeError,
            match="failed to download adapter from s3://missing-bucket/adapters/model",
        ):
            storage.download(
                "s3://missing-bucket/adapters/model",
                str(tmp_path / "downloaded"),
            )
