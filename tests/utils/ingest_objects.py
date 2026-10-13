"""An ingest bucket on a real MinIO, written through the production writers."""

from __future__ import annotations

import socket
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion_worker import minio_client


@contextmanager
def object_store(config_manager, minio, *, endpoint: str | None = None):
    """A bucket of the caller's own as the runtime's ingest bucket, at
    ``endpoint`` when given instead of ``minio``'s. Yields the bucket name."""
    bucket = f"ingest-{uuid.uuid4().hex[:12]}"
    minio.boto3_client().create_bucket(Bucket=bucket)
    system = config_manager.get_system_config()
    previous = system.minio_endpoint
    system.minio_endpoint = endpoint or minio.endpoint
    config_manager.set_system_config(system)
    with pytest.MonkeyPatch.context() as env:
        env.setenv("MINIO_ENDPOINT", endpoint or minio.endpoint)
        env.setenv("MINIO_ACCESS_KEY", minio.access_key)
        env.setenv("MINIO_SECRET_KEY", minio.secret_key)
        env.setenv("MINIO_DEFAULT_BUCKET", bucket)
        try:
            yield bucket
        finally:
            system = config_manager.get_system_config()
            system.minio_endpoint = previous
            config_manager.set_system_config(system)


@contextmanager
def unreachable_object_store():
    """MINIO_ENDPOINT naming a port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
    with pytest.MonkeyPatch.context() as env:
        env.setenv("MINIO_ENDPOINT", dead)
        yield dead


def bucket_keys(minio, bucket: str) -> set:
    client = minio.boto3_client()
    keys: set = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        keys.update(entry["Key"] for entry in page.get("Contents", []))
    return keys


def seed_tenant_objects(tenant_id: str, workdir: Path) -> set:
    """Upload one video and two keyframes for ``tenant_id`` as the upload
    route and the ingestion pipeline do. Returns the keys written."""
    video = f"video of {tenant_id}".encode()
    frames = []
    for index in range(2):
        frame = workdir / f"{uuid.uuid4().hex}.jpg"
        frame.write_bytes(f"frame {index} of {tenant_id}".encode())
        frames.append(str(frame))
    uris = [
        minio_client.upload_bytes(video, tenant_id=tenant_id, filename="clip.mp4"),
        *minio_client.upload_keyframes(
            tenant_id=tenant_id, video_id="clip", keyframe_paths=frames
        ),
    ]
    return {uri.split("/", 3)[3] for uri in uris}
