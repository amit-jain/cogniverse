"""MinIO upload helper for the ``/ingestion/upload`` multipart path.

Multipart uploads land in MinIO under
``s3://{default_bucket}/{tenant_id}/{ingest_uuid}.{ext}``. The
ingestion queue then carries the resulting ``s3://`` URL — workers
fetch via ``MediaLocator`` which already speaks ``s3://`` against the
same MinIO endpoint.

Reading credentials at function-call time (not module-import time)
keeps the module loadable in test environments that don't have
MinIO env wired up, and matches the env-vars-only-at-startup pattern
used elsewhere in the runtime.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional


def _client():
    """Build a boto3 S3 client pointed at MinIO. boto3 is heavy to
    import; do it lazily so test paths that never upload don't pay
    the cost."""
    import boto3
    from botocore.client import Config

    endpoint = os.environ.get("MINIO_ENDPOINT")
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")
    if not (endpoint and access_key and secret_key):
        raise RuntimeError(
            "MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY must all be "
            "set for ingestion uploads. Enable minio in the chart values "
            "or set the env vars directly."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",  # MinIO ignores; boto3 requires *some* region.
    )


def _default_bucket() -> str:
    bucket = os.environ.get("MINIO_DEFAULT_BUCKET")
    if not bucket:
        raise RuntimeError(
            "MINIO_DEFAULT_BUCKET is not set; cannot upload without a target bucket."
        )
    return bucket


def upload_bytes(
    content: bytes,
    *,
    tenant_id: str,
    filename: Optional[str],
    content_type: Optional[str] = None,
    bucket: Optional[str] = None,
) -> str:
    """Upload ``content`` to MinIO under a tenant-scoped key, return s3:// URL.

    The object key is ``{tenant_id}/{uuid}.{ext}`` so collisions across
    tenants and across submissions of the same filename are
    impossible. ``filename`` is only used to derive the suffix.
    """
    bucket_name = bucket or _default_bucket()
    suffix = Path(filename).suffix if filename else ""
    key = f"{tenant_id}/{uuid.uuid4().hex}{suffix}"

    client = _client()
    extra: dict = {}
    if content_type:
        extra["ContentType"] = content_type
    client.put_object(Bucket=bucket_name, Key=key, Body=content, **extra)
    return f"s3://{bucket_name}/{key}"


def upload_keyframes(
    *,
    tenant_id: str,
    video_id: str,
    keyframe_paths: list,
    bucket: Optional[str] = None,
) -> list[str]:
    """Upload extracted keyframes to MinIO under the shared keyframe-key
    contract, so answer-time agents fetch them by deriving the same key from a
    search hit.

    ``keyframe_paths`` MUST be ordered by segment: the i-th path is uploaded
    under ``keyframe_object_key(tenant_id, video_id, i)`` — the same ``i`` the
    embedding step assigns as ``segment_id`` and the hit later carries. Returns
    the ``s3://`` URIs in that order.
    """
    from cogniverse_core.common.media import keyframe_object_key

    bucket_name = bucket or _default_bucket()
    client = _client()
    uris: list[str] = []
    for segment_id, path in enumerate(keyframe_paths):
        key = keyframe_object_key(tenant_id, video_id, segment_id)
        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=Path(path).read_bytes(),
            ContentType="image/jpeg",
        )
        uris.append(f"s3://{bucket_name}/{key}")
    return uris
