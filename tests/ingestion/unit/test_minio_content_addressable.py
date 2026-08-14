"""Uploaded objects are keyed by content hash, not a random uuid.

The /upload path computes its idempotency sha from the s3:// URL it writes. A
uuid4 key made every upload of the same bytes land at a different URL, so the
idempotency sha was always fresh — re-uploading an identical file re-ran the
whole pipeline and doubled the index. Content-addressable keys make identical
bytes map to one object (and one idempotency sha), so a re-upload dedupes.
The upload basename is preserved in object metadata so the ingestion worker can
restore it into schema titles.
"""

from __future__ import annotations

import hashlib

import pytest
from botocore.exceptions import ClientError

from cogniverse_runtime.ingestion_worker import minio_client

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def fake_client(monkeypatch):
    class _FakeS3Client:
        def __init__(self):
            self.objects: dict[tuple[str, str], dict] = {}
            self.put_calls: list[dict] = []
            self.head_calls: list[tuple[str, str]] = []

        def head_object(self, Bucket, Key):
            self.head_calls.append((Bucket, Key))
            stored = self.objects.get((Bucket, Key))
            if stored is None:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "NoSuchKey",
                            "Message": "The specified key does not exist.",
                        }
                    },
                    "HeadObject",
                )
            return stored

        def put_object(self, **kw):
            self.put_calls.append(kw)
            self.objects[(kw["Bucket"], kw["Key"])] = kw
            return {"ETag": '"fake"'}

    fake = _FakeS3Client()
    monkeypatch.setattr(minio_client, "_client", lambda: fake)
    monkeypatch.setattr(minio_client, "_default_bucket", lambda: "media")
    return fake


def test_identical_bytes_dedup_preserves_first_filename(fake_client):
    content = b"the same video bytes"
    url1 = minio_client.upload_bytes(
        content, tenant_id="acme:acme", filename="first_name.mp4"
    )
    url2 = minio_client.upload_bytes(
        content, tenant_id="acme:acme", filename="second_name.mp4"
    )

    assert url1 == url2, "identical bytes must resolve to the same s3:// URL"
    digest = hashlib.sha256(content).hexdigest()
    assert url1 == f"s3://media/acme:acme/{digest}.mp4"
    assert len(fake_client.put_calls) == 1
    stored = fake_client.objects[("media", f"acme:acme/{digest}.mp4")]
    assert stored["Metadata"]["original_filename"] == "first_name.mp4"


def test_different_bytes_map_to_different_keys(fake_client):
    url_a = minio_client.upload_bytes(b"aaa", tenant_id="acme:acme", filename="v.mp4")
    url_b = minio_client.upload_bytes(b"bbb", tenant_id="acme:acme", filename="v.mp4")
    assert url_a != url_b, "different bytes must resolve to different keys"
    assert len(fake_client.put_calls) == 2


def test_key_is_tenant_scoped_and_keeps_suffix(fake_client):
    url = minio_client.upload_bytes(
        b"x", tenant_id="t1:t1", filename="movie.MOV", content_type="video/quicktime"
    )
    assert url.startswith("s3://media/t1:t1/")
    assert url.endswith(".MOV")
    assert fake_client.put_calls[0]["ContentType"] == "video/quicktime"
    assert fake_client.put_calls[0]["Metadata"]["original_filename"] == "movie.MOV"


def test_missing_filename_uploads_without_metadata(fake_client):
    content = b"no filename bytes"
    url = minio_client.upload_bytes(content, tenant_id="t1:t1", filename=None)

    digest = hashlib.sha256(content).hexdigest()
    assert url == f"s3://media/t1:t1/{digest}"
    assert "Metadata" not in fake_client.put_calls[0]
    assert minio_client.get_original_filename(url) == ""
