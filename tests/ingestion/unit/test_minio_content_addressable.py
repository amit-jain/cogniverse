"""Uploaded objects are keyed by content hash, not a random uuid.

The /upload path computes its idempotency sha from the s3:// URL it writes. A
uuid4 key made every upload of the same bytes land at a different URL, so the
idempotency sha was always fresh — re-uploading an identical file re-ran the
whole pipeline and doubled the index. Content-addressable keys make identical
bytes map to one object (and one idempotency sha), so a re-upload dedupes.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.ingestion_worker import minio_client

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def captured_puts(monkeypatch):
    puts = []
    fake = MagicMock()
    fake.put_object = lambda **kw: puts.append(kw)
    monkeypatch.setattr(minio_client, "_client", lambda: fake)
    monkeypatch.setattr(minio_client, "_default_bucket", lambda: "media")
    return puts


def test_identical_bytes_map_to_one_key(captured_puts):
    content = b"the same video bytes"
    url1 = minio_client.upload_bytes(
        content, tenant_id="acme:acme", filename="clip.mp4"
    )
    url2 = minio_client.upload_bytes(
        content, tenant_id="acme:acme", filename="clip.mp4"
    )

    assert url1 == url2, "identical bytes must resolve to the same s3:// URL"
    digest = hashlib.sha256(content).hexdigest()
    assert url1 == f"s3://media/acme:acme/{digest}.mp4"
    # Both writes target the same object key (idempotent overwrite).
    assert {p["Key"] for p in captured_puts} == {f"acme:acme/{digest}.mp4"}


def test_different_bytes_map_to_different_keys(captured_puts):
    url_a = minio_client.upload_bytes(b"aaa", tenant_id="acme:acme", filename="v.mp4")
    url_b = minio_client.upload_bytes(b"bbb", tenant_id="acme:acme", filename="v.mp4")
    assert url_a != url_b, "different bytes must resolve to different keys"


def test_key_is_tenant_scoped_and_keeps_suffix(captured_puts):
    url = minio_client.upload_bytes(
        b"x", tenant_id="t1:t1", filename="movie.MOV", content_type="video/quicktime"
    )
    assert url.startswith("s3://media/t1:t1/")
    assert url.endswith(".MOV")
    assert captured_puts[0]["ContentType"] == "video/quicktime"
