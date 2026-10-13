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


@pytest.fixture
def stubbed_store(monkeypatch):
    """A real boto3 S3 client whose answers a ``Stubber`` supplies; botocore
    validates each request and answer against the S3 service model."""
    import boto3
    from botocore.stub import Stubber

    client = boto3.client(
        "s3",
        endpoint_url="http://minio.invalid:9000",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        region_name="us-east-1",
    )
    built_with: list[dict] = []

    def build(**config):
        built_with.append(config)
        return client

    monkeypatch.setattr(minio_client, "_client", build)
    monkeypatch.setattr(minio_client, "_default_bucket", lambda: "media")
    with Stubber(client) as stubber:
        yield stubber, built_with
        stubber.assert_no_pending_responses()


def test_delete_tenant_objects_reports_each_object_the_store_refuses(stubbed_store):
    stubber, built_with = stubbed_store
    keys = ["acme:acme/a.mp4", "acme:acme/keyframes/v/0000.jpg", "acme:acme/z.mp4"]
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": key, "Size": 1} for key in keys], "IsTruncated": False},
        {"Bucket": "media", "Prefix": "acme:acme/"},
    )
    stubber.add_response(
        "delete_objects",
        {
            "Errors": [
                {"Key": keys[1], "Code": "AccessDenied", "Message": "Access Denied."}
            ]
        },
        {
            "Bucket": "media",
            "Delete": {"Objects": [{"Key": key} for key in keys], "Quiet": True},
        },
    )

    removed, refused = minio_client.delete_tenant_objects("acme:acme")

    assert (removed, refused) == (2, {keys[1]: "AccessDenied: Access Denied."})
    assert built_with == [
        {
            "connect_timeout": 5,
            "read_timeout": 30,
            "retries": {"max_attempts": 2, "mode": "standard"},
        }
    ]


def test_delete_tenant_objects_with_nothing_stored_sends_no_delete(stubbed_store):
    stubber, _ = stubbed_store
    stubber.add_response(
        "list_objects_v2",
        {"IsTruncated": False},
        {"Bucket": "media", "Prefix": "acme:acme/"},
    )

    assert minio_client.delete_tenant_objects("acme:acme") == (0, {})


def test_delete_tenant_objects_raises_when_the_listing_fails(stubbed_store):
    stubber, _ = stubbed_store
    stubber.add_client_error(
        "list_objects_v2", service_error_code="SlowDown", http_status_code=503
    )

    with pytest.raises(ClientError) as failure:
        minio_client.delete_tenant_objects("acme:acme")

    assert failure.value.response["Error"]["Code"] == "SlowDown"


@pytest.mark.parametrize("tenant_id", ["", "acme:acme/keyframes", "/"])
def test_delete_tenant_objects_refuses_an_id_that_names_no_tenant_prefix(
    stubbed_store, tenant_id
):
    with pytest.raises(ValueError) as refused:
        minio_client.delete_tenant_objects(tenant_id)

    assert str(refused.value) == f"{tenant_id!r} names no tenant prefix"
