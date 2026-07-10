"""Unit tests for the S3 cache backend against an in-memory fake boto3 client.

The fake models S3's contract (string-only user metadata, ``ClientError`` /
``NoSuchKey`` on a missing object, ``delete_objects`` batch) so the assertions
exercise the backend's real serialization, key-shaping, and TTL logic — not a
self-confirming mock.
"""

import json
import time

import pytest
from botocore.exceptions import ClientError

from cogniverse_core.common.cache.backends.s3 import (
    _META_KEY,
    S3CacheBackend,
    S3CacheBackendConfig,
)
from cogniverse_core.common.cache.base import CacheBackend
from cogniverse_core.common.cache.registry import CacheBackendRegistry


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _not_found(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "NoSuchKey"}}, op)


class _Paginator:
    def __init__(self, objects):
        self._objects = objects

    def paginate(self, Bucket, Prefix=""):
        contents = [
            {"Key": k, "Size": len(body)}
            for k, (body, _md) in self._objects.items()
            if k.startswith(Prefix)
        ]
        yield {"Contents": contents}


class FakeS3:
    """Minimal in-memory S3 modelling only what the backend calls."""

    def __init__(self):
        self.objects = {}  # key -> (body_bytes, metadata_dict)
        self.put_calls = []

    def head_bucket(self, Bucket):
        return {}

    def put_object(self, Bucket, Key, Body, Metadata=None):
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "Metadata": Metadata})
        self.objects[Key] = (Body, Metadata or {})

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _not_found("GetObject")
        body, md = self.objects[Key]
        return {"Body": _Body(body), "Metadata": md}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _not_found("HeadObject")
        _body, md = self.objects[Key]
        return {"Metadata": md}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        return _Paginator(self.objects)

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
        return {}


@pytest.fixture
def fake():
    return FakeS3()


def _backend(fake, fmt="pickle"):
    backend = S3CacheBackend(
        S3CacheBackendConfig(
            bucket="test-bucket", key_prefix="pipeline/", serialization_format=fmt
        )
    )
    backend._client = fake  # bypass real boto3 / env resolution
    return backend


@pytest.mark.unit
@pytest.mark.ci_fast
async def test_dict_round_trips_exact_value(fake):
    backend = _backend(fake)
    payload = {"segments": [{"text": "hi", "start": 0.0}]}

    assert await backend.set("p:video:abc:transcript", payload) is True
    assert await backend.get("p:video:abc:transcript") == payload


@pytest.mark.unit
@pytest.mark.ci_fast
async def test_image_bytes_round_trip_not_deserialized(fake):
    backend = _backend(fake)
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"

    await backend.set("p:video:abc:keyframes:strategy=fps:frame_5", jpeg)
    out = await backend.get("p:video:abc:keyframes:strategy=fps:frame_5")

    assert isinstance(out, bytes)
    assert out == jpeg
    # envelope records raw so get() does not attempt to deserialize
    _body, md = fake.objects["pipeline/p:video:abc:keyframes:strategy=fps:frame_5"]
    assert json.loads(md[_META_KEY])["format"] == "raw"


async def test_put_uses_exact_prefixed_key_and_bucket(fake):
    backend = _backend(fake)
    await backend.set("p:video:abc:transcript", {"x": 1})

    assert len(fake.put_calls) == 1
    assert fake.put_calls[0]["Key"] == "pipeline/p:video:abc:transcript"
    assert fake.put_calls[0]["Bucket"] == "test-bucket"


@pytest.mark.unit
@pytest.mark.ci_fast
async def test_expired_entry_is_a_miss(fake):
    backend = _backend(fake)
    await backend.set("p:video:abc:transcript", {"x": 1}, ttl=1)

    # force expiry by rewriting the stored envelope into the past
    key = "pipeline/p:video:abc:transcript"
    body, md = fake.objects[key]
    env = json.loads(md[_META_KEY])
    env["expires_at"] = time.time() - 10
    fake.objects[key] = (body, {_META_KEY: json.dumps(env)})

    assert await backend.get("p:video:abc:transcript") is None
    assert await backend.exists("p:video:abc:transcript") is False
    # the expired object was deleted on read
    assert key not in fake.objects


async def test_clear_pattern_removes_matching_only(fake):
    backend = _backend(fake)
    await backend.set("p:video:abc:transcript", {"a": 1})
    await backend.set("p:video:abc:descriptions", {"b": 2})
    await backend.set("p:video:zzz:transcript", {"c": 3})

    removed = await backend.clear("p:video:abc:*")

    assert removed == 2
    assert await backend.get("p:video:zzz:transcript") == {"c": 3}


async def test_stats_counters_match_op_sequence(fake):
    backend = _backend(fake)
    await backend.set("p:video:abc:transcript", {"a": 1})  # sets=1
    await backend.get("p:video:abc:transcript")  # hits=1
    await backend.get("p:video:abc:missing")  # misses=1
    await backend.delete("p:video:abc:transcript")  # deletes=1

    stats = await backend.get_stats()
    assert stats["sets"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["deletes"] == 1


@pytest.mark.unit
@pytest.mark.ci_fast
def test_registry_creates_s3_via_config_class():
    backend = CacheBackendRegistry.create(
        {
            "backend_type": "s3",
            "bucket": "b",
            "key_prefix": "px/",
            "priority": 1,
            "enabled": True,
            "default_ttl": 0,  # shared/extra key must be filtered, not crash
        }
    )
    assert isinstance(backend, S3CacheBackend)
    assert isinstance(backend.config, S3CacheBackendConfig)
    assert backend.config.bucket == "b"
    assert backend.config.key_prefix == "px/"


def test_registry_still_creates_filesystem(tmp_path):
    from cogniverse_core.common.cache.backends.structured_filesystem import (
        StructuredFilesystemBackend,
        StructuredFilesystemConfig,
    )

    backend = CacheBackendRegistry.create(
        {
            "backend_type": "structured_filesystem",
            "base_path": str(tmp_path / "c"),
            "priority": 0,
        }
    )
    assert isinstance(backend, StructuredFilesystemBackend)
    assert isinstance(backend.config, StructuredFilesystemConfig)


def test_registry_rejects_backend_without_config_class():
    class _Bare(CacheBackend):
        async def get(self, key):
            return None

        async def set(self, key, value, ttl=None):
            return True

        async def delete(self, key):
            return True

        async def exists(self, key):
            return False

        async def clear(self, pattern=None):
            return 0

        async def get_stats(self):
            return {}

    CacheBackendRegistry.register("bare_no_config", _Bare)
    try:
        with pytest.raises(ValueError, match="CONFIG_CLASS"):
            CacheBackendRegistry.create({"backend_type": "bare_no_config"})
    finally:
        CacheBackendRegistry._backends.pop("bare_no_config", None)
