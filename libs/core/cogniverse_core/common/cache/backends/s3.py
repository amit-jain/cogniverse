"""S3 / MinIO cache backend — a shared, durable L2 tier for the pipeline cache.

The structured-filesystem backend is per-pod-ephemeral: cache hits only happen
when the same worker pod re-processes the same video, and a pod restart wipes
it. This backend stores artifacts in an S3-compatible bucket (the same MinIO
deployment used for media uploads, different bucket/prefix), so a re-ingest of
the same video hits cache regardless of which pod handled the first pass.

boto3 is synchronous; every network call is wrapped in ``asyncio.to_thread`` so
the cache stays drop-in compatible with the async ``CacheBackend`` contract.
"""

import asyncio
import json
import logging
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import msgpack

from ..base import CacheBackend

logger = logging.getLogger(__name__)

# Single S3 user-metadata key holding the cache envelope (format + expiry).
# One JSON blob avoids per-field header-name pitfalls (underscores, casing).
_META_KEY = "cache-meta"


@dataclass
class S3CacheBackendConfig:
    """Configuration for the S3 cache backend.

    Pure data: credential fields default to ``None`` and are resolved from the
    ``MINIO_*`` environment at client-build time (mirroring
    ``ingestion_worker.minio_client``), with config values taking precedence.
    """

    backend_type: str = "s3"
    endpoint: Optional[str] = None  # falls back to MINIO_ENDPOINT
    access_key: Optional[str] = None  # falls back to MINIO_ACCESS_KEY
    secret_key: Optional[str] = None  # falls back to MINIO_SECRET_KEY
    bucket: str = "cogniverse-pipeline-cache"
    key_prefix: str = "pipeline/"
    region: str = "us-east-1"
    serialization_format: str = "pickle"  # or "json", "msgpack"
    enabled: bool = True
    priority: int = 1
    enable_ttl: bool = True
    lifecycle_expiration_days: Optional[int] = None  # bucket ILM backstop


class S3CacheBackend(CacheBackend):
    """Cache backend backed by an S3-compatible object store (MinIO/AWS S3)."""

    CONFIG_CLASS = S3CacheBackendConfig

    def __init__(self, config: S3CacheBackendConfig):
        self.config = config
        self.bucket = config.bucket
        self.key_prefix = config.key_prefix
        self.format = config.serialization_format
        self._client = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }

    # -- client / key helpers ------------------------------------------------

    def _s3(self):
        """Lazily build the boto3 client and ensure the bucket exists.

        Credentials come from config first, then ``MINIO_*`` env. boto3 is
        heavy to import, so the cost is paid only when the cache is actually
        used (not at config-load time).
        """
        if self._client is not None:
            return self._client

        import os

        import boto3
        from botocore.client import Config

        endpoint = self.config.endpoint or os.environ.get("MINIO_ENDPOINT")
        access_key = self.config.access_key or os.environ.get("MINIO_ACCESS_KEY")
        secret_key = self.config.secret_key or os.environ.get("MINIO_SECRET_KEY")
        if not (endpoint and access_key and secret_key):
            raise RuntimeError(
                "S3 cache backend needs endpoint/access_key/secret_key via "
                "config or the MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY "
                "environment variables."
            )

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name=self.config.region,
        )
        self._ensure_bucket(client)
        self._client = client
        return self._client

    def _ensure_bucket(self, client) -> None:
        from botocore.exceptions import ClientError

        try:
            client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                client.create_bucket(Bucket=self.bucket)
            except ClientError as e:
                logger.warning("Could not create cache bucket %s: %s", self.bucket, e)
        self._apply_lifecycle(client)

    def _apply_lifecycle(self, client) -> None:
        """Bound bucket growth server-side: expire cache objects after N days.

        A backstop independent of per-object TTL/cleanup — MinIO/S3 ILM
        deletes expired objects whether or not they are ever read again.
        """
        days = self.config.lifecycle_expiration_days
        if not days:
            return
        from botocore.exceptions import ClientError

        try:
            client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "cogniverse-cache-expiry",
                            "Filter": {"Prefix": self.key_prefix},
                            "Status": "Enabled",
                            "Expiration": {"Days": days},
                        }
                    ]
                },
            )
        except ClientError as e:
            logger.warning(
                "Could not set lifecycle on cache bucket %s: %s", self.bucket, e
            )

    def _s3_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    @staticmethod
    def _is_not_found(exc) -> bool:
        from botocore.exceptions import ClientError

        if not isinstance(exc, ClientError):
            return False
        code = exc.response.get("Error", {}).get("Code")
        return code in ("NoSuchKey", "NoSuchBucket", "404", "NotFound")

    # -- serialization (mirrors structured_filesystem) -----------------------

    def _serialize(self, data: Any) -> bytes:
        if self.format == "pickle":
            return pickle.dumps(data)
        if self.format == "json":
            return json.dumps(data).encode("utf-8")
        if self.format == "msgpack":
            return msgpack.packb(data)
        raise ValueError(f"Unknown serialization format: {self.format}")

    def _deserialize(self, data: bytes, fmt: str) -> Any:
        if fmt == "pickle":
            return pickle.loads(data)
        if fmt == "json":
            return json.loads(data.decode("utf-8"))
        if fmt == "msgpack":
            return msgpack.unpackb(data)
        raise ValueError(f"Unknown serialization format: {fmt}")

    # -- sync workers (run inside asyncio.to_thread) -------------------------

    def _put(self, s3_key: str, body: bytes, meta: Dict[str, str]) -> None:
        self._s3().put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=body,
            Metadata={_META_KEY: json.dumps(meta)},
        )

    def _get_object(self, s3_key: str) -> Tuple[Dict[str, Any], bytes]:
        resp = self._s3().get_object(Bucket=self.bucket, Key=s3_key)
        body = resp["Body"].read()
        envelope = json.loads(resp.get("Metadata", {}).get(_META_KEY, "{}"))
        return envelope, body

    def _head(self, s3_key: str) -> Dict[str, Any]:
        resp = self._s3().head_object(Bucket=self.bucket, Key=s3_key)
        return json.loads(resp.get("Metadata", {}).get(_META_KEY, "{}"))

    def _list(self, prefix: str) -> List[Dict[str, Any]]:
        client = self._s3()
        paginator = client.get_paginator("list_objects_v2")
        objects: List[Dict[str, Any]] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects.extend(page.get("Contents", []) or [])
        return objects

    def _delete_keys(self, keys: List[str]) -> int:
        if not keys:
            return 0
        client = self._s3()
        deleted = 0
        for i in range(0, len(keys), 1000):  # S3 delete_objects caps at 1000
            batch = keys[i : i + 1000]
            resp = client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
            deleted += len(batch) - len(resp.get("Errors", []) or [])
        return deleted

    @staticmethod
    def _expired(envelope: Dict[str, Any]) -> bool:
        expires_at = envelope.get("expires_at")
        return expires_at is not None and time.time() > expires_at

    # -- CacheBackend interface ----------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        s3_key = self._s3_key(key)
        try:
            envelope, body = await asyncio.to_thread(self._get_object, s3_key)
        except Exception as e:
            if self._is_not_found(e):
                self._stats["misses"] += 1
                return None
            logger.error("Error reading cache object %s: %s", s3_key, e)
            self._stats["misses"] += 1
            return None

        if self.config.enable_ttl and self._expired(envelope):
            await self.delete(key)
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1
        fmt = envelope.get("format", self.format)
        if fmt == "raw":
            return body
        return self._deserialize(body, fmt)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        s3_key = self._s3_key(key)
        if isinstance(value, bytes):
            body = value
            envelope: Dict[str, Any] = {"format": "raw"}
        else:
            body = self._serialize(value)
            envelope = {"format": self.format}
        if ttl is not None and ttl > 0:
            envelope["expires_at"] = time.time() + ttl
            envelope["ttl"] = ttl

        try:
            await asyncio.to_thread(self._put, s3_key, body, envelope)
            self._stats["sets"] += 1
            return True
        except Exception as e:
            logger.error("Error writing cache object %s: %s", s3_key, e)
            return False

    async def delete(self, key: str) -> bool:
        s3_key = self._s3_key(key)
        try:
            existed = await self.exists(key)
            await asyncio.to_thread(
                lambda: self._s3().delete_object(Bucket=self.bucket, Key=s3_key)
            )
            if existed:
                self._stats["deletes"] += 1
            return existed
        except Exception as e:
            logger.error("Error deleting cache object %s: %s", s3_key, e)
            return False

    async def exists(self, key: str) -> bool:
        s3_key = self._s3_key(key)
        try:
            envelope = await asyncio.to_thread(self._head, s3_key)
        except Exception as e:
            if self._is_not_found(e):
                return False
            logger.error("Error heading cache object %s: %s", s3_key, e)
            return False
        if self.config.enable_ttl and self._expired(envelope):
            return False
        return True

    async def clear(self, pattern: Optional[str] = None) -> int:
        if pattern and pattern.endswith("*"):
            prefix = f"{self.key_prefix}{pattern[:-1]}"
        else:
            prefix = self.key_prefix
        try:
            objects = await asyncio.to_thread(self._list, prefix)
            keys = [o["Key"] for o in objects]
            return await asyncio.to_thread(self._delete_keys, keys)
        except Exception as e:
            logger.error("Error clearing cache prefix %s: %s", prefix, e)
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        try:
            objects = await asyncio.to_thread(self._list, self.key_prefix)
            stats["total_files"] = len(objects)
            stats["size_bytes"] = sum(o.get("Size", 0) for o in objects)
        except Exception as e:
            logger.warning("Error computing S3 cache stats: %s", e)
        return stats

    async def cleanup_expired(self) -> int:
        try:
            objects = await asyncio.to_thread(self._list, self.key_prefix)
        except Exception as e:
            logger.error("Error listing for cleanup: %s", e)
            return 0

        expired_keys: List[str] = []
        for obj in objects:
            try:
                envelope = await asyncio.to_thread(self._head, obj["Key"])
                if self._expired(envelope):
                    expired_keys.append(obj["Key"])
            except Exception:
                continue

        removed = await asyncio.to_thread(self._delete_keys, expired_keys)
        self._stats["evictions"] += removed
        return removed


# Register the backend
from ..registry import CacheBackendRegistry  # noqa: E402

CacheBackendRegistry.register("s3", S3CacheBackend)
