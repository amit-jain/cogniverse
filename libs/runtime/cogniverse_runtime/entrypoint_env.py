"""Shared runtime entrypoint env resolution."""

from __future__ import annotations

import os

from cogniverse_core.common.cache.backends.s3 import configure_s3_backend_defaults


def _resolve_tenant_cache_capacity() -> int:
    try:
        return max(
            1,
            int(os.environ.get("COGNIVERSE_TENANT_CACHE_CAPACITY", 16)),
        )
    except (TypeError, ValueError):
        return 16


def resolve_library_env_defaults() -> dict[str, str | int | None]:
    """Read the process env values used by runtime entrypoints exactly once."""
    return {
        "minio_endpoint": os.environ.get("MINIO_ENDPOINT"),
        "minio_access_key": os.environ.get("MINIO_ACCESS_KEY"),
        "minio_secret_key": os.environ.get("MINIO_SECRET_KEY"),
        "telemetry_otlp_endpoint": os.environ.get("TELEMETRY_OTLP_ENDPOINT"),
        "telemetry_http_endpoint": os.environ.get("TELEMETRY_HTTP_ENDPOINT"),
        "semantic_embed_url": os.environ.get("COGNIVERSE_SEMANTIC_EMBED_URL"),
        "semantic_embed_model": os.environ.get("COGNIVERSE_SEMANTIC_EMBED_MODEL"),
        "tenant_cache_capacity": _resolve_tenant_cache_capacity(),
    }


def configure_runtime_library_defaults(
    runtime_defaults: dict[str, str | int | None],
) -> None:
    """Apply the resolved MinIO defaults to AWS and S3 cache state."""
    minio_access_key = runtime_defaults["minio_access_key"]
    minio_secret_key = runtime_defaults["minio_secret_key"]
    if minio_access_key:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", minio_access_key)
    if minio_secret_key:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", minio_secret_key)
    configure_s3_backend_defaults(
        endpoint=runtime_defaults["minio_endpoint"],
        access_key=minio_access_key,
        secret_key=minio_secret_key,
    )
