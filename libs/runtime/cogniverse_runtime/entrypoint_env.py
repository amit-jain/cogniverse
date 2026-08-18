"""Shared runtime entrypoint env resolution."""

from __future__ import annotations

import os


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
