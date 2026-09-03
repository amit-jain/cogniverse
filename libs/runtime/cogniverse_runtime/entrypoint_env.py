"""Shared runtime entrypoint env resolution."""

from __future__ import annotations

import os

from cogniverse_agents._rlm_promotion import configure_rlm_promotion
from cogniverse_agents.inference.deno_check import configure_deno_check
from cogniverse_core.common.cache.backends.s3 import configure_s3_backend_defaults

_RLM_PROMOTION_DEFAULT_FRACTION = 0.75


def _resolve_rlm_promotion_fraction() -> float:
    try:
        return float(
            os.environ.get(
                "COGNIVERSE_ORCH_RLM_PROMOTION_FRACTION",
                _RLM_PROMOTION_DEFAULT_FRACTION,
            )
        )
    except (TypeError, ValueError):
        return _RLM_PROMOTION_DEFAULT_FRACTION


def _resolve_tenant_cache_capacity() -> int:
    try:
        return max(
            1,
            int(os.environ.get("COGNIVERSE_TENANT_CACHE_CAPACITY", 16)),
        )
    except (TypeError, ValueError):
        return 16


def resolve_library_env_defaults() -> dict[str, str | int | float | bool | None]:
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
        "rlm_promotion_enabled": (
            os.environ.get("COGNIVERSE_ORCH_RLM_PROMOTION", "").lower() != "disabled"
        ),
        "rlm_promotion_fraction": _resolve_rlm_promotion_fraction(),
        "rlm_skip_deno_check": (
            os.environ.get("COGNIVERSE_RLM_SKIP_DENO_CHECK", "").lower()
            in {"1", "true", "yes"}
        ),
    }


def configure_runtime_library_defaults(
    runtime_defaults: dict[str, str | int | float | bool | None],
) -> None:
    """Apply the resolved env defaults to library module state."""
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
    configure_rlm_promotion(
        enabled=runtime_defaults["rlm_promotion_enabled"],
        fraction=runtime_defaults["rlm_promotion_fraction"],
    )
    configure_deno_check(skip=runtime_defaults["rlm_skip_deno_check"])
