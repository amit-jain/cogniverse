"""The Redis client behind state every runtime process shares.

Agent registrations, annotation requests and ``/ingestion/start`` job status
live in Redis so every worker process and replica serves the same ones. One
client per process carries them, with every command, connect and wait for a
pooled connection bounded, so a Redis that stops answering fails the request
instead of hanging it.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from redis.asyncio import BlockingConnectionPool, Redis
from redis.exceptions import RedisError

SHARED_STATE_REDIS_TIMEOUT_SECONDS = 5.0
SHARED_STATE_REDIS_MAX_CONNECTIONS = 64
_HEALTH_CHECK_INTERVAL_SECONDS = 30


def redacted_redis_url(redis_url: str) -> str:
    """``redis_url`` as scheme, host, port and database only.

    Credentials ride in the userinfo or a ``password`` query parameter, so
    both are dropped from anything logged or raised.
    """
    parts = urlsplit(redis_url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return f"{parts.scheme}://{netloc}{parts.path}"


class SharedStateUnavailableError(RuntimeError):
    """Raised when the shared-state Redis cannot be reached at startup."""


async def connect_shared_state_redis(
    redis_url: str,
    *,
    timeout_seconds: float = SHARED_STATE_REDIS_TIMEOUT_SECONDS,
    max_connections: int = SHARED_STATE_REDIS_MAX_CONNECTIONS,
) -> Redis:
    """Connect to ``redis_url`` and ping it before any request is served."""
    if not redis_url.strip():
        raise ValueError("redis_url must be non-empty")
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
    if max_connections < 1:
        raise ValueError(f"max_connections must be >= 1, got {max_connections}")
    client = Redis.from_pool(
        BlockingConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=max_connections,
            timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
            socket_keepalive=True,
            health_check_interval=_HEALTH_CHECK_INTERVAL_SECONDS,
        )
    )
    try:
        await client.ping()
    except RedisError as exc:
        await client.aclose()
        raise SharedStateUnavailableError(
            f"shared state Redis unavailable at {redacted_redis_url(redis_url)}"
        ) from exc
    return client
