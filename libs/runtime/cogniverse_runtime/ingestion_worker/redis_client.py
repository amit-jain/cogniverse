"""Redis connection management for the ingestion queue.

Wraps ``redis.asyncio`` with a lazy singleton keyed by URL so the
runtime, ingestor workers, and tests share one connection pool per
process. URL is read at startup boundaries (FastAPI lifespan, worker
``__main__``, test fixtures) and passed in explicitly — no env reads
inside the helpers.
"""

from __future__ import annotations

from typing import Optional

import redis.asyncio as aioredis

_pool: Optional[aioredis.Redis] = None
_pool_url: Optional[str] = None


async def get_redis(url: str) -> aioredis.Redis:
    """Return a process-wide ``redis.asyncio.Redis`` for ``url``.

    The pool is created lazily on first call and reused thereafter. If
    a different URL is requested, raises — switching URLs mid-process
    indicates a bug in the caller (env should be set once at startup).
    """
    global _pool, _pool_url
    if _pool is None:
        _pool = aioredis.from_url(url, decode_responses=True)
        _pool_url = url
        return _pool
    if _pool_url != url:
        raise RuntimeError(
            f"Redis pool already initialised with {_pool_url!r}; refusing "
            f"to switch to {url!r}. Reuse the existing pool or call "
            "close_redis() first."
        )
    return _pool


async def close_redis() -> None:
    """Close the pool — used by FastAPI shutdown and test teardown."""
    global _pool, _pool_url
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        _pool_url = None
