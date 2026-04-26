"""Idempotency keys for ingestion submissions.

Same ``(source_url, profile, tenant_id)`` produces the same SHA, which
keys two records:
  - ``ingest:done:<sha>`` — the ``ingest_id`` of the completed run.
    Set by the worker on successful completion. TTL is configurable
    (default 7 days from chart values).
  - ``ingest:by_sha:<sha>`` — the ``ingest_id`` of the in-flight run
    (set by the submitter, deleted by the worker on terminal state).
    Lets a re-submission while a job is in flight return the same id
    rather than enqueue a duplicate.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import redis.asyncio as aioredis

DONE_KEY_PREFIX = "ingest:done:"
INFLIGHT_KEY_PREFIX = "ingest:by_sha:"


def compute_sha(source_url: str, profile: str, tenant_id: str) -> str:
    """16-hex-char SHA256 prefix of ``source_url + profile + tenant_id``.

    Truncating to 16 hex chars (64 bits) keeps Redis keys short while
    keeping collision probability negligible at the volumes in scope
    (≤ 10⁹ submissions = 1 in 5×10⁹ collision rate).
    """
    if not source_url or not profile or not tenant_id:
        raise ValueError(
            f"compute_sha requires non-empty source_url, profile, tenant_id; "
            f"got {source_url!r}, {profile!r}, {tenant_id!r}"
        )
    h = hashlib.sha256(f"{source_url}|{profile}|{tenant_id}".encode())
    return h.hexdigest()[:16]


async def get_existing_ingest_id(redis: aioredis.Redis, sha: str) -> Optional[str]:
    """Return the existing ingest_id for ``sha``, or None if not found.

    Checks the in-flight key first (job currently being processed),
    then the done key (job completed within the TTL window). Either is
    a valid reason to return the existing id rather than re-enqueue.
    """
    inflight = await redis.get(f"{INFLIGHT_KEY_PREFIX}{sha}")
    if inflight:
        return inflight
    done = await redis.get(f"{DONE_KEY_PREFIX}{sha}")
    return done


async def mark_inflight(redis: aioredis.Redis, sha: str, ingest_id: str) -> None:
    """Record that ``ingest_id`` is in flight for ``sha``. No TTL —
    the worker clears this on terminal state."""
    await redis.set(f"{INFLIGHT_KEY_PREFIX}{sha}", ingest_id)


async def clear_inflight(redis: aioredis.Redis, sha: str) -> None:
    """Worker calls this on terminal state (success or failure)."""
    await redis.delete(f"{INFLIGHT_KEY_PREFIX}{sha}")


async def mark_done(
    redis: aioredis.Redis, sha: str, ingest_id: str, ttl_seconds: int
) -> None:
    """Record successful completion. ``ttl_seconds=0`` keeps forever."""
    if ttl_seconds > 0:
        await redis.set(f"{DONE_KEY_PREFIX}{sha}", ingest_id, ex=ttl_seconds)
    else:
        await redis.set(f"{DONE_KEY_PREFIX}{sha}", ingest_id)


async def clear_done(redis: aioredis.Redis, sha: str) -> None:
    """Force re-ingest for ``sha`` — drops both the done and inflight
    records so the next submission re-enqueues."""
    await redis.delete(f"{DONE_KEY_PREFIX}{sha}", f"{INFLIGHT_KEY_PREFIX}{sha}")
