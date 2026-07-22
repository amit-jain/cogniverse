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


async def get_done_ingest_id(redis: aioredis.Redis, sha: str) -> Optional[str]:
    """Return the ingest_id of a COMPLETED run for ``sha``, or None.

    Unlike ``get_existing_ingest_id`` this ignores the in-flight key — the
    reaper uses it to distinguish "worker died before acking a finished job"
    (just settle the leftovers) from "worker died mid-processing" (re-drive
    the job), where the stale in-flight marker is present in both cases.
    """
    return await redis.get(f"{DONE_KEY_PREFIX}{sha}")


async def mark_inflight(
    redis: aioredis.Redis, sha: str, ingest_id: str, ttl_seconds: int
) -> None:
    """Record that ``ingest_id`` is in flight for ``sha``.

    The worker clears this on terminal state (the normal path). The TTL
    is a crash-recovery bound: if the worker is SIGKILLed / OOM-evicted
    between claiming the job and ``clear_inflight``, the key would
    otherwise persist forever and every future re-submission of the same
    ``(source_url, profile, tenant_id)`` would return the dead job's id
    and never re-enqueue. The TTL must exceed the longest realistic job
    duration so it never expires a live run; ``ttl_seconds=0`` disables
    it (persist forever — restores the old poisoning behaviour, for
    tests only).
    """
    if ttl_seconds > 0:
        await redis.set(f"{INFLIGHT_KEY_PREFIX}{sha}", ingest_id, ex=ttl_seconds)
    else:
        await redis.set(f"{INFLIGHT_KEY_PREFIX}{sha}", ingest_id)


async def claim_inflight(
    redis: aioredis.Redis, sha: str, ingest_id: str, ttl_seconds: int
) -> Optional[str]:
    """Atomically claim the in-flight slot for ``sha``.

    Returns None when this call WON the claim (``ingest_id`` recorded), or
    the ingest_id already holding the slot when a concurrent or earlier
    submission owns it — the caller returns that id instead of enqueuing a
    duplicate. SET NX makes check-and-claim one step; the separate
    get-then-mark it replaces had an await between check and write, so two
    identical submissions racing through the gap both enqueued (double
    ingest, and enough duplicates exhausted the tenant's concurrency
    budget and 429'd a different legitimate upload).
    """
    key = f"{INFLIGHT_KEY_PREFIX}{sha}"
    ex = ttl_seconds if ttl_seconds > 0 else None
    for _ in range(2):
        won = await redis.set(key, ingest_id, ex=ex, nx=True)
        if won:
            return None
        holder = await redis.get(key)
        if holder:
            return holder
        # The holder's key expired between our NX loss and the read —
        # retry the claim once; second failure falls through to claiming.
    await redis.set(key, ingest_id, ex=ex)
    return None


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
