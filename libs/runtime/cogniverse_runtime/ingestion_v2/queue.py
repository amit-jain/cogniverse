"""Redis Streams-backed ingestion queue.

Producer-consumer split:
  submit() — runtime POST /ingest, XADDs to ``ingest:queue``.
  claim()  — ingestor worker XREADGROUP with BLOCK, exclusive ownership
             until ack() or until the pending entry is reclaimed by
             another consumer (handled outside this module).
  ack()    — worker on terminal state, XACKs the message id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

QUEUE_STREAM = "ingest:queue"
ACTIVE_KEY_PREFIX = "ingest:active:"


@dataclass(frozen=True)
class IngestJob:
    """One queue entry. ``message_id`` is the Redis Stream id; the
    worker passes it back to ack()."""

    message_id: str
    ingest_id: str
    source_url: str
    profile: str
    tenant_id: str
    sha: str


async def ensure_consumer_group(redis: aioredis.Redis, group: str) -> None:
    """Create the consumer group if it doesn't exist.

    ``MKSTREAM`` creates an empty stream if needed so the first call on
    a fresh deployment succeeds. ``BUSYGROUP`` is the
    already-exists error and is silently ignored.
    """
    try:
        await redis.xgroup_create(QUEUE_STREAM, group, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def submit(
    redis: aioredis.Redis,
    ingest_id: str,
    source_url: str,
    profile: str,
    tenant_id: str,
    sha: str,
) -> str:
    """XADD a new ingestion request. Returns the Redis message id."""
    fields = {
        "ingest_id": ingest_id,
        "source_url": source_url,
        "profile": profile,
        "tenant_id": tenant_id,
        "sha": sha,
    }
    return await redis.xadd(QUEUE_STREAM, fields)


async def claim(
    redis: aioredis.Redis,
    group: str,
    consumer_id: str,
    *,
    block_ms: int = 5000,
    count: int = 1,
) -> List[IngestJob]:
    """Block-pull the next ``count`` entries assigned to this consumer.

    Returns ``[]`` after ``block_ms`` if no work appears. Each returned
    entry is now exclusively owned by ``consumer_id`` until it's ACKed
    (or the pending entry is reclaimed by another consumer via XCLAIM,
    handled outside this module by a separate reaper).
    """
    response = await redis.xreadgroup(
        groupname=group,
        consumername=consumer_id,
        streams={QUEUE_STREAM: ">"},
        count=count,
        block=block_ms,
    )
    if not response:
        return []
    jobs: List[IngestJob] = []
    for _stream, entries in response:
        for message_id, fields in entries:
            jobs.append(
                IngestJob(
                    message_id=message_id,
                    ingest_id=fields["ingest_id"],
                    source_url=fields["source_url"],
                    profile=fields["profile"],
                    tenant_id=fields["tenant_id"],
                    sha=fields["sha"],
                )
            )
    return jobs


async def ack(redis: aioredis.Redis, group: str, message_id: str) -> int:
    """XACK the message. Returns 1 if the message was pending, 0
    otherwise (already acked or never delivered to this group)."""
    return await redis.xack(QUEUE_STREAM, group, message_id)


async def queue_depth(redis: aioredis.Redis) -> int:
    """XLEN the queue stream — used by backpressure to decide 429."""
    return await redis.xlen(QUEUE_STREAM)


async def increment_active(redis: aioredis.Redis, tenant_id: str) -> int:
    """INCR the per-tenant active counter and return the new value."""
    return await redis.incr(f"{ACTIVE_KEY_PREFIX}{tenant_id}")


async def decrement_active(redis: aioredis.Redis, tenant_id: str) -> int:
    """DECR the per-tenant active counter, clamped at 0.

    Worker calls this on terminal state. Use a Lua-like floor at zero
    so a stray double-decrement (e.g. duplicate terminal-event flush)
    can't leave the counter negative and underflow the backpressure
    check.
    """
    new_val = await redis.decr(f"{ACTIVE_KEY_PREFIX}{tenant_id}")
    if new_val < 0:
        await redis.set(f"{ACTIVE_KEY_PREFIX}{tenant_id}", 0)
        return 0
    return new_val


async def get_active(redis: aioredis.Redis, tenant_id: str) -> int:
    """Read the current per-tenant active counter."""
    raw = await redis.get(f"{ACTIVE_KEY_PREFIX}{tenant_id}")
    return int(raw) if raw else 0


# ---------------------------------------------------------------------------
# Status streams (per-job event log)
# ---------------------------------------------------------------------------


def _status_stream_key(ingest_id: str) -> str:
    return f"ingest:status:{ingest_id}"


async def publish_status(redis: aioredis.Redis, ingest_id: str, event: dict) -> str:
    """Worker call: append one status event to the job's stream.

    Events are JSON-serialised into a single ``data`` field so any
    nested structure (progress dicts, artifact lists) round-trips
    without being flattened into stream fields.
    """
    return await redis.xadd(_status_stream_key(ingest_id), {"data": json.dumps(event)})


async def read_status_since(
    redis: aioredis.Redis,
    ingest_id: str,
    *,
    last_id: str = "0-0",
    block_ms: Optional[int] = None,
    count: int = 100,
) -> List[tuple]:
    """SSE call: read events newer than ``last_id``.

    Returns ``[(message_id, parsed_event), ...]`` ordered oldest first.
    Pass ``block_ms=None`` for non-blocking; pass an int to long-poll
    for new events. Caller advances ``last_id`` to the last seen
    message_id between calls.
    """
    if block_ms is not None:
        response = await redis.xread(
            streams={_status_stream_key(ingest_id): last_id},
            count=count,
            block=block_ms,
        )
    else:
        response = await redis.xrange(
            _status_stream_key(ingest_id),
            min=f"({last_id}" if last_id != "0-0" else "-",
            max="+",
            count=count,
        )
        return [(mid, json.loads(fields["data"])) for mid, fields in response]

    if not response:
        return []
    out: List[tuple] = []
    for _stream, entries in response:
        for message_id, fields in entries:
            out.append((message_id, json.loads(fields["data"])))
    return out
