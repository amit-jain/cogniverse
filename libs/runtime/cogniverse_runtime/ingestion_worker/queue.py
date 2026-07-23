"""Redis Streams-backed ingestion queue.

Producer-consumer split:
  submit() — runtime POST /ingest, XADDs to ``ingest:queue``.
  claim()  — ingestor worker XREADGROUP with BLOCK, exclusive ownership
             until ack() or until the pending entry is reclaimed by
             another consumer (handled outside this module).
  ack()    — worker on terminal state, XACKs then XDELs the message id
             so completed jobs leave the stream (XLEN = live backlog).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` on a malformed value.

    A bad value (e.g. a typo'd number) would otherwise raise at import and
    crash-loop the worker instead of degrading to the default with a warning.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


QUEUE_STREAM = "ingest:queue"
ACTIVE_KEY_PREFIX = "ingest:active:"

# Self-healing TTL on the per-tenant active counter. The counter is decremented
# in the worker's terminal cleanup; a job that's incremented but never reaches
# cleanup (e.g. a poison message redelivered forever) would leak it and wedge
# backpressure. Each increment refreshes the TTL, so a continuously-ingesting
# tenant keeps the key alive while a stale leaked counter expires on its own.
# Must exceed the longest expected ingestion job.
ACTIVE_TTL_SECONDS = _int_env("INGEST_ACTIVE_TTL_SECONDS", 3600)

# Per-job status stream bounds. Without them each ingest left an immortal
# ``ingest:status:<id>`` stream in Redis: no length cap (a chatty job grew it
# unbounded) and no TTL (it lived forever after the job finished). The cap
# keeps the most recent events; the sliding TTL reclaims a finished job's
# stream while a still-active job keeps refreshing it.
STATUS_STREAM_MAXLEN = _int_env("INGEST_STATUS_STREAM_MAXLEN", 1000)
STATUS_STREAM_TTL_SECONDS = _int_env("INGEST_STATUS_STREAM_TTL_SECONDS", 6 * 60 * 60)


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
            job = _parse_entry(message_id, fields)
            if job is None:
                await settle_malformed_entry(redis, fields)
                await ack(redis, group, message_id)
                continue
            jobs.append(job)
    return jobs


async def settle_malformed_entry(redis: aioredis.Redis, fields: dict) -> None:
    """Best-effort resolve a client whose submission produced a malformed entry.

    A version-skew entry can still carry ``ingest_id`` / ``sha`` even when a
    newer required field is missing. Publish a failed terminal so a client
    polling status stops waiting for a job that will never run, and release the
    inflight idempotency marker so a resubmit is not blocked for the 6h TTL.
    The active counter is intentionally left alone (it self-heals via its TTL,
    and decrementing here could free a slot a different running job holds).
    Every step is best-effort; missing fields are skipped.
    """
    ingest_id = fields.get("ingest_id")
    if ingest_id:
        try:
            await publish_status(
                redis,
                ingest_id,
                {
                    "state": "failed",
                    "ingest_id": ingest_id,
                    "error": "malformed queue entry (producer/consumer version skew)",
                    "error_type": "MalformedQueueEntry",
                },
            )
        except Exception:
            logger.exception(
                "settle malformed: publish_status failed for %s", ingest_id
            )
    sha = fields.get("sha")
    if sha:
        try:
            from cogniverse_runtime.ingestion_worker import idempotency

            await idempotency.clear_inflight(redis, sha)
        except Exception:
            logger.exception("settle malformed: clear_inflight failed for %s", sha)


def _parse_entry(message_id: str, fields: dict) -> Optional[IngestJob]:
    """Decode one stream entry, or None for a malformed one.

    An entry missing a required field (external XADD, producer/consumer
    version skew) can never be processed; raising here used to abort the
    whole claim/reclaim batch — in the reaper's sweep that happened BEFORE
    the dead-letter check, so one malformed entry stalled orphan recovery
    for every entry behind it. Callers settle the client (see
    settle_malformed_entry) and ack the malformed entry away.
    """
    try:
        return IngestJob(
            message_id=message_id,
            ingest_id=fields["ingest_id"],
            source_url=fields["source_url"],
            profile=fields["profile"],
            tenant_id=fields["tenant_id"],
            sha=fields["sha"],
        )
    except KeyError as exc:
        logger.error(
            "dropping malformed queue entry %s (missing %s): %r",
            message_id,
            exc,
            fields,
        )
        return None


async def autoclaim(
    redis: aioredis.Redis,
    group: str,
    consumer_id: str,
    *,
    min_idle_ms: int,
    start_id: str = "0-0",
    count: int = 10,
) -> tuple:
    """XAUTOCLAIM entries pending longer than ``min_idle_ms`` over to
    ``consumer_id``. Returns ``(next_cursor, jobs)``; a next_cursor of
    ``"0-0"`` means the scan wrapped and the PEL holds nothing further to
    examine. Entries that were XDEL'd from the stream while still pending
    are dropped from the PEL by Redis itself and never surface as jobs.
    """
    response = await redis.xautoclaim(
        QUEUE_STREAM,
        group,
        consumer_id,
        min_idle_time=min_idle_ms,
        start_id=start_id,
        count=count,
    )
    # Redis 7+ replies (cursor, entries, deleted); Redis 6 omits the third.
    next_cursor, entries = response[0], response[1]
    jobs: List[IngestJob] = []
    for message_id, fields in entries:
        if not fields:
            continue
        job = _parse_entry(message_id, fields)
        if job is None:
            await settle_malformed_entry(redis, fields)
            await ack(redis, group, message_id)
            continue
        jobs.append(job)
    return next_cursor, jobs


async def refresh_claim(
    redis: aioredis.Redis, group: str, consumer_id: str, message_id: str
) -> bool:
    """Reset the PEL idle clock on a claimed entry (XCLAIM to self).

    XAUTOCLAIM reclaims on idle time alone — it cannot tell a crashed
    owner from a live one mid-pipeline. The worker heartbeats this while
    processing, so the reaper's min-idle threshold only ever fires on
    entries whose owner stopped heartbeating. ``justid`` leaves the
    delivery counter untouched: heartbeats must not advance the
    poison-message cap. Returns True if the entry was still pending.
    """
    claimed = await redis.xclaim(
        QUEUE_STREAM,
        group,
        consumer_id,
        min_idle_time=0,
        message_ids=[message_id],
        justid=True,
    )
    return bool(claimed)


async def times_delivered(redis: aioredis.Redis, group: str, message_id: str) -> int:
    """How many times ``message_id`` has been delivered to any consumer.

    Every XREADGROUP claim, XCLAIM and XAUTOCLAIM bumps the PEL delivery
    counter — the reaper reads it to abandon poison messages instead of
    re-driving them forever. Returns 0 for an entry no longer pending.
    """
    detail = await redis.xpending_range(
        QUEUE_STREAM, group, min=message_id, max=message_id, count=1
    )
    if not detail:
        return 0
    return int(detail[0]["times_delivered"])


async def ack(redis: aioredis.Redis, group: str, message_id: str) -> int:
    """XACK then XDEL the message so a terminal job leaves the stream.

    XACK alone only clears the PEL; the entry lingers in the stream and
    XLEN (queue_depth, the cluster backpressure axis) keeps counting
    completed jobs, so after queue_depth_limit lifetime submissions every
    new submit 429s and Redis memory grows unbounded. XDEL removes the
    entry, bounding both XLEN and memory to the live (unacked) backlog.

    Returns 1 if the message was pending, 0 otherwise (already acked or
    never delivered to this group).
    """
    acked = await redis.xack(QUEUE_STREAM, group, message_id)
    await redis.xdel(QUEUE_STREAM, message_id)
    return acked


async def queue_depth(redis: aioredis.Redis) -> int:
    """XLEN the queue stream — used by backpressure to decide 429."""
    return await redis.xlen(QUEUE_STREAM)


# INCR and EXPIRE in one atomic script: with two round-trips, a connection
# drop after the INCR leaked a +1 with NO TTL, and the submit-side
# compensation (which never saw the increment succeed) couldn't decrement
# it — wedging the tenant's backpressure until a later increment happened
# to re-arm the TTL.
_INCR_WITH_TTL_LUA = """
local v = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[1])
return v
"""


async def increment_active(redis: aioredis.Redis, tenant_id: str) -> int:
    """Atomically INCR the per-tenant active counter with a fresh TTL.

    The TTL self-heals a leaked counter (incremented but never
    decremented) instead of wedging backpressure forever.
    """
    key = f"{ACTIVE_KEY_PREFIX}{tenant_id}"
    new_val = await redis.eval(_INCR_WITH_TTL_LUA, 1, key, ACTIVE_TTL_SECONDS)
    return int(new_val)


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

    The stream is capped (``MAXLEN``) and given a sliding TTL so it can't grow
    unbounded and is reclaimed after the job goes quiet, instead of leaking one
    immortal stream per ingest.
    """
    key = _status_stream_key(ingest_id)
    msg_id = await redis.xadd(
        key,
        {"data": json.dumps(event)},
        maxlen=STATUS_STREAM_MAXLEN,
        approximate=False,
    )
    await redis.expire(key, STATUS_STREAM_TTL_SECONDS)
    return msg_id


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
