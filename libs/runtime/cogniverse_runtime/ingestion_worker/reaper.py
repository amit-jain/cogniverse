"""Crash recovery for orphaned in-flight ingestion jobs.

``claim()`` reads only never-delivered entries (``>``) and consumer ids
embed the pid, so a SIGKILLed / OOM-evicted worker's claimed-but-unacked
entry stays in its dead consumer's PEL forever: the upload is silently
lost, XLEN stays inflated, and queue-depth backpressure eventually 429s
every submit. The reaper XAUTOCLAIMs entries idle beyond a threshold back
to a live consumer and re-drives them through the normal processing path,
idempotently:

  - sha already marked done — the dead worker finished but never acked:
    settle the leftovers (clear the stale in-flight marker, ack) without
    re-running the pipeline or touching the active counter (the finished
    run already decremented it; a second decrement would free a slot a
    DIFFERENT still-running job of the same tenant holds).
  - anything else — re-drive through ``_process_job`` exactly like a
    fresh claim: the run publishes status, marks done, decrements the
    active counter the dead run never released, and acks.
  - redelivered past ``reaper_max_deliveries`` without completing — a
    poison message that kills the pod each run: abandon to the
    ``ingest:queue:dead`` stream with an observable ``failed`` terminal
    instead of crash-looping the ingestor forever.

``run_reaper_once`` is a single sweep; ``reaper_loop`` runs it on an
interval and is started by ``worker.run()`` when
``INGEST_REAPER_ENABLED`` is set (default on, first sweep only after one
full interval so short-lived processes never reap).
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as aioredis

from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.worker import (
    WorkerConfig,
    _default_processor,
    _process_job,
)

logger = logging.getLogger(__name__)

DEAD_STREAM = "ingest:queue:dead"
DEAD_MARKER_PREFIX = "ingest:dead:"
DEAD_MARKER_TTL_SECONDS = 6 * 60 * 60

# Atomic exactly-once dead-letter settle. The three side effects (dead-stream
# xadd, inflight clear, floored active decrement) are not individually
# idempotent — a crash between any two of them and the ack would redeliver the
# entry and re-run them: a second decrement frees a slot a DIFFERENT
# still-running job of the same tenant holds, and the dead stream gains a
# duplicate entry. Running them inside one server-side script gated on a
# SET NX marker means they happen exactly once no matter how many times the
# entry is redelivered.
# KEYS: marker, dead stream, inflight key, active-counter key.
# ARGV: ingest_id, marker TTL, source_url, profile, tenant_id, sha, delivered.
_SETTLE_LUA = """
if not redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2]) then
  return 0
end
redis.call('XADD', KEYS[2], '*',
  'ingest_id', ARGV[1], 'source_url', ARGV[3], 'profile', ARGV[4],
  'tenant_id', ARGV[5], 'sha', ARGV[6], 'times_delivered', ARGV[7])
redis.call('DEL', KEYS[3])
local v = redis.call('DECR', KEYS[4])
if v < 0 then redis.call('SET', KEYS[4], '0') end
return 1
"""


async def _dead_letter(
    redis: aioredis.Redis, config: WorkerConfig, job, delivered: int
) -> None:
    """Abandon a job redelivered past the cap without ever completing.

    A poison message that kills the pod each time it runs (an OOM-sized
    video) would otherwise crash-loop the ingestor forever, one re-drive
    per sweep. The job lands on the dead stream for operator inspection
    and its submit-side state is settled (slot freed, stale inflight
    cleared) in one atomic exactly-once step, so a crash-redelivery can
    never double-free a tenant slot or duplicate the dead entry. The
    observable ``failed`` terminal publishes after the settle (a watcher
    never sees it with invariants torn) and before the ack, so a crash in
    between redelivers the entry and re-publishes rather than losing the
    terminal forever. No done marker is written — a corrected
    re-submission re-enqueues.
    """
    logger.error(
        "Reaper abandoning ingest %s after %d deliveries (tenant=%s, "
        "source=%s) — moved to %s",
        job.ingest_id,
        delivered,
        job.tenant_id,
        job.source_url,
        DEAD_STREAM,
    )
    await redis.eval(
        _SETTLE_LUA,
        4,
        f"{DEAD_MARKER_PREFIX}{job.message_id}",
        DEAD_STREAM,
        f"{idempotency.INFLIGHT_KEY_PREFIX}{job.sha}",
        f"{queue.ACTIVE_KEY_PREFIX}{job.tenant_id}",
        job.ingest_id,
        str(DEAD_MARKER_TTL_SECONDS),
        job.source_url,
        job.profile,
        job.tenant_id,
        job.sha,
        str(delivered),
    )
    await queue.publish_status(
        redis,
        job.ingest_id,
        {
            "state": "failed",
            "ingest_id": job.ingest_id,
            "error": (f"abandoned after {delivered} deliveries without completing"),
            "error_type": "MaxDeliveriesExceeded",
        },
    )
    await queue.ack(redis, config.consumer_group, job.message_id)


async def run_reaper_once(
    redis: aioredis.Redis,
    config: WorkerConfig,
    *,
    min_idle_ms: int,
    processor=_default_processor,
    count: int = 10,
) -> int:
    """One full PEL sweep. Returns the number of entries recovered
    (settled or reprocessed). Entries idle less than ``min_idle_ms`` —
    a live worker's in-progress jobs — are never touched.
    """
    recovered = 0
    cursor = "0-0"
    while True:
        cursor, jobs = await queue.autoclaim(
            redis,
            config.consumer_group,
            config.consumer_id,
            min_idle_ms=min_idle_ms,
            start_id=cursor,
            count=count,
        )
        for job in jobs:
            done_id = await idempotency.get_done_ingest_id(redis, job.sha)
            if done_id:
                logger.info(
                    "Reaper settling finished-but-unacked ingest %s (completed as %s)",
                    job.ingest_id,
                    done_id,
                )
                await idempotency.clear_inflight(redis, job.sha)
                await queue.ack(redis, config.consumer_group, job.message_id)
            else:
                delivered = await queue.times_delivered(
                    redis, config.consumer_group, job.message_id
                )
                if delivered > config.reaper_max_deliveries:
                    await _dead_letter(redis, config, job, delivered)
                else:
                    logger.warning(
                        "Reaper re-driving orphaned ingest %s (tenant=%s, "
                        "source=%s, delivery %d/%d)",
                        job.ingest_id,
                        job.tenant_id,
                        job.source_url,
                        delivered,
                        config.reaper_max_deliveries,
                    )
                    await _process_job(redis, job, config, processor=processor)
            recovered += 1
        if cursor == "0-0" or not jobs:
            return recovered


async def reaper_loop(
    redis: aioredis.Redis,
    config: WorkerConfig,
    stop: asyncio.Event,
    *,
    processor=_default_processor,
) -> None:
    """Sweep every ``config.reaper_interval_s`` until ``stop`` is set.

    Sleeps BEFORE the first sweep, so a process that lives less than one
    interval (tests, crash loops) never reclaims anything. A failed sweep
    is logged and retried on the next interval — the reaper must outlive
    transient Redis blips.
    """
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=config.reaper_interval_s)
            return
        except asyncio.TimeoutError:
            pass
        try:
            n = await run_reaper_once(
                redis,
                config,
                min_idle_ms=config.reaper_min_idle_ms,
                processor=processor,
            )
            if n:
                logger.info("Reaper recovered %d orphaned entries", n)
        except Exception:
            logger.exception("Reaper sweep failed; retrying next interval")
