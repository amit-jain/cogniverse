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
  - graph stage marked pending — never dead-lettered, whatever the
    delivery count, because acking would strand content whose graph
    transaction has not completed. Re-driven once the hold since its last
    recorded failure has elapsed: ``min_idle_ms`` doubled per re-drive so
    far, clamped at ``GRAPH_REDRIVE_HOLD_CAP_MS``, so a deterministic
    failure stops costing a full pipeline run every sweep while a
    transient one still recovers on the first re-drive.
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
from functools import partial

import redis.asyncio as aioredis

from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.worker import (
    GRAPH_REDRIVE_KEY_PREFIX,
    WorkerConfig,
    _clear_graph_pending,
    _default_processor,
    _is_graph_pending,
    _mark_graph_pending,
    _process_job,
    _server_time_ms,
)

logger = logging.getLogger(__name__)

DEAD_STREAM = "ingest:queue:dead"
DEAD_MARKER_PREFIX = "ingest:dead:"
# Must outlive the longest PEL idle a redelivery can arrive after (matches
# the done-marker's 7-day default) — at 6h, an entry that idled past the
# marker's expiry re-ran the settle and wrote a duplicate dead-stream row.
DEAD_MARKER_TTL_SECONDS = 7 * 24 * 60 * 60
GRAPH_REDRIVE_HOLD_CAP_MS = 6 * 60 * 60 * 1000

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


def graph_redrive_hold_ms(redrives: int, *, base_ms: int) -> int:
    """Wait after a graph-stage failure before the next re-drive: ``base_ms``
    doubled for each re-drive already made, clamped at
    ``GRAPH_REDRIVE_HOLD_CAP_MS``."""
    if redrives < 0:
        raise ValueError(f"redrives must be >= 0, got {redrives}")
    hold = base_ms
    for _ in range(redrives):
        if hold >= GRAPH_REDRIVE_HOLD_CAP_MS:
            break
        hold *= 2
    return min(hold, GRAPH_REDRIVE_HOLD_CAP_MS)


def _seconds(ms: int) -> str:
    return f"{ms / 1000:g}s"


async def _redrive_graph_pending(
    redis: aioredis.Redis,
    config: WorkerConfig,
    job,
    *,
    min_idle_ms: int,
    processor,
) -> bool:
    """Re-drive a graph-pending entry once its hold has elapsed. Returns
    False when the entry is still inside the hold; it stays claimed and
    unprocessed for the next sweep. The re-drive count and start time are
    recorded before the pipeline runs, so a run that dies without reporting
    still lengthens the next hold."""
    key = f"{GRAPH_REDRIVE_KEY_PREFIX}{job.message_id}"
    state = await redis.hgetall(key)
    redrives = int(state.get("redrives", 0))
    cause = state.get("cause", "none recorded")
    now_ms = await _server_time_ms(redis)
    if state:
        due_ms = int(state["at_ms"]) + graph_redrive_hold_ms(
            redrives, base_ms=min_idle_ms
        )
        if due_ms > now_ms:
            logger.info(
                "Reaper holding graph-pending ingest %s (tenant=%s, source=%s): "
                "%d re-drives so far, last outcome: %s; re-drive %d due in %s",
                job.ingest_id,
                job.tenant_id,
                job.source_url,
                redrives,
                cause,
                redrives + 1,
                _seconds(due_ms - now_ms),
            )
            return False
    redrive = redrives + 1
    logger.warning(
        "Reaper re-driving graph-pending ingest %s (tenant=%s, source=%s): "
        "re-drive %d, last outcome: %s; another failure holds re-drive %d for %s",
        job.ingest_id,
        job.tenant_id,
        job.source_url,
        redrive,
        cause,
        redrive + 1,
        _seconds(graph_redrive_hold_ms(redrive, base_ms=min_idle_ms)),
    )
    await redis.hset(
        key,
        mapping={
            "redrives": redrive,
            "at_ms": now_ms,
            "cause": f"re-drive {redrive} did not report an outcome",
        },
    )
    await _process_job(redis, job, config, processor=processor)
    return True


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
    processor=None,
    count: int = 1,
) -> int:
    """One full PEL sweep. Returns the number of entries recovered
    (settled or reprocessed). Entries idle less than ``min_idle_ms`` —
    a live worker's in-progress jobs — are never touched.

    ``count`` stays 1 so each orphan is claimed only when it is about to be
    processed. A larger batch parks the tail behind a minutes-long re-drive
    with its idle clock running: past ``min_idle_ms`` another replica's
    reaper reclaims the tail entry and both re-drive it concurrently —
    duplicate ingestion and a double-decremented tenant counter.

    After recovery, consumer names idle past ``min_idle_ms`` that own
    nothing are dropped from the group: every pod incarnation leaves one
    behind, and a reclaimed orphan's dead owner owns nothing once re-driven.
    """
    if processor is None:
        processor = partial(
            _default_processor,
            service_urls=config.inference_service_urls,
            mark_graph_pending=partial(_mark_graph_pending, redis),
            graph_deadline_s=config.graph_deadline_s,
        )
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
                await _clear_graph_pending(redis, job.message_id)
                await queue.ack(redis, config.consumer_group, job.message_id)
            else:
                delivered = await queue.times_delivered(
                    redis, config.consumer_group, job.message_id
                )
                if await _is_graph_pending(redis, job.message_id):
                    redriven = await _redrive_graph_pending(
                        redis,
                        config,
                        job,
                        min_idle_ms=min_idle_ms,
                        processor=processor,
                    )
                    if not redriven:
                        continue
                elif delivered > config.reaper_max_deliveries:
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
            break
    pruned = await queue.prune_consumers(
        redis, config.consumer_group, keep=config.consumer_id, min_idle_ms=min_idle_ms
    )
    if pruned:
        logger.info(
            "Reaper dropped %d idle consumer name(s) owning nothing: %s",
            len(pruned),
            pruned,
        )
    return recovered


async def reaper_loop(
    redis: aioredis.Redis,
    config: WorkerConfig,
    stop: asyncio.Event,
    *,
    processor,
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
