"""Submission helper for the ingestion queue.

Single entry point ``enqueue_ingestion`` used by both
``POST /ingestion/upload`` (after writing the multipart bytes to
MinIO) and ``POST /ingestion/start`` (which iterates a source list
and submits one job per item). No router defined here — the public
URLs live in ``routers/ingestion.py``.

Idempotency: same ``(source_url, profile, tenant_id)`` returns the
existing ``ingest_id`` (whether in-flight or completed within the
TTL window) without re-enqueuing. ``force=True`` clears the
idempotency record and forces a re-enqueue.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis

from cogniverse_runtime.ingestion_worker import backpressure, idempotency, queue

logger = logging.getLogger(__name__)


class BackpressureError(Exception):
    """Raised when a submission is rejected by either backpressure axis.

    Carries the rejection reason so the HTTP layer can shape a 429
    response with structured ``{axis, current, limit}`` detail.
    """

    def __init__(self, rejection: backpressure.BackpressureRejection):
        super().__init__(rejection.message)
        self.rejection = rejection


@dataclass(frozen=True)
class EnqueueResult:
    ingest_id: str
    sha: str
    state: str  # "queued" | "in_flight" | "complete" | "failed"
    existing: bool  # True iff this is an idempotency hit
    final_event: Optional[dict] = None  # populated only when wait=True


def _backpressure_limits() -> tuple[int, int]:
    """Read the two thresholds from env (set by the chart).

    Defaults are conservative for laptop-scale dev; production sets
    them via the chart's ``ingestor.{queueDepthLimit,perTenantConcurrency}``.
    """
    return (
        int(os.environ.get("INGEST_QUEUE_DEPTH_LIMIT", "1000")),
        int(os.environ.get("INGEST_PER_TENANT_CONCURRENCY", "4")),
    )


def _inflight_ttl_seconds() -> int:
    """Crash-recovery TTL for the in-flight idempotency key (default 6h).

    The worker clears the key on terminal state; this bound only fires
    when a worker dies mid-job. It must exceed the longest realistic
    ingestion so a live run is never expired out from under a re-submit.
    """
    return int(os.environ.get("INGEST_INFLIGHT_TTL_SECONDS", "21600"))


# A submit goes claim_inflight -> increment -> publish_status -> submit ->
# mark_submitted in a handful of local Redis round-trips (tens of ms). An
# inflight marker without a submitted marker that is older than this grace did
# not merely lose a race with a concurrent winner mid-submit — it is a phantom
# from a crash before submit. Far shorter than the 6h inflight TTL so a phantom
# self-heals in seconds, far longer than the real claim->submit window so a live
# winner is never re-enqueued.
_COMMIT_GRACE_SECONDS = 30.0


async def _existing_run_is_real(redis: aioredis.Redis, sha: str) -> bool:
    """Whether an existing idempotency record names a run that actually exists.

    True for a completed run (done marker) or an in-flight run that reached the
    work stream (submitted marker). A bare inflight marker with no submitted
    marker is a phantom from a crash between claim and submit — unless it was
    claimed within the commit grace, in which case a concurrent winner may still
    be mid-submit and we must not re-enqueue.
    """
    if await idempotency.get_done_ingest_id(redis, sha):
        return True
    if await idempotency.is_submitted(redis, sha):
        return True
    age = await idempotency.inflight_claim_age_seconds(
        redis, sha, _inflight_ttl_seconds()
    )
    if age is None:
        # Cannot age the marker (no TTL) — stay conservative, treat as real.
        return True
    return age < _COMMIT_GRACE_SECONDS


async def _wait_for_terminal(
    redis: aioredis.Redis, ingest_id: str, deadline_seconds: float
) -> Optional[dict]:
    """Long-poll the status stream until a terminal event is observed
    or ``deadline_seconds`` elapses. Terminal = ``state in {complete,
    failed}``. Returns None on timeout."""
    last_id = "0-0"
    deadline = asyncio.get_event_loop().time() + deadline_seconds
    while asyncio.get_event_loop().time() < deadline:
        remaining_ms = max(
            100, int((deadline - asyncio.get_event_loop().time()) * 1000)
        )
        events = await queue.read_status_since(
            redis, ingest_id, last_id=last_id, block_ms=min(remaining_ms, 5000)
        )
        for message_id, event in events:
            last_id = message_id
            if event.get("state") in ("complete", "failed"):
                return event
    return None


async def enqueue_ingestion(
    redis: aioredis.Redis,
    *,
    source_url: str,
    profile: str,
    tenant_id: str,
    force: bool = False,
    wait: bool = False,
    wait_timeout: int = 300,
) -> EnqueueResult:
    """Enqueue an ingestion or return the existing run.

    Raises ``BackpressureError`` when either backpressure axis is
    exceeded. Caller (the HTTP route) maps it to 429.
    """
    sha = idempotency.compute_sha(source_url, profile, tenant_id)

    if force:
        await idempotency.clear_done(redis, sha)
    else:
        existing_id = await idempotency.get_existing_ingest_id(redis, sha)
        if existing_id and await _existing_run_is_real(redis, sha):
            return EnqueueResult(
                ingest_id=existing_id,
                sha=sha,
                state="in_flight",
                existing=True,
            )
        if existing_id:
            # The inflight marker exists but the run never reached the work
            # stream — a hard crash between claim_inflight and submit orphaned
            # it. Clear it and fall through to re-enqueue instead of handing
            # back a phantom id that never completes. claim_inflight (SET NX)
            # below still dedupes concurrent resubmits.
            logger.warning(
                "Clearing phantom inflight marker for sha=%s (id=%s): the run "
                "never reached the work stream",
                sha,
                existing_id,
            )
            await idempotency.clear_inflight(redis, sha)

    queue_limit, tenant_limit = _backpressure_limits()
    rejection = await backpressure.check(
        redis,
        tenant_id,
        queue_depth_limit=queue_limit,
        per_tenant_concurrency=tenant_limit,
    )
    if rejection:
        raise BackpressureError(rejection)

    ingest_id = f"ingest_{uuid.uuid4().hex}"
    await queue.ensure_consumer_group(
        redis, os.environ.get("INGEST_CONSUMER_GROUP", "ingestors")
    )
    holder = await idempotency.claim_inflight(
        redis, sha, ingest_id, ttl_seconds=_inflight_ttl_seconds()
    )
    if holder is not None:
        # A concurrent identical submission won the claim between the
        # early existing-id check and here — return its run, enqueue
        # nothing, touch no counters.
        return EnqueueResult(
            ingest_id=holder,
            sha=sha,
            state="in_flight",
            existing=True,
        )
    # Increment the active counter BEFORE the job reaches the work stream, so
    # the increment always precedes claimability. If it ran after submit(), a
    # fast worker could claim + process + decrement (floored at 0) in the
    # window before the increment, sticking the per-tenant counter at 1 for the
    # whole ACTIVE_TTL. The inflight key is written before all of this; any
    # failure between it and a successful submit must compensate every step
    # that committed, or the orphaned key (6h TTL) makes every non-force retry
    # return "in_flight" for a job that was never queued. Compensation steps
    # are each best-effort and only for steps that actually ran: a failing
    # decrement must not skip clearing the inflight marker (the counter
    # self-heals via ACTIVE_TTL; the marker poisons resubmits for its whole
    # TTL), and the original submit error must surface, not the
    # compensation's.
    incremented = False
    try:
        await queue.increment_active(redis, tenant_id)
        incremented = True
        await queue.publish_status(
            redis,
            ingest_id,
            {
                "state": "queued",
                "ingest_id": ingest_id,
                "source_url": source_url,
                "profile": profile,
                "tenant_id": tenant_id,
            },
        )
        await queue.submit(
            redis,
            ingest_id=ingest_id,
            source_url=source_url,
            profile=profile,
            tenant_id=tenant_id,
            sha=sha,
        )
        # The job is now in the work stream — mark it committed so a resubmit
        # can tell a genuine in-flight run from a phantom left by a crash
        # before this point.
        await idempotency.mark_submitted(redis, sha, _inflight_ttl_seconds())
    except Exception:
        if incremented:
            try:
                await queue.decrement_active(redis, tenant_id)
            except Exception:
                logger.exception(
                    "Enqueue compensation: decrement_active failed for "
                    "tenant=%s; counter self-heals via ACTIVE_TTL",
                    tenant_id,
                )
        try:
            await idempotency.clear_inflight(redis, sha)
        except Exception:
            logger.exception(
                "Enqueue compensation: clear_inflight failed for sha=%s; "
                "resubmits return the dead id until the inflight TTL lapses",
                sha,
            )
        raise

    logger.info(
        "Ingest enqueued: id=%s tenant=%s profile=%s source=%s",
        ingest_id,
        tenant_id,
        profile,
        source_url,
    )

    if not wait:
        return EnqueueResult(
            ingest_id=ingest_id, sha=sha, state="queued", existing=False
        )

    final = await _wait_for_terminal(redis, ingest_id, wait_timeout)
    if final is None:
        return EnqueueResult(
            ingest_id=ingest_id,
            sha=sha,
            state="queued",
            existing=False,
            final_event=None,
        )
    return EnqueueResult(
        ingest_id=ingest_id,
        sha=sha,
        state=final.get("state", "complete"),
        existing=False,
        final_event=final,
    )
