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

from cogniverse_runtime.ingestion_v2 import backpressure, idempotency, queue

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
        if existing_id:
            return EnqueueResult(
                ingest_id=existing_id,
                sha=sha,
                state="in_flight",
                existing=True,
            )

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
    await idempotency.mark_inflight(redis, sha, ingest_id)
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
    await queue.increment_active(redis, tenant_id)

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
