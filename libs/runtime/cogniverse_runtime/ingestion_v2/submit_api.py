"""``POST /ingest`` — enqueue a submission for the worker pool.

Defaults to async (returns 202 + ingest_id immediately). Pass
``?wait=true`` to long-poll the status stream and return when the
job hits a terminal event; useful for short jobs (documents, audio
under a few minutes). Long videos should always use the async path
+ SSE — proxies typically kill HTTP connections at 5 min.

Idempotency: same ``(source_url, profile, tenant_id)`` returns the
existing ``ingest_id`` (whether in-flight or completed within the
TTL window) without re-enqueuing. ``?force=true`` clears the
idempotency record and forces a re-enqueue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from cogniverse_runtime.ingestion_v2 import backpressure, idempotency, queue
from cogniverse_runtime.ingestion_v2.redis_client import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestRequest(BaseModel):
    source_url: str = Field(
        ...,
        description=(
            "URL of the source asset. Schemes: ``s3://`` (MinIO/S3), "
            "``http(s)://``, ``file://``."
        ),
    )
    profile: str = Field(
        ...,
        description="Ingestion profile (e.g. ``video_colpali_smol500_mv_frame``).",
    )
    tenant_id: str = Field(..., description="Tenant the ingestion belongs to.")


class IngestResponse(BaseModel):
    ingest_id: str
    status: str  # "queued" | "in_flight" | "complete" | "failed"
    existing: bool = Field(
        default=False,
        description=(
            "True when the response references an existing ingestion "
            "(idempotency hit). False on a fresh enqueue."
        ),
    )
    sha: str
    final_event: Optional[dict] = Field(
        default=None,
        description=(
            "Populated only when ``wait=true`` and the job reached a "
            "terminal state before the deadline."
        ),
    )


def _redis_url() -> str:
    """Read the Redis URL from env at request time. The env var is set
    once by the chart at pod startup; reading it here keeps this module
    free of import-time env coupling."""
    url = os.environ.get("REDIS_URL")
    if not url:
        raise HTTPException(
            status_code=503,
            detail=(
                "REDIS_URL is not set on the runtime pod — the ingestion "
                "queue is not configured. Enable redis in chart values."
            ),
        )
    return url


def _backpressure_limits() -> tuple[int, int]:
    return (
        int(os.environ.get("INGEST_QUEUE_DEPTH_LIMIT", "1000")),
        int(os.environ.get("INGEST_PER_TENANT_CONCURRENCY", "4")),
    )


async def _wait_for_terminal(
    redis: Any, ingest_id: str, deadline_seconds: float
) -> Optional[dict]:
    """Poll the status stream until a terminal event is observed or
    ``deadline_seconds`` elapses. Terminal = ``state in {complete, failed}``."""
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


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def submit_ingest(
    body: IngestRequest,
    request: Request,
    wait: bool = Query(
        False, description="Block until the job reaches a terminal state."
    ),
    wait_timeout: int = Query(
        300, ge=10, le=900, description="Max seconds to block when wait=true."
    ),
    force: bool = Query(
        False,
        description=(
            "Bypass the idempotency cache and re-enqueue even if a "
            "matching submission completed within the TTL window."
        ),
    ),
) -> IngestResponse:
    """Enqueue an ingestion or return the existing run."""
    redis = await get_redis(_redis_url())

    sha = idempotency.compute_sha(body.source_url, body.profile, body.tenant_id)

    if force:
        await idempotency.clear_done(redis, sha)
    else:
        existing = await idempotency.get_existing_ingest_id(redis, sha)
        if existing:
            return IngestResponse(
                ingest_id=existing, status="in_flight", existing=True, sha=sha
            )

    queue_limit, tenant_limit = _backpressure_limits()
    rejection = await backpressure.check(
        redis,
        body.tenant_id,
        queue_depth_limit=queue_limit,
        per_tenant_concurrency=tenant_limit,
    )
    if rejection:
        raise HTTPException(
            status_code=429,
            detail={
                "axis": rejection.axis,
                "current": rejection.current,
                "limit": rejection.limit,
                "message": rejection.message,
            },
        )

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
            "source_url": body.source_url,
            "profile": body.profile,
            "tenant_id": body.tenant_id,
        },
    )
    await queue.submit(
        redis,
        ingest_id=ingest_id,
        source_url=body.source_url,
        profile=body.profile,
        tenant_id=body.tenant_id,
        sha=sha,
    )
    await queue.increment_active(redis, body.tenant_id)

    logger.info(
        "Ingest submitted: id=%s tenant=%s profile=%s source=%s",
        ingest_id,
        body.tenant_id,
        body.profile,
        body.source_url,
    )

    if not wait:
        return IngestResponse(ingest_id=ingest_id, status="queued", sha=sha)

    final = await _wait_for_terminal(redis, ingest_id, wait_timeout)
    if final is None:
        raise HTTPException(
            status_code=504,
            detail={
                "ingest_id": ingest_id,
                "message": (
                    f"Ingestion did not reach terminal state within "
                    f"{wait_timeout}s. Poll /ingest/{ingest_id}/events instead."
                ),
            },
        )
    return IngestResponse(
        ingest_id=ingest_id,
        status=final.get("state", "complete"),
        sha=sha,
        final_event=final,
    )


@router.get("/ingest/{ingest_id}/status", response_model=dict)
async def get_status(ingest_id: str, request: Request) -> dict:
    """Snapshot the latest event on the status stream — point-read
    alternative to the SSE stream for callers that don't speak EventSource."""
    redis = await get_redis(_redis_url())
    events = await queue.read_status_since(redis, ingest_id)
    if not events:
        raise HTTPException(
            status_code=404,
            detail=f"No status events for ingest_id={ingest_id!r}",
        )
    _, latest = events[-1]
    return {
        "ingest_id": ingest_id,
        "state": latest.get("state", "unknown"),
        "events_count": len(events),
        "latest": latest,
        # Echo the wire format for tools that prefer the raw stream:
        "history": [json.loads(json.dumps(e)) for _, e in events],
    }
