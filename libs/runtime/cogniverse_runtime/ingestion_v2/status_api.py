"""SSE + status snapshot endpoints for the ingestion queue.

Mounted under the existing ``/ingestion`` prefix in main.py so the
public URLs are:
  - ``GET /ingestion/{ingest_id}/events``  — SSE event stream
  - ``GET /ingestion/{ingest_id}/status``  — point-read snapshot

Events replay from the start of the job (so a late-connecting client
sees the full history), then long-poll the Redis stream for new
events. The connection closes when a terminal event (``complete`` or
``failed``) is observed, or after ``timeout_seconds`` of inactivity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from cogniverse_runtime.ingestion_v2 import queue
from cogniverse_runtime.ingestion_v2.redis_client import get_redis

logger = logging.getLogger(__name__)

# When mounted with ``prefix="/ingestion"`` the full paths become
# ``/ingestion/{id}/events`` and ``/ingestion/{id}/status``.
router = APIRouter()

TERMINAL_STATES = {"complete", "failed"}


def _redis_url() -> str:
    url = os.environ.get("REDIS_URL")
    if not url:
        raise HTTPException(
            status_code=503,
            detail="REDIS_URL is not set; ingestion queue is not configured.",
        )
    return url


def _format_sse(event: dict, message_id: str) -> str:
    """Format one Server-Sent-Event frame with id + data fields. The
    id lets the browser EventSource auto-resume on reconnect via the
    ``Last-Event-ID`` header."""
    return f"id: {message_id}\ndata: {json.dumps(event)}\n\n"


@router.get("/{ingest_id}/events")
async def stream_events(
    ingest_id: str,
    last_event_id: str | None = Query(
        default=None,
        alias="last-event-id",
        description=(
            "Resume from this stream id; the EventSource client sends "
            "this automatically as the Last-Event-ID HTTP header."
        ),
    ),
    timeout_seconds: int = Query(
        default=600,
        ge=30,
        le=3600,
        description="Close the stream after this much idle time.",
    ),
) -> StreamingResponse:
    """SSE endpoint streaming the per-ingest event history then live
    updates. Terminal events end the stream; otherwise it ends after
    ``timeout_seconds`` of no new events."""
    redis = await get_redis(_redis_url())

    async def gen():
        last_id = last_event_id or "0-0"
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        sent_terminal = False

        while not sent_terminal and asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            block_ms = max(100, min(int(remaining * 1000), 30_000))
            events = await queue.read_status_since(
                redis, ingest_id, last_id=last_id, block_ms=block_ms
            )
            if not events:
                # Heartbeat to keep proxies from closing the connection
                # on long idle gaps. Comment lines (`:` prefix) are
                # ignored by EventSource clients.
                yield ": keep-alive\n\n"
                continue

            for message_id, event in events:
                last_id = message_id
                yield _format_sse(event, message_id)
                if event.get("state") in TERMINAL_STATES:
                    sent_terminal = True
                    break

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx response buffering
        },
    )


@router.get("/{ingest_id}/status")
async def get_status(ingest_id: str) -> dict:
    """Snapshot the latest event on the status stream — point-read
    alternative to the SSE stream for callers that don't speak
    EventSource. Returns the latest event plus the full history."""
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
        "history": [event for _, event in events],
    }
