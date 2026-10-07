"""
SSE Streaming Endpoints for Real-Time Event Notifications

Provides Server-Sent Events (SSE) streaming for:
- Orchestrator and deep-research workflow progress
- Ingestion job progress
- Task cancellation

Every runtime process serves every task: events, cancellations and the
active-task index live in the shared Redis task event store
(``cogniverse_runtime.task_events``), whichever process runs the task.

Endpoints:
- GET /events/workflows/{workflow_id} - Stream workflow events
- GET /events/ingestion/{job_id} - Stream ingestion events
- POST /events/workflows/{workflow_id}/cancel - Cancel workflow
- POST /events/ingestion/{job_id}/cancel - Cancel ingestion
- GET /events/queues - List a tenant's active tasks
- GET /events/queues/{task_id} - A task's state
- GET /events/queues/{task_id}/offset - A task's next event offset
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cogniverse_runtime.http_errors import failure_response, record_failure
from cogniverse_runtime.task_events import (
    INGESTION,
    WORKFLOW,
    TaskEventStore,
    TaskEventsUnavailable,
    TaskRead,
    stream_event,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_store: Optional[TaskEventStore] = None


def set_task_event_store(store: Optional[TaskEventStore]) -> None:
    """Install the process's task event store (``None`` at shutdown)."""
    global _store
    _store = store


def get_task_event_store() -> TaskEventStore:
    if _store is None:
        raise TaskEventsUnavailable(
            "task event store unavailable: none is configured in this process"
        )
    return _store


def _unavailable(exc: TaskEventsUnavailable, task_id: Optional[str] = None):
    fields = {"task_id": task_id} if task_id is not None else {}
    return failure_response(
        503,
        "task_events_unavailable",
        "The task event store did not answer; retry.",
        exc,
        **fields,
    )


class CancelRequest(BaseModel):
    """Request body for cancellation"""

    reason: Optional[str] = None


class CancelResponse(BaseModel):
    """Response for cancellation request"""

    task_id: str
    cancelled: bool
    message: str


class QueueInfo(BaseModel):
    """A task's state"""

    task_id: str
    kind: str
    tenant_id: str
    event_count: int
    subscriber_count: int
    is_closed: bool
    is_cancelled: bool
    created_at: str


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _event_stream(
    store: TaskEventStore,
    kind: str,
    task_id: str,
    first: TaskRead,
    subscriber: str,
    from_offset: int = 0,
    heartbeat_interval: float = 15.0,
) -> AsyncGenerator[str, None]:
    """
    Generate the SSE stream of a task's events.

    Opens with a ``connected`` event, then delivers every event from
    ``from_offset`` on, and ends once the task has ended and every event was
    delivered. A task that stops reporting before it ends gets an ``error``
    event; a store that stops answering gets a ``stream_error`` event.

    Args:
        first: The read the route made before answering
        subscriber: This reader's subscriber id, registered by that read
        from_offset: Start from this offset (for replay)
        heartbeat_interval: Seconds between heartbeat comments
    """
    logger.info(f"SSE stream started for task {task_id} (offset: {from_offset})")
    loop = asyncio.get_running_loop()
    try:
        yield _sse(
            {
                "type": "connected",
                "task_id": task_id,
                "offset": from_offset,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        read: Optional[TaskRead] = first
        after, last_id = from_offset - 1, None
        quiet_since = loop.time()
        while True:
            if read is None:
                yield _sse(
                    {
                        "type": "error",
                        "message": f"Task {task_id} expired while it was streamed",
                    }
                )
                return
            for offset, entry_id, data in read.events:
                event = stream_event(kind, task_id, read.tenant_id, entry_id, data)
                yield f"data: {json.dumps(event)}\n\n"
                after, last_id = offset, entry_id
                quiet_since = loop.time()
            if read.closed and after + 1 >= read.appended:
                return
            if read.stopped_reporting:
                yield _sse(
                    {
                        "type": "error",
                        "message": f"Task {task_id} stopped reporting before it "
                        "finished",
                    }
                )
                return
            if not read.events:
                if loop.time() - quiet_since >= heartbeat_interval:
                    # SSE comment line — ignored by clients, keeps the pipe warm.
                    yield ": heartbeat\n\n"
                    quiet_since = loop.time()
                await asyncio.sleep(store.read_interval_s)
            read = await store.read(
                task_id,
                kind=kind,
                after_offset=after,
                last_entry_id=last_id,
                subscriber=subscriber,
            )
    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled for task {task_id}")
        raise
    except Exception as e:
        record_failure(e, "stream_error")
        yield _sse(
            {
                "type": "stream_error",
                "message": f"The event stream for task {task_id} failed.",
                "failure": type(e).__name__,
            }
        )
    finally:
        try:
            await asyncio.shield(store.leave(task_id, subscriber))
        except Exception as e:
            logger.warning(
                "Subscriber %s of task %s not released: %s", subscriber, task_id, e
            )
        logger.info(f"SSE stream ended for task {task_id}")


async def _stream(kind: str, task_id: str, from_offset: int) -> StreamingResponse:
    subscriber = uuid.uuid4().hex
    try:
        store = get_task_event_store()
        first = await store.read(
            task_id,
            kind=kind,
            after_offset=from_offset - 1,
            subscriber=subscriber,
        )
    except TaskEventsUnavailable as exc:
        raise _unavailable(exc, task_id) from exc
    if first is None:

        async def missing() -> AsyncGenerator[str, None]:
            yield _sse(
                {"type": "error", "message": f"No active queue for task {task_id}"}
            )

        body = missing()
    else:
        body = _event_stream(store, kind, task_id, first, subscriber, from_offset)
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/workflows/{workflow_id}")
async def stream_workflow_events(
    workflow_id: str,
    from_offset: int = Query(default=0, ge=0, description="Event offset to start from"),
):
    """
    Stream workflow events via SSE.

    Subscribe to real-time updates for an orchestrator or deep-research
    workflow, whichever runtime process runs it. Supports replay from offset
    for reconnection.

    Args:
        workflow_id: Workflow ID to stream events for
        from_offset: Start streaming from this event offset
    """
    return await _stream(WORKFLOW, workflow_id, from_offset)


@router.get("/ingestion/{job_id}")
async def stream_ingestion_events(
    job_id: str,
    from_offset: int = Query(default=0, ge=0, description="Event offset to start from"),
):
    """
    Stream ingestion job events via SSE.

    Reads the job's ingestion status stream (the one ``/ingestion/{id}/events``
    serves) and reports each status as a task event. Supports replay from
    offset for reconnection.

    Args:
        job_id: Ingestion job ID to stream events for
        from_offset: Start streaming from this event offset
    """
    return await _stream(INGESTION, job_id, from_offset)


async def _cancel(kind: str, task_id: str, request: Optional[CancelRequest], noun):
    reason = request.reason if request else None
    try:
        outcome = await get_task_event_store().cancel(kind, task_id, reason)
    except TaskEventsUnavailable as exc:
        raise _unavailable(exc, task_id) from exc
    if outcome == "missing":
        raise HTTPException(
            status_code=404,
            detail=f"No active {noun.lower()} found with ID {task_id}",
        )
    if outcome == "finished":
        raise HTTPException(
            status_code=409, detail=f"{noun} {task_id} already finished"
        )
    if outcome == "stopped":
        raise HTTPException(
            status_code=409,
            detail=f"{noun} {task_id} stopped reporting before it finished",
        )
    return CancelResponse(
        task_id=task_id,
        cancelled=True,
        message=f"{noun} {task_id} cancellation requested",
    )


@router.post("/workflows/{workflow_id}/cancel", response_model=CancelResponse)
async def cancel_workflow(workflow_id: str, request: CancelRequest = None):
    """
    Cancel a running workflow.

    Records the cancellation; the process running the workflow picks it up
    and stops the workflow at its next phase boundary. Does not interrupt a
    phase in progress.

    Args:
        workflow_id: Workflow ID to cancel
        request: Optional cancellation reason
    """
    return await _cancel(WORKFLOW, workflow_id, request, "Workflow")


@router.post("/ingestion/{job_id}/cancel", response_model=CancelResponse)
async def cancel_ingestion(job_id: str, request: CancelRequest = None):
    """
    Cancel a running ingestion job.

    Records the cancellation; the process running the job stops it before its
    next video. Does not interrupt the video being processed; a queued job
    is cancelled before it starts.

    Args:
        job_id: Ingestion job ID to cancel
        request: Optional cancellation reason
    """
    return await _cancel(INGESTION, job_id, request, "Ingestion job")


@router.get("/queues", response_model=list[QueueInfo])
async def list_active_queues(
    tenant_id: str = Query(..., description="Tenant ID to list queues for"),
):
    """
    List a tenant's active tasks: running or queued workflows and ingestion
    jobs, whichever process runs them.

    When auth is added, tenant_id will be extracted from the auth token
    and validated against the query param.

    Args:
        tenant_id: Tenant ID (required — users only see their own queues)
    """
    try:
        queues = await get_task_event_store().list_active(tenant_id)
    except TaskEventsUnavailable as exc:
        raise _unavailable(exc) from exc
    return [QueueInfo(**queue) for queue in queues]


async def _read_task(task_id: str) -> TaskRead:
    try:
        read = await get_task_event_store().read(task_id, count=0)
    except TaskEventsUnavailable as exc:
        raise _unavailable(exc, task_id) from exc
    if read is None:
        raise HTTPException(
            status_code=404,
            detail=f"No queue found for task {task_id}",
        )
    return read


@router.get("/queues/{task_id}")
async def get_queue_info(task_id: str) -> QueueInfo:
    """
    Get a task's state, while its events are retained.

    Args:
        task_id: Task ID (workflow_id or job_id)
    """
    return QueueInfo(**(await _read_task(task_id)).info(task_id))


@router.get("/queues/{task_id}/offset")
async def get_queue_offset(task_id: str):
    """
    Get the offset the task's next event will take.

    Useful for clients to determine where to resume from.

    Args:
        task_id: Task ID (workflow_id or job_id)
    """
    read = await _read_task(task_id)
    return {"task_id": task_id, "offset": read.appended}
