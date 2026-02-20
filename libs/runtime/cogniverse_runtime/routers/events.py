"""
SSE Streaming Endpoints for Real-Time Event Notifications

Provides Server-Sent Events (SSE) streaming for:
- Orchestrator workflow progress
- Ingestion pipeline progress
- Task cancellation

Endpoints:
- GET /events/workflows/{workflow_id} - Stream workflow events
- GET /events/ingestion/{job_id} - Stream ingestion events
- POST /events/workflows/{workflow_id}/cancel - Cancel workflow
- POST /events/ingestion/{job_id}/cancel - Cancel ingestion
- GET /events/queues - List active queues (admin)
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cogniverse_core.events import (
    get_queue_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class CancelRequest(BaseModel):
    """Request body for cancellation"""

    reason: Optional[str] = None


class CancelResponse(BaseModel):
    """Response for cancellation request"""

    task_id: str
    cancelled: bool
    message: str


class QueueInfo(BaseModel):
    """Queue information"""

    task_id: str
    tenant_id: str
    event_count: int
    subscriber_count: int
    is_closed: bool
    is_cancelled: bool
    created_at: str


async def _event_stream(
    task_id: str,
    from_offset: int = 0,
    heartbeat_interval: float = 15.0,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream from EventQueue.

    Args:
        task_id: Task ID to stream events for
        from_offset: Start from this offset (for replay)
        heartbeat_interval: Seconds between heartbeat comments
    """
    queue_manager = get_queue_manager()
    queue = await queue_manager.get_queue(task_id)

    if queue is None:
        # Send error event and close
        error_data = json.dumps(
            {
                "type": "error",
                "message": f"No active queue for task {task_id}",
            }
        )
        yield f"data: {error_data}\n\n"
        return

    logger.info(f"SSE stream started for task {task_id} (offset: {from_offset})")

    try:
        # Send initial connection event
        connect_data = json.dumps(
            {
                "type": "connected",
                "task_id": task_id,
                "offset": from_offset,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        yield f"data: {connect_data}\n\n"

        # Subscribe to queue
        async for event in queue.subscribe(from_offset=from_offset):
            # Serialize event to JSON
            event_data = event.model_dump_json()
            yield f"data: {event_data}\n\n"

            # Check if this is a completion or error event
            if event.event_type in ("complete", "error"):
                break

    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled for task {task_id}")
        raise
    except Exception as e:
        logger.error(f"SSE stream error for task {task_id}: {e}")
        error_data = json.dumps(
            {
                "type": "stream_error",
                "message": str(e),
            }
        )
        yield f"data: {error_data}\n\n"
    finally:
        logger.info(f"SSE stream ended for task {task_id}")


@router.get("/workflows/{workflow_id}")
async def stream_workflow_events(
    workflow_id: str,
    from_offset: int = Query(default=0, ge=0, description="Event offset to start from"),
):
    """
    Stream workflow events via SSE.

    Subscribe to real-time updates for an orchestrator workflow.
    Supports replay from offset for reconnection.

    Args:
        workflow_id: Workflow ID to stream events for
        from_offset: Start streaming from this event offset
    """
    return StreamingResponse(
        _event_stream(workflow_id, from_offset),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/ingestion/{job_id}")
async def stream_ingestion_events(
    job_id: str,
    from_offset: int = Query(default=0, ge=0, description="Event offset to start from"),
):
    """
    Stream ingestion job events via SSE.

    Subscribe to real-time updates for a video ingestion job.
    Supports replay from offset for reconnection.

    Args:
        job_id: Ingestion job ID to stream events for
        from_offset: Start streaming from this event offset
    """
    return StreamingResponse(
        _event_stream(job_id, from_offset),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/workflows/{workflow_id}/cancel", response_model=CancelResponse)
async def cancel_workflow(workflow_id: str, request: CancelRequest = None):
    """
    Cancel a running workflow.

    Signals the workflow to stop gracefully at the next phase boundary.
    Does not immediately terminate running tasks.

    Args:
        workflow_id: Workflow ID to cancel
        request: Optional cancellation reason
    """
    queue_manager = get_queue_manager()
    reason = request.reason if request else None

    cancelled = await queue_manager.cancel_task(workflow_id, reason)

    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"No active workflow found with ID {workflow_id}",
        )

    return CancelResponse(
        task_id=workflow_id,
        cancelled=True,
        message=f"Workflow {workflow_id} cancellation requested",
    )


@router.post("/ingestion/{job_id}/cancel", response_model=CancelResponse)
async def cancel_ingestion(job_id: str, request: CancelRequest = None):
    """
    Cancel a running ingestion job.

    Signals the ingestion pipeline to stop gracefully between videos.
    Does not immediately terminate the currently processing video.

    Args:
        job_id: Ingestion job ID to cancel
        request: Optional cancellation reason
    """
    queue_manager = get_queue_manager()
    reason = request.reason if request else None

    cancelled = await queue_manager.cancel_task(job_id, reason)

    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"No active ingestion job found with ID {job_id}",
        )

    return CancelResponse(
        task_id=job_id,
        cancelled=True,
        message=f"Ingestion job {job_id} cancellation requested",
    )


@router.get("/queues", response_model=list[QueueInfo])
async def list_active_queues(
    tenant_id: str = Query(..., description="Tenant ID to list queues for"),
):
    """
    List active event queues for a tenant.

    Users see their own tenant's active workflows and ingestion jobs.
    When auth is added, tenant_id will be extracted from the auth token
    and validated against the query param.

    Args:
        tenant_id: Tenant ID (required — users only see their own queues)
    """
    queue_manager = get_queue_manager()
    queues = await queue_manager.list_active_queues(tenant_id)

    return [
        QueueInfo(
            task_id=q["task_id"],
            tenant_id=q["tenant_id"],
            event_count=q["event_count"],
            subscriber_count=q["subscriber_count"],
            is_closed=q["is_closed"],
            is_cancelled=q["is_cancelled"],
            created_at=q["created_at"],
        )
        for q in queues
    ]


@router.get("/queues/{task_id}")
async def get_queue_info(task_id: str):
    """
    Get information about a specific queue.

    Args:
        task_id: Task ID (workflow_id or job_id)
    """
    queue_manager = get_queue_manager()
    queue = await queue_manager.get_queue(task_id)

    if queue is None:
        raise HTTPException(
            status_code=404,
            detail=f"No queue found for task {task_id}",
        )

    stats = queue.get_stats()
    return QueueInfo(
        task_id=stats["task_id"],
        tenant_id=stats["tenant_id"],
        event_count=stats["event_count"],
        subscriber_count=stats["subscriber_count"],
        is_closed=stats["is_closed"],
        is_cancelled=stats["is_cancelled"],
        created_at=stats["created_at"],
    )


@router.get("/queues/{task_id}/offset")
async def get_queue_offset(task_id: str):
    """
    Get current event offset for a queue.

    Useful for clients to determine where to resume from.

    Args:
        task_id: Task ID (workflow_id or job_id)
    """
    queue_manager = get_queue_manager()
    queue = await queue_manager.get_queue(task_id)

    if queue is None:
        raise HTTPException(
            status_code=404,
            detail=f"No queue found for task {task_id}",
        )

    offset = await queue.get_latest_offset()
    return {"task_id": task_id, "offset": offset}
