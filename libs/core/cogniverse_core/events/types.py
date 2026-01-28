"""
Event Types for A2A-Compatible Task Notifications

Defines typed event structures for real-time progress notifications across
orchestrator workflows and ingestion pipelines. Designed for A2A protocol
interoperability.

Event Types:
- StatusEvent: Task state transitions (working, input-required, completed, failed)
- ProgressEvent: Incremental progress updates with percentage
- ArtifactEvent: Intermediate results/artifacts produced
- ErrorEvent: Error notifications with context
- CompleteEvent: Task completion with final result

Design Principles:
- Type-safe Pydantic models for validation
- Multi-tenant isolation via tenant_id
- Event ordering via event_id and timestamp
- A2A standard compatibility
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field


class TaskState(str, Enum):
    """A2A-compatible task states"""

    PENDING = "pending"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    """Event type discriminator"""

    STATUS = "status"
    PROGRESS = "progress"
    ARTIFACT = "artifact"
    ERROR = "error"
    COMPLETE = "complete"


class BaseEvent(BaseModel):
    """Base class for all task events"""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    task_id: str = Field(..., description="Workflow ID or Job ID")
    tenant_id: str = Field(
        ..., description="Tenant identifier for multi-tenant isolation"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: EventType

    class Config:
        use_enum_values = True


class StatusEvent(BaseEvent):
    """
    Task state transition event.

    Maps to A2A TaskStatusUpdateEvent for interoperability.
    """

    event_type: EventType = Field(default=EventType.STATUS)
    state: TaskState = Field(..., description="Current task state")
    phase: Optional[str] = Field(None, description="Current execution phase")
    message: Optional[str] = Field(None, description="Human-readable status message")

    class Config:
        use_enum_values = True


class ProgressEvent(BaseEvent):
    """
    Incremental progress update.

    Used for long-running operations like video processing.
    """

    event_type: EventType = Field(default=EventType.PROGRESS)
    current: int = Field(..., ge=0, description="Current progress value")
    total: int = Field(..., ge=0, description="Total progress value")
    percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    step: Optional[str] = Field(None, description="Current step name")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional progress details"
    )


class ArtifactEvent(BaseEvent):
    """
    Intermediate artifact/result event.

    Maps to A2A TaskArtifactUpdateEvent for interoperability.
    Used for streaming partial results.
    """

    event_type: EventType = Field(default=EventType.ARTIFACT)
    artifact_type: str = Field(
        ..., description="Type of artifact (e.g., 'search_result', 'summary')"
    )
    artifact_id: str = Field(default_factory=lambda: f"art_{uuid.uuid4().hex[:8]}")
    data: Dict[str, Any] = Field(..., description="Artifact payload")
    is_partial: bool = Field(
        False, description="Whether this is a partial/streaming result"
    )


class ErrorEvent(BaseEvent):
    """
    Error notification with context.

    Provides detailed error information for debugging.
    """

    event_type: EventType = Field(default=EventType.ERROR)
    error_type: str = Field(..., description="Error classification")
    error_message: str = Field(..., description="Human-readable error message")
    error_code: Optional[str] = Field(
        None, description="Error code for programmatic handling"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Error context/stack trace"
    )
    recoverable: bool = Field(True, description="Whether the error is recoverable")


class CompleteEvent(BaseEvent):
    """
    Task completion event with final result.

    Signals successful task completion.
    """

    event_type: EventType = Field(default=EventType.COMPLETE)
    result: Dict[str, Any] = Field(..., description="Final result payload")
    summary: Optional[str] = Field(
        None, description="Human-readable completion summary"
    )
    execution_time_seconds: Optional[float] = Field(
        None, description="Total execution time"
    )


# Union type for all events
TaskEvent = Union[StatusEvent, ProgressEvent, ArtifactEvent, ErrorEvent, CompleteEvent]


@runtime_checkable
class TaskEventProtocol(Protocol):
    """Protocol for task events - for type checking"""

    event_id: str
    task_id: str
    tenant_id: str
    timestamp: datetime
    event_type: EventType


def create_status_event(
    task_id: str,
    tenant_id: str,
    state: TaskState,
    phase: Optional[str] = None,
    message: Optional[str] = None,
) -> StatusEvent:
    """Factory function to create StatusEvent"""
    return StatusEvent(
        task_id=task_id,
        tenant_id=tenant_id,
        state=state,
        phase=phase,
        message=message,
    )


def create_progress_event(
    task_id: str,
    tenant_id: str,
    current: int,
    total: int,
    step: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> ProgressEvent:
    """Factory function to create ProgressEvent"""
    percentage = (current / total * 100) if total > 0 else 0
    return ProgressEvent(
        task_id=task_id,
        tenant_id=tenant_id,
        current=current,
        total=total,
        percentage=percentage,
        step=step,
        details=details,
    )


def create_artifact_event(
    task_id: str,
    tenant_id: str,
    artifact_type: str,
    data: Dict[str, Any],
    is_partial: bool = False,
) -> ArtifactEvent:
    """Factory function to create ArtifactEvent"""
    return ArtifactEvent(
        task_id=task_id,
        tenant_id=tenant_id,
        artifact_type=artifact_type,
        data=data,
        is_partial=is_partial,
    )


def create_error_event(
    task_id: str,
    tenant_id: str,
    error_type: str,
    error_message: str,
    error_code: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    recoverable: bool = True,
) -> ErrorEvent:
    """Factory function to create ErrorEvent"""
    return ErrorEvent(
        task_id=task_id,
        tenant_id=tenant_id,
        error_type=error_type,
        error_message=error_message,
        error_code=error_code,
        context=context,
        recoverable=recoverable,
    )


def create_complete_event(
    task_id: str,
    tenant_id: str,
    result: Dict[str, Any],
    summary: Optional[str] = None,
    execution_time_seconds: Optional[float] = None,
) -> CompleteEvent:
    """Factory function to create CompleteEvent"""
    return CompleteEvent(
        task_id=task_id,
        tenant_id=tenant_id,
        result=result,
        summary=summary,
        execution_time_seconds=execution_time_seconds,
    )
