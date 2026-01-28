"""
A2A-Compatible Event Queue System

Real-time progress notifications for orchestrator workflows and ingestion pipelines.

Components:
- types: Event type definitions (StatusEvent, ProgressEvent, etc.)
- queue: EventQueue and QueueManager protocols
- backends: Implementation backends (memory, redis)

Usage:
    from cogniverse_core.events import (
        # Event types
        TaskEvent,
        StatusEvent,
        ProgressEvent,
        ArtifactEvent,
        ErrorEvent,
        CompleteEvent,
        TaskState,
        EventType,
        # Factory functions
        create_status_event,
        create_progress_event,
        create_artifact_event,
        create_error_event,
        create_complete_event,
        # Queue interfaces
        EventQueue,
        QueueManager,
        CancellationToken,
        # In-memory backend
        get_queue_manager,
    )

    # Create queue manager
    manager = get_queue_manager()

    # Create queue for a workflow
    queue = await manager.create_queue(
        task_id="workflow_abc123",
        tenant_id="tenant1",
    )

    # Emit events
    await queue.enqueue(create_status_event(
        task_id="workflow_abc123",
        tenant_id="tenant1",
        state=TaskState.WORKING,
        phase="planning",
    ))

    # Subscribe to events (in another coroutine)
    async for event in queue.subscribe():
        print(f"Received: {event.event_type}")
"""

# Event types
# In-memory backend (default)
from cogniverse_core.events.backends.memory import (
    InMemoryEventQueue,
    InMemoryQueueManager,
    get_queue_manager,
    reset_queue_manager,
)

# Queue interfaces
from cogniverse_core.events.queue import (
    BaseEventQueue,
    BaseQueueManager,
    CancellationToken,
    EventQueue,
    QueueManager,
)
from cogniverse_core.events.types import (
    ArtifactEvent,
    BaseEvent,
    CompleteEvent,
    ErrorEvent,
    EventType,
    ProgressEvent,
    StatusEvent,
    TaskEvent,
    TaskEventProtocol,
    TaskState,
    create_artifact_event,
    create_complete_event,
    create_error_event,
    create_progress_event,
    create_status_event,
)

__all__ = [
    # Event types
    "TaskEvent",
    "TaskEventProtocol",
    "BaseEvent",
    "StatusEvent",
    "ProgressEvent",
    "ArtifactEvent",
    "ErrorEvent",
    "CompleteEvent",
    "TaskState",
    "EventType",
    # Factory functions
    "create_status_event",
    "create_progress_event",
    "create_artifact_event",
    "create_error_event",
    "create_complete_event",
    # Queue interfaces
    "EventQueue",
    "QueueManager",
    "BaseEventQueue",
    "BaseQueueManager",
    "CancellationToken",
    # In-memory backend
    "InMemoryEventQueue",
    "InMemoryQueueManager",
    "get_queue_manager",
    "reset_queue_manager",
]
