# A2A EventQueue System

Real-time event notifications for orchestrator workflows and ingestion pipelines.

## Overview

The EventQueue system provides A2A-compatible real-time progress notifications. It enables:

- **Multiple subscribers**: Dashboard + CLI can watch the same workflow simultaneously
- **Replay on reconnect**: Clients can resume from a specific offset after disconnection
- **Graceful cancellation**: Tasks abort at phase/video boundaries
- **Multi-tenant isolation**: Events are scoped by tenant_id

## Architecture

```mermaid
flowchart TB
    subgraph QueueManager["<span style='color:#000'>QueueManager<br/>(Manages lifecycle of EventQueues)</span>"]
        Queue1["<span style='color:#000'>EventQueue<br/>workflow_123</span>"]
        Queue2["<span style='color:#000'>EventQueue<br/>workflow_456</span>"]
        Queue3["<span style='color:#000'>EventQueue<br/>ingestion_789</span>"]

        Sub1["<span style='color:#000'>subscribe()<br/>(Dashboard)</span>"]
        Sub2["<span style='color:#000'>subscribe()<br/>(CLI)</span>"]
        Sub3["<span style='color:#000'>subscribe()<br/>(Monitor)</span>"]

        Queue1 -- "AsyncIterator" --> Sub1
        Queue2 -- "AsyncIterator" --> Sub2
        Queue3 -- "AsyncIterator" --> Sub3
    end

    style QueueManager fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Queue1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style Queue2 fill:#ffcc80,stroke:#ef6c00,color:#000
    style Queue3 fill:#ffcc80,stroke:#ef6c00,color:#000
    style Sub1 fill:#90caf9,stroke:#1565c0,color:#000
    style Sub2 fill:#90caf9,stroke:#1565c0,color:#000
    style Sub3 fill:#90caf9,stroke:#1565c0,color:#000
```

## Event Types

### StatusEvent
Task state transitions (A2A-compatible):
```python
StatusEvent(
    task_id="workflow_123",
    tenant_id="tenant1",
    state=TaskState.WORKING,  # pending, working, input-required, completed, failed, cancelled
    phase="planning",
    message="Planning workflow execution",
)
```

### ProgressEvent
Incremental progress updates:
```python
ProgressEvent(
    task_id="ingestion_456",
    tenant_id="tenant1",
    current=5,
    total=10,
    percentage=50.0,
    step="processing_video_5",
    details={"video": "sample.mp4"},
)
```

### ArtifactEvent
Intermediate results (A2A TaskArtifactUpdateEvent):
```python
ArtifactEvent(
    task_id="workflow_123",
    tenant_id="tenant1",
    artifact_type="search_result",
    data={"results": [...]},
    is_partial=True,
)
```

### ErrorEvent
Error notifications:
```python
ErrorEvent(
    task_id="workflow_123",
    tenant_id="tenant1",
    error_type="ValidationError",
    error_message="Invalid input",
    recoverable=True,
)
```

### CompleteEvent
Task completion:
```python
CompleteEvent(
    task_id="workflow_123",
    tenant_id="tenant1",
    result={"answer": "..."},
    summary="Workflow completed successfully",
    execution_time_seconds=10.5,
)
```

## Usage

### Basic Usage

```python
from cogniverse_core.events import (
    get_queue_manager,
    create_status_event,
    create_progress_event,
    TaskState,
)

# Get the global queue manager
manager = get_queue_manager()

# Create a queue for a workflow
queue = await manager.create_queue(
    task_id="workflow_123",
    tenant_id="tenant1",
    ttl_minutes=30,
)

# Emit events
await queue.enqueue(create_status_event(
    task_id="workflow_123",
    tenant_id="tenant1",
    state=TaskState.WORKING,
    phase="planning",
))

# Subscribe to events (in another coroutine)
async for event in queue.subscribe():
    print(f"Received: {event.event_type}")
    if event.event_type == "complete":
        break
```

### With Orchestrator

```python
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.events import get_queue_manager
from cogniverse_foundation.config.utils import create_default_config_manager

# Create event queue for this workflow
manager = get_queue_manager()
workflow_id = "workflow_123"
queue = await manager.create_queue(
    task_id=workflow_id,
    tenant_id="tenant1",
)

# Create orchestrator with event queue (config_manager and registry are REQUIRED)
config_manager = create_default_config_manager()
registry = AgentRegistry(tenant_id="tenant1", config_manager=config_manager)
deps = OrchestratorDeps()
orchestrator = OrchestratorAgent(
    deps=deps,
    registry=registry,
    config_manager=config_manager,
    event_queue=queue,  # forwarded to checkpoint saves + the sufficiency-gate RLM
)

# Process query via A2A task protocol. If checkpoint_storage is configured,
# checkpoint saves after each task-group batch emit Status/ProgressEvent;
# the sufficiency-gate InstrumentedRLM emits events once evidence crosses
# the RLM promotion threshold. See "How It Works" below for the full picture.
input_data = OrchestratorInput(query="Find videos about cats", tenant_id="tenant1")
result = await orchestrator._process_impl(input_data)
```

### With Ingestion Pipeline

```python
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_core.events import get_queue_manager
from cogniverse_foundation.config.utils import create_default_config_manager

# Create event queue for ingestion job
manager = get_queue_manager()
job_id = "ingestion_456"
queue = await manager.create_queue(
    task_id=job_id,
    tenant_id="tenant1",
)

# Create pipeline with event queue
config_manager = create_default_config_manager()
pipeline = VideoIngestionPipeline(
    tenant_id="tenant1",
    config_manager=config_manager,
    event_queue=queue,  # Events emitted automatically during processing
)

# Process videos - pipeline.job_id is set during execution
result = await pipeline.process_videos_concurrent(video_files)
# Subscribe to events using the job_id from queue or pipeline.job_id
```

### With RLM Inference

Agents that mix in `RLMAwareMixin` (Recursive Language Model processing for
oversized contexts) can forward an `EventQueue` into `get_rlm()`. When an
`event_queue` is provided, `RLMInference` swaps in `InstrumentedRLM`, which
emits `StatusEvent`/`ProgressEvent` per REPL iteration and checks the
queue's `CancellationToken` between iterations:

```python
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_core.events import get_queue_manager

manager = get_queue_manager()
task_id = "search_agent_task_1"
queue = await manager.create_queue(task_id=task_id, tenant_id="tenant1")
llm_config = LLMEndpointConfig(model="openai/gpt-4o")

# Inside an agent mixing in RLMAwareMixin
rlm = self.get_rlm(
    llm_config=llm_config,
    max_iterations=10,
    event_queue=queue,
    task_id=task_id,
    tenant_id="tenant1",
)
result = rlm.process(query="Summarize the main findings", context=large_context_string)
```

`event_queue`/`task_id`/`tenant_id` are optional on `get_rlm()` /
`RLMInference` / `InstrumentedRLM` — when omitted, RLM behaves like plain
`dspy.RLM` with no event emission. When `event_queue` is provided, `tenant_id`
becomes required (both raise `ValueError` otherwise, since RLM events must be
tenant-scoped).

### SSE Streaming (HTTP Clients)

```bash
# Subscribe to workflow events
curl -N "http://localhost:8000/events/workflows/workflow_123"

# Subscribe to ingestion events
curl -N "http://localhost:8000/events/ingestion/ingestion_456"

# Cancel a workflow
curl -X POST "http://localhost:8000/events/workflows/workflow_123/cancel" \
  -H "Content-Type: application/json" \
  -d '{"reason": "User requested"}'
```

### Reconnection with Replay

```python
# First connection - get offset
last_offset = 0
async for event in queue.subscribe():
    last_offset = await queue.get_latest_offset()
    # ... process event ...
    if connection_lost:
        break

# Reconnect from last offset
async for event in queue.subscribe(from_offset=last_offset):
    # ... resume processing ...
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events/workflows/{workflow_id}` | GET | SSE stream of workflow events |
| `/events/ingestion/{job_id}` | GET | SSE stream of ingestion events |
| `/events/workflows/{workflow_id}/cancel` | POST | Cancel running workflow |
| `/events/ingestion/{job_id}/cancel` | POST | Cancel running ingestion |
| `/events/queues` | GET | List active queues (admin) |
| `/events/queues/{task_id}` | GET | Get queue info |
| `/events/queues/{task_id}/offset` | GET | Get current event offset |

## Configuration

### Queue Manager

```python
manager = InMemoryQueueManager(
    default_ttl_minutes=30,    # Event TTL (default 30 min)
    max_buffer_size=1000,      # Max events per queue
)

# Start cleanup loop (optional)
await manager.start_cleanup_loop(interval_seconds=60)
```

`cogniverse_runtime.main` starts this cleanup loop automatically on app
startup via `get_queue_manager()` + `start_cleanup_loop(interval_seconds=60)`.
Every search/ingestion/mem0 operation creates a queue holding up to
`max_buffer_size` events; without the loop, closed/expired queues are never
evicted and the runtime accumulates buffers until it OOMs.

### Queue Options

```python
queue = await manager.create_queue(
    task_id="workflow_123",
    tenant_id="tenant1",
    ttl_minutes=60,  # Override default TTL
)
```

## Cancellation

```python
# Via queue manager
await manager.cancel_task("workflow_123", reason="User requested")

# Via queue directly
queue.cancel("Timeout exceeded")

# Check in producer
if queue.cancellation_token.is_cancelled:
    # Clean up and stop
    pass
```

## Backend Options

### In-Memory (Default)
- Single-pod deployments
- Development/testing
- No persistence (events lost on restart)

### Redis Pub/Sub (Future)
- Multi-pod production
- No long-term persistence needed
- Fast real-time notification

### Related but Separate: Queue-Driven Ingestion Events

When `REDIS_URL` is set, `/ingestion/{ingest_id}/events` (mounted from
`cogniverse_runtime.ingestion_worker.status_api`) streams SSE events for the
Redis-queue-backed ingestion path from a Redis stream directly — it does
**not** use `cogniverse_core.events`/`EventQueue`/`InMemoryQueueManager` at
all. It is a distinct mechanism with its own replay semantics
(`last-event-id` query param, not `from_offset`). Don't confuse it with the
`/events/ingestion/{job_id}` endpoint documented above, which streams from an
in-memory `EventQueue` for the in-process (non-Redis) ingestion pipeline.

## Checkpoint-Event Integration

Checkpoint saves automatically emit A2A events when an EventQueue is provided. This unifies state persistence with real-time notifications:

```python
from cogniverse_agents.orchestrator.checkpoint_storage import WorkflowCheckpointStorage
from cogniverse_core.events import get_queue_manager

manager = get_queue_manager()
queue = await manager.create_queue("workflow_123", "tenant1")

# Create checkpoint storage with event queue
checkpoint_storage = WorkflowCheckpointStorage(
    grpc_endpoint="localhost:4317",
    http_endpoint="http://localhost:6006",
    tenant_id="tenant1",
    event_queue=queue,  # Events emitted automatically on checkpoint save
)

# When a checkpoint is saved, these events are automatically emitted:
# 1. StatusEvent - state change notification
# 2. ProgressEvent - task completion progress (if tasks exist)
```

### How It Works

1. **Checkpoint saves** emit A2A-compatible status/progress events automatically (`WorkflowCheckpointStorage._emit_checkpoint_event`) — this is the primary source of workflow-level `StatusEvent`/`ProgressEvent` notifications today.
2. **Orchestrator** forwards its `event_queue` into the sufficient-context-gate `InstrumentedRLM` (used once accumulated evidence crosses the RLM promotion threshold in the iterative retrieval loop), which emits `StatusEvent`/`ProgressEvent` per REPL iteration. The orchestrator also signals `event_queue.cancel()` when an inbound `"stop"` message is drained, so any `InstrumentedRLM` running inside a sub-agent's chain observes cancellation at its next iteration.
3. **`OrchestratorAgent._emit_event()`** is a generic hook (`enqueue` if `event_queue` is configured, no-op otherwise) available for emitting ad-hoc events, but it is not currently called from the planning/execution/completion boundaries of `_process_impl` — checkpoint saves and the sufficiency-gate RLM are the actual emitters.
4. **No duplicate code paths** - state changes trigger notifications automatically

### Event Flow

```mermaid
flowchart TD
    Start["<span style='color:#000'>Workflow Execution</span>"]

    Iteration["<span style='color:#000'>Iterative retrieval loop<br/>(sufficiency gate)</span>"]
    RLMPromotion{"<span style='color:#000'>Evidence exceeds<br/>RLM promotion threshold?</span>"}
    RLMEvent["<span style='color:#000'>InstrumentedRLM emits<br/>StatusEvent + ProgressEvent<br/>per REPL iteration</span>"]

    TaskGroup1["<span style='color:#000'>First task group completes</span>"]
    TaskGroup1Event["<span style='color:#000'>Checkpoint saves<br/>StatusEvent + ProgressEvent</span>"]

    TaskGroup2["<span style='color:#000'>Second task group completes</span>"]
    TaskGroup2Event["<span style='color:#000'>Checkpoint saves<br/>StatusEvent + ProgressEvent</span>"]

    Completion["<span style='color:#000'>Completion</span>"]
    CompletionEvent1["<span style='color:#000'>Checkpoint saves<br/>StatusEvent(COMPLETED)</span>"]

    Start --> Iteration
    Iteration --> RLMPromotion
    RLMPromotion -- "yes" --> RLMEvent
    RLMEvent --> TaskGroup1
    RLMPromotion -- "no" --> TaskGroup1
    TaskGroup1 --> TaskGroup1Event
    TaskGroup1Event --> TaskGroup2
    TaskGroup2 --> TaskGroup2Event
    TaskGroup2Event --> Completion
    Completion --> CompletionEvent1

    style Start fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Iteration fill:#a5d6a7,stroke:#388e3c,color:#000
    style RLMPromotion fill:#a5d6a7,stroke:#388e3c,color:#000
    style RLMEvent fill:#ffcc80,stroke:#ef6c00,color:#000
    style TaskGroup1 fill:#81d4fa,stroke:#0288d1,color:#000
    style TaskGroup1Event fill:#ffcc80,stroke:#ef6c00,color:#000
    style TaskGroup2 fill:#81d4fa,stroke:#0288d1,color:#000
    style TaskGroup2Event fill:#ffcc80,stroke:#ef6c00,color:#000
    style Completion fill:#a5d6a7,stroke:#388e3c,color:#000
    style CompletionEvent1 fill:#ffcc80,stroke:#ef6c00,color:#000
```

## Relationship with Checkpoints

| Concern | Solution |
|---------|----------|
| Pod crashes mid-workflow | Checkpoints → resume from last state |
| Client wants progress updates | EventQueue → push notifications |
| Client disconnects/reconnects | EventQueue replay (short-term) |
| Dashboard shows live status | EventQueue subscription |

**Key Design Principle**: Checkpoints handle crash recovery (durable), EventQueue handles notifications (ephemeral). When both are configured together, checkpoint saves automatically emit events - single source of truth for state changes.

## Testing

```bash
# Run all event tests
uv run pytest tests/events/ -v

# Unit tests only
uv run pytest tests/events/unit/ -v

# Integration tests only
uv run pytest tests/events/integration/ -v
```

## Files

| File | Description |
|------|-------------|
| `libs/core/cogniverse_core/events/types.py` | Event type definitions |
| `libs/core/cogniverse_core/events/queue.py` | EventQueue/QueueManager protocols |
| `libs/core/cogniverse_core/events/backends/memory.py` | In-memory backend |
| `libs/runtime/cogniverse_runtime/routers/events.py` | SSE streaming endpoints |
| `libs/runtime/cogniverse_runtime/ingestion/pipeline.py` | `VideoIngestionPipeline` — emits events during video ingestion |
| `libs/agents/cogniverse_agents/orchestrator_agent.py` | `OrchestratorAgent` — accepts `event_queue`, emits workflow events |
| `libs/agents/cogniverse_agents/orchestrator/checkpoint_storage.py` | Checkpoint storage with event emission |
| `libs/agents/cogniverse_agents/mixins/rlm_aware_mixin.py` | `RLMAwareMixin.get_rlm()` — wires `event_queue` into RLM inference |
| `libs/agents/cogniverse_agents/inference/rlm_inference.py` | `RLMInference` — forwards `event_queue` to `InstrumentedRLM` |
| `libs/agents/cogniverse_agents/inference/instrumented_rlm.py` | `InstrumentedRLM` — emits Status/Progress events per REPL iteration |
| `tests/events/unit/test_event_queue.py` | Unit tests for event types, `InMemoryEventQueue`, `InMemoryQueueManager`, tenant isolation |
| `tests/events/unit/test_event_queue_integration.py` | Unit-level integration tests for queue wiring |
| `tests/events/integration/test_event_queue_real.py` | Real-boundary integration tests |
