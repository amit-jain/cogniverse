"""
Real integration tests for InMemoryEventQueue enqueue + subscribe round-trip.

InMemoryEventQueue IS the production implementation — not a test mock.
These tests exercise the full publish → consume path including cancellation.
"""

import asyncio
import logging

import pytest

from cogniverse_core.events.backends.memory import (
    InMemoryEventQueue,
    reset_queue_manager,
)
from cogniverse_core.events.types import (
    StatusEvent,
    TaskState,
    create_status_event,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration]


def _make_status_event(task_id: str, tenant_id: str, seq: int) -> StatusEvent:
    return create_status_event(
        task_id=task_id,
        tenant_id=tenant_id,
        state=TaskState.WORKING,
        phase=f"step_{seq}",
        message=f"Processing step {seq}",
    )


@pytest.fixture(autouse=True)
def reset_global_queue():
    """Ensure the global queue manager singleton doesn't leak between tests."""
    reset_queue_manager()
    yield
    reset_queue_manager()


@pytest.mark.asyncio
async def test_enqueue_and_subscribe_roundtrip():
    """Enqueue 3 events then subscribe — subscriber must receive all 3 in order."""
    queue = InMemoryEventQueue(task_id="task-rt-001", tenant_id="test")

    events_to_send = [_make_status_event("task-rt-001", "test", i) for i in range(3)]

    # Enqueue all events before subscribing (tests replay from offset 0)
    for evt in events_to_send:
        await queue.enqueue(evt)

    received = []
    async for event in queue.subscribe(from_offset=0):
        received.append(event)
        if len(received) == 3:
            break

    assert len(received) == 3, f"Expected 3 events, got {len(received)}"

    for i, (sent, got) in enumerate(zip(events_to_send, received)):
        assert got.event_id == sent.event_id, (
            f"Event {i}: expected event_id {sent.event_id!r}, got {got.event_id!r}"
        )
        assert got.task_id == "task-rt-001"
        assert got.tenant_id == "test"
        assert got.phase == f"step_{i}"


@pytest.mark.asyncio
async def test_enqueue_concurrent_producer_consumer():
    """Consumer starts before producer — must receive events as they arrive."""
    queue = InMemoryEventQueue(task_id="task-concurrent-001", tenant_id="test")
    received = []

    async def producer():
        for i in range(3):
            await asyncio.sleep(0.01)
            await queue.enqueue(_make_status_event("task-concurrent-001", "test", i))
        await queue.close()

    async def consumer():
        async for event in queue.subscribe(from_offset=0):
            received.append(event)

    await asyncio.gather(producer(), consumer())

    assert len(received) == 3, f"Expected 3 events, got {len(received)}"
    phases = [e.phase for e in received]
    assert phases == ["step_0", "step_1", "step_2"], f"Events out of order: {phases}"


@pytest.mark.asyncio
async def test_cancel_stops_subscriber():
    """Cancelling the queue must signal the subscriber to stop consuming."""
    queue = InMemoryEventQueue(task_id="task-cancel-001", tenant_id="test")
    received = []
    cancelled_signal = asyncio.Event()

    async def consumer():
        async for event in queue.subscribe(from_offset=0):
            received.append(event)
        cancelled_signal.set()

    # Start consumer in background
    consumer_task = asyncio.create_task(consumer())

    # Enqueue one event then cancel the queue
    await queue.enqueue(_make_status_event("task-cancel-001", "test", 0))
    await asyncio.sleep(0.05)  # Allow consumer to receive the first event
    await queue.close()

    # Wait for consumer to finish (max 2s)
    try:
        await asyncio.wait_for(cancelled_signal.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        consumer_task.cancel()
        pytest.fail("Consumer did not stop after queue.close() within 2 seconds")

    assert len(received) >= 1, "Consumer must have received at least the first event"
    assert queue.is_closed, "Queue must be marked closed"


@pytest.mark.asyncio
async def test_multiple_subscribers_all_receive():
    """Two subscribers on the same queue must both receive all enqueued events."""
    queue = InMemoryEventQueue(task_id="task-multi-001", tenant_id="test")

    # Pre-enqueue events
    for i in range(3):
        await queue.enqueue(_make_status_event("task-multi-001", "test", i))

    async def collect(from_offset=0):
        items = []
        async for event in queue.subscribe(from_offset=from_offset):
            items.append(event)
            if len(items) == 3:
                break
        return items

    results = await asyncio.gather(collect(0), collect(0))

    for sub_idx, items in enumerate(results):
        assert len(items) == 3, (
            f"Subscriber {sub_idx} expected 3 events, got {len(items)}"
        )
        for i, item in enumerate(items):
            assert item.phase == f"step_{i}", (
                f"Subscriber {sub_idx}, event {i}: phase mismatch. "
                f"Expected step_{i}, got {item.phase!r}"
            )
