"""
Unit tests for EventQueue system.

Tests:
- Event type serialization
- Queue operations (enqueue, subscribe, replay)
- Multiple subscribers
- Cancellation
- TTL expiration
- QueueManager lifecycle
"""

import asyncio
from datetime import datetime

import pytest

from cogniverse_core.events import (
    EventType,
    InMemoryEventQueue,
    InMemoryQueueManager,
    TaskState,
    create_artifact_event,
    create_complete_event,
    create_error_event,
    create_progress_event,
    create_status_event,
    reset_queue_manager,
)


class TestEventTypes:
    """Test event type creation and serialization"""

    @pytest.mark.ci_fast
    def test_status_event_creation(self):
        """Test StatusEvent factory function"""
        event = create_status_event(
            task_id="workflow_123",
            tenant_id="tenant1",
            state=TaskState.WORKING,
            phase="planning",
            message="Planning workflow",
        )

        assert event.task_id == "workflow_123"
        assert event.tenant_id == "tenant1"
        assert event.state == TaskState.WORKING
        assert event.phase == "planning"
        assert event.message == "Planning workflow"
        assert event.event_type == EventType.STATUS
        assert event.event_id.startswith("evt_")
        assert isinstance(event.timestamp, datetime)

    @pytest.mark.ci_fast
    def test_progress_event_creation(self):
        """Test ProgressEvent factory function"""
        event = create_progress_event(
            task_id="ingestion_456",
            tenant_id="tenant1",
            current=5,
            total=10,
            step="processing_video",
            details={"video": "test.mp4"},
        )

        assert event.task_id == "ingestion_456"
        assert event.tenant_id == "tenant1"
        assert event.current == 5
        assert event.total == 10
        assert event.percentage == 50.0
        assert event.step == "processing_video"
        assert event.details == {"video": "test.mp4"}
        assert event.event_type == EventType.PROGRESS

    @pytest.mark.ci_fast
    def test_artifact_event_creation(self):
        """Test ArtifactEvent factory function"""
        event = create_artifact_event(
            task_id="workflow_123",
            tenant_id="tenant1",
            artifact_type="search_result",
            data={"results": [1, 2, 3]},
            is_partial=True,
        )

        assert event.task_id == "workflow_123"
        assert event.artifact_type == "search_result"
        assert event.data == {"results": [1, 2, 3]}
        assert event.is_partial is True
        assert event.event_type == EventType.ARTIFACT

    @pytest.mark.ci_fast
    def test_error_event_creation(self):
        """Test ErrorEvent factory function"""
        event = create_error_event(
            task_id="workflow_123",
            tenant_id="tenant1",
            error_type="ValidationError",
            error_message="Invalid input",
            error_code="E001",
            context={"field": "query"},
            recoverable=True,
        )

        assert event.task_id == "workflow_123"
        assert event.error_type == "ValidationError"
        assert event.error_message == "Invalid input"
        assert event.error_code == "E001"
        assert event.context == {"field": "query"}
        assert event.recoverable is True
        assert event.event_type == EventType.ERROR

    @pytest.mark.ci_fast
    def test_complete_event_creation(self):
        """Test CompleteEvent factory function"""
        event = create_complete_event(
            task_id="workflow_123",
            tenant_id="tenant1",
            result={"answer": "42"},
            summary="Task completed successfully",
            execution_time_seconds=10.5,
        )

        assert event.task_id == "workflow_123"
        assert event.result == {"answer": "42"}
        assert event.summary == "Task completed successfully"
        assert event.execution_time_seconds == 10.5
        assert event.event_type == EventType.COMPLETE

    @pytest.mark.ci_fast
    def test_event_json_serialization(self):
        """Test event JSON serialization"""
        event = create_status_event(
            task_id="test",
            tenant_id="tenant1",
            state=TaskState.WORKING,
        )

        # Should serialize without error
        json_str = event.model_dump_json()
        assert "test" in json_str
        assert "tenant1" in json_str
        assert "working" in json_str


class TestInMemoryEventQueue:
    """Test InMemoryEventQueue operations"""

    @pytest.fixture
    def queue(self):
        """Create test queue"""
        return InMemoryEventQueue(
            task_id="test_task",
            tenant_id="test_tenant",
            ttl_minutes=30,
            max_buffer_size=100,
        )

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_enqueue_and_subscribe(self, queue):
        """Test basic enqueue and subscribe"""
        event = create_status_event(
            task_id="test_task",
            tenant_id="test_tenant",
            state=TaskState.WORKING,
        )

        # Enqueue event
        await queue.enqueue(event)

        # Subscribe and receive
        received_events = []
        async for evt in queue.subscribe():
            received_events.append(evt)
            break  # Get just one event

        assert len(received_events) == 1
        assert received_events[0].task_id == "test_task"
        assert received_events[0].state == TaskState.WORKING

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_multiple_events_ordering(self, queue):
        """Test event ordering is preserved"""
        events = []
        for i in range(5):
            event = create_progress_event(
                task_id="test_task",
                tenant_id="test_tenant",
                current=i,
                total=5,
            )
            events.append(event)
            await queue.enqueue(event)

        # Subscribe and verify ordering
        received = []
        async for evt in queue.subscribe():
            received.append(evt)
            if len(received) >= 5:
                break

        for i, evt in enumerate(received):
            assert evt.current == i

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_replay_from_offset(self, queue):
        """Test replay from specific offset"""
        # Enqueue 5 events
        for i in range(5):
            event = create_progress_event(
                task_id="test_task",
                tenant_id="test_tenant",
                current=i,
                total=5,
            )
            await queue.enqueue(event)

        # Subscribe from offset 3
        received = []
        async for evt in queue.subscribe(from_offset=3):
            received.append(evt)
            if len(received) >= 2:
                break

        # Should get events 3 and 4
        assert len(received) == 2
        assert received[0].current == 3
        assert received[1].current == 4

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_get_latest_offset(self, queue):
        """Test getting latest offset"""
        offset = await queue.get_latest_offset()
        assert offset == 0

        for i in range(3):
            event = create_status_event(
                task_id="test_task",
                tenant_id="test_tenant",
                state=TaskState.WORKING,
            )
            await queue.enqueue(event)

        offset = await queue.get_latest_offset()
        assert offset == 3

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_cancellation(self, queue):
        """Test cancellation token"""
        assert not queue.cancellation_token.is_cancelled

        queue.cancel("User requested")

        assert queue.cancellation_token.is_cancelled
        assert queue.cancellation_token.reason == "User requested"
        assert queue.cancellation_token.cancel_time is not None

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_close_queue(self, queue):
        """Test closing queue terminates subscribers"""
        # Start a subscriber in background
        received = []
        subscriber_done = asyncio.Event()

        async def subscriber():
            try:
                async for evt in queue.subscribe():
                    received.append(evt)
            finally:
                subscriber_done.set()

        task = asyncio.create_task(subscriber())

        # Enqueue an event
        event = create_status_event(
            task_id="test_task",
            tenant_id="test_tenant",
            state=TaskState.WORKING,
        )
        await queue.enqueue(event)
        await asyncio.sleep(0.1)  # Let subscriber process

        # Close queue
        await queue.close()
        await asyncio.wait_for(subscriber_done.wait(), timeout=2.0)

        assert queue.is_closed
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, queue):
        """Test multiple subscribers receive same events"""
        subscriber1_events = []
        subscriber2_events = []

        async def subscriber1():
            async for evt in queue.subscribe():
                subscriber1_events.append(evt)
                if len(subscriber1_events) >= 3:
                    break

        async def subscriber2():
            async for evt in queue.subscribe():
                subscriber2_events.append(evt)
                if len(subscriber2_events) >= 3:
                    break

        # Start subscribers
        task1 = asyncio.create_task(subscriber1())
        task2 = asyncio.create_task(subscriber2())
        await asyncio.sleep(0.1)  # Let subscribers start

        # Enqueue events
        for i in range(3):
            event = create_progress_event(
                task_id="test_task",
                tenant_id="test_tenant",
                current=i,
                total=3,
            )
            await queue.enqueue(event)

        # Wait for subscribers
        await asyncio.gather(task1, task2)

        # Both should have all events
        assert len(subscriber1_events) == 3
        assert len(subscriber2_events) == 3

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_backpressure_drops_oldest(self):
        """Test that oldest events are dropped when buffer is full"""
        queue = InMemoryEventQueue(
            task_id="test_task",
            tenant_id="test_tenant",
            max_buffer_size=5,
        )

        # Enqueue more than buffer size
        for i in range(10):
            event = create_progress_event(
                task_id="test_task",
                tenant_id="test_tenant",
                current=i,
                total=10,
            )
            await queue.enqueue(event)

        # Buffer should only have last 5
        offset = await queue.get_latest_offset()
        assert offset == 5  # 5 events in buffer

    @pytest.mark.ci_fast
    def test_queue_stats(self, queue):
        """Test queue statistics"""
        stats = queue.get_stats()

        assert stats["task_id"] == "test_task"
        assert stats["tenant_id"] == "test_tenant"
        assert stats["event_count"] == 0
        assert stats["subscriber_count"] == 0
        assert stats["is_closed"] is False
        assert stats["is_cancelled"] is False

    @pytest.mark.ci_fast
    def test_queue_expiration(self):
        """Test queue expiration check"""
        # Create queue with 0 TTL (immediately expired)
        queue = InMemoryEventQueue(
            task_id="test_task",
            tenant_id="test_tenant",
            ttl_minutes=0,
        )

        # Queue should be expired
        assert queue.is_expired


class TestInMemoryQueueManager:
    """Test InMemoryQueueManager operations"""

    @pytest.fixture
    def manager(self):
        """Create test manager"""
        reset_queue_manager()
        return InMemoryQueueManager(default_ttl_minutes=30)

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_create_queue(self, manager):
        """Test queue creation"""
        queue = await manager.create_queue(
            task_id="task_123",
            tenant_id="tenant1",
        )

        assert queue.task_id == "task_123"
        assert queue.tenant_id == "tenant1"

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_create_duplicate_queue_raises(self, manager):
        """Test creating duplicate queue raises error"""
        await manager.create_queue(
            task_id="task_123",
            tenant_id="tenant1",
        )

        with pytest.raises(ValueError, match="Queue already exists"):
            await manager.create_queue(
                task_id="task_123",
                tenant_id="tenant1",
            )

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_get_queue(self, manager):
        """Test getting existing queue"""
        await manager.create_queue(
            task_id="task_123",
            tenant_id="tenant1",
        )

        queue = await manager.get_queue("task_123")
        assert queue is not None
        assert queue.task_id == "task_123"

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_get_nonexistent_queue(self, manager):
        """Test getting nonexistent queue returns None"""
        queue = await manager.get_queue("nonexistent")
        assert queue is None

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_get_or_create_queue(self, manager):
        """Test get_or_create returns existing or creates new"""
        # First call creates
        queue1 = await manager.get_or_create_queue(
            task_id="task_123",
            tenant_id="tenant1",
        )
        assert queue1 is not None

        # Second call returns existing
        queue2 = await manager.get_or_create_queue(
            task_id="task_123",
            tenant_id="tenant1",
        )
        assert queue1 is queue2

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_close_queue(self, manager):
        """Test closing and removing queue"""
        await manager.create_queue(
            task_id="task_123",
            tenant_id="tenant1",
        )

        result = await manager.close_queue("task_123")
        assert result is True

        queue = await manager.get_queue("task_123")
        assert queue is None

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_close_nonexistent_queue(self, manager):
        """Test closing nonexistent queue returns False"""
        result = await manager.close_queue("nonexistent")
        assert result is False

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_list_active_queues(self, manager):
        """Test listing active queues"""
        await manager.create_queue("task_1", "tenant1")
        await manager.create_queue("task_2", "tenant1")
        await manager.create_queue("task_3", "tenant2")

        # List all
        all_queues = await manager.list_active_queues()
        assert len(all_queues) == 3

        # Filter by tenant
        tenant1_queues = await manager.list_active_queues(tenant_id="tenant1")
        assert len(tenant1_queues) == 2

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, manager):
        """Test cleanup of expired queues"""
        # Create queue with 0 TTL
        queue = InMemoryEventQueue(
            task_id="expired_task",
            tenant_id="tenant1",
            ttl_minutes=0,
        )
        manager._queues["expired_task"] = queue

        # Create normal queue
        await manager.create_queue("active_task", "tenant1")

        count = await manager.cleanup_expired()
        assert count == 1

        # Only active queue should remain
        queues = await manager.list_active_queues()
        assert len(queues) == 1
        assert queues[0]["task_id"] == "active_task"

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_cancel_task(self, manager):
        """Test task cancellation via manager"""
        await manager.create_queue("task_123", "tenant1")

        result = await manager.cancel_task("task_123", "Test reason")
        assert result is True

        queue = await manager.get_queue("task_123")
        assert queue.cancellation_token.is_cancelled
        assert queue.cancellation_token.reason == "Test reason"

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, manager):
        """Test cancelling nonexistent task returns False"""
        result = await manager.cancel_task("nonexistent")
        assert result is False

    @pytest.mark.ci_fast
    def test_manager_stats(self, manager):
        """Test manager statistics"""
        stats = manager.get_manager_stats()

        assert stats["total_queues"] == 0
        assert stats["active_queues"] == 0
        assert stats["default_ttl_minutes"] == 30


class TestMultiTenantIsolation:
    """Test multi-tenant queue isolation"""

    @pytest.fixture
    def manager(self):
        reset_queue_manager()
        return InMemoryQueueManager()

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_tenant_isolation_in_list(self, manager):
        """Test that queues are isolated by tenant in listing"""
        await manager.create_queue("task_a1", "tenant_a")
        await manager.create_queue("task_a2", "tenant_a")
        await manager.create_queue("task_b1", "tenant_b")

        tenant_a_queues = await manager.list_active_queues(tenant_id="tenant_a")
        tenant_b_queues = await manager.list_active_queues(tenant_id="tenant_b")

        assert len(tenant_a_queues) == 2
        assert len(tenant_b_queues) == 1
        assert all(q["tenant_id"] == "tenant_a" for q in tenant_a_queues)
        assert all(q["tenant_id"] == "tenant_b" for q in tenant_b_queues)

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_events_include_tenant_id(self, manager):
        """Test that events include tenant_id"""
        queue = await manager.create_queue("task_123", "my_tenant")

        event = create_status_event(
            task_id="task_123",
            tenant_id="my_tenant",
            state=TaskState.WORKING,
        )
        await queue.enqueue(event)

        async for received in queue.subscribe():
            assert received.tenant_id == "my_tenant"
            break
