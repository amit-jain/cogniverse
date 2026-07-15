"""
Integration tests for EventQueue with Orchestrator and Ingestion Pipeline.

Tests that events are properly emitted during workflow execution and
video processing.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.events import (
    InMemoryEventQueue,
    InMemoryQueueManager,
    TaskState,
    create_status_event,
    reset_queue_manager,
)


@pytest.fixture
def queue_manager():
    """Create fresh queue manager for each test."""
    reset_queue_manager()
    return InMemoryQueueManager(default_ttl_minutes=30)


@pytest.fixture
def event_queue(queue_manager):
    """Create event queue for testing."""
    return asyncio.run(
        queue_manager.create_queue(
            task_id="test_workflow_123",
            tenant_id="test_tenant",
        )
    )


class TestEventQueueWithIngestionPipeline:
    """Test EventQueue integration with VideoIngestionPipeline."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock config manager."""
        manager = MagicMock()
        manager.get_config.return_value = {
            "pipeline": {
                "video_dir": "/tmp/videos",
                "output_dir": "/tmp/output",
            },
            "backend": {
                "type": "vespa",
            },
        }
        return manager

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_pipeline_accepts_event_queue_parameter(
        self, mock_config_manager, tmp_path
    ):
        """Test that pipeline accepts event_queue parameter."""
        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        queue = InMemoryEventQueue(
            task_id="test_ingestion",
            tenant_id="test_tenant",
        )

        config = PipelineConfig(
            video_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

        # Mock the backend initialization to avoid actual backend connection
        with patch.object(VideoIngestionPipeline, "_init_backend"):
            with patch.object(VideoIngestionPipeline, "_resolve_strategy"):
                with patch.object(
                    VideoIngestionPipeline,
                    "_create_strategy_set_from_config",
                    return_value=MagicMock(),
                ):
                    pipeline = VideoIngestionPipeline(
                        tenant_id="test_tenant",
                        config=config,
                        config_manager=mock_config_manager,
                        event_queue=queue,
                    )

                    assert pipeline.event_queue is queue
                    assert pipeline.tenant_id == "test_tenant"

                    # The accepted queue must be wired for emission, not just
                    # stored — an event emitted by the pipeline lands in it.
                    pipeline.job_id = "test_job_123"
                    await pipeline._emit_event(
                        create_status_event(
                            task_id="test_job_123",
                            tenant_id="test_tenant",
                            state=TaskState.WORKING,
                        )
                    )
                    assert await queue.get_latest_offset() == 1

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_pipeline_emit_event_helper(self, mock_config_manager, tmp_path):
        """Test that _emit_event helper works correctly."""
        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        queue = InMemoryEventQueue(
            task_id="test_ingestion",
            tenant_id="test_tenant",
        )

        config = PipelineConfig(
            video_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

        with patch.object(VideoIngestionPipeline, "_init_backend"):
            with patch.object(VideoIngestionPipeline, "_resolve_strategy"):
                with patch.object(
                    VideoIngestionPipeline,
                    "_create_strategy_set_from_config",
                    return_value=MagicMock(),
                ):
                    pipeline = VideoIngestionPipeline(
                        tenant_id="test_tenant",
                        config=config,
                        config_manager=mock_config_manager,
                        event_queue=queue,
                    )

                    # Set job_id to enable event emission
                    pipeline.job_id = "test_job_123"

                    event = create_status_event(
                        task_id="test_job_123",
                        tenant_id="test_tenant",
                        state=TaskState.WORKING,
                    )

                    await pipeline._emit_event(event)

                    # Verify event was enqueued
                    offset = await queue.get_latest_offset()
                    assert offset == 1

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_pipeline_cancellation_check(self, mock_config_manager, tmp_path):
        """Test that pipeline respects cancellation."""
        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        queue = InMemoryEventQueue(
            task_id="test_ingestion",
            tenant_id="test_tenant",
        )

        config = PipelineConfig(
            video_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

        with patch.object(VideoIngestionPipeline, "_init_backend"):
            with patch.object(VideoIngestionPipeline, "_resolve_strategy"):
                with patch.object(
                    VideoIngestionPipeline,
                    "_create_strategy_set_from_config",
                    return_value=MagicMock(),
                ):
                    pipeline = VideoIngestionPipeline(
                        tenant_id="test_tenant",
                        config=config,
                        config_manager=mock_config_manager,
                        event_queue=queue,
                    )

                    # Initially not cancelled
                    assert not pipeline._is_cancelled()

                    # Cancel the queue
                    queue.cancel("Test cancellation")

                    # Now should be cancelled
                    assert pipeline._is_cancelled()


class TestEventQueueMultipleSubscribers:
    """Test multiple subscribers receiving events."""

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_dashboard_and_cli_both_receive_events(self):
        """Simulate dashboard and CLI subscribing to same workflow."""
        queue = InMemoryEventQueue(
            task_id="workflow_shared",
            tenant_id="test_tenant",
        )

        dashboard_events = []
        cli_events = []

        async def dashboard_subscriber():
            async for event in queue.subscribe():
                dashboard_events.append(event)
                if len(dashboard_events) >= 3:
                    break

        async def cli_subscriber():
            async for event in queue.subscribe():
                cli_events.append(event)
                if len(cli_events) >= 3:
                    break

        # Start both subscribers
        task1 = asyncio.create_task(dashboard_subscriber())
        task2 = asyncio.create_task(cli_subscriber())

        await asyncio.sleep(0.1)  # Let subscribers start

        # Emit events
        for i in range(3):
            event = create_status_event(
                task_id="workflow_shared",
                tenant_id="test_tenant",
                state=TaskState.WORKING,
                phase=f"phase_{i}",
            )
            await queue.enqueue(event)

        # Wait for subscribers
        await asyncio.gather(task1, task2)

        # Both should have received all events
        assert len(dashboard_events) == 3
        assert len(cli_events) == 3

        # Same events in same order
        for i in range(3):
            assert dashboard_events[i].phase == f"phase_{i}"
            assert cli_events[i].phase == f"phase_{i}"


class TestEventQueueReconnection:
    """Test client reconnection with replay."""

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_client_reconnect_from_offset(self):
        """Test client can reconnect and resume from offset."""
        queue = InMemoryEventQueue(
            task_id="workflow_reconnect",
            tenant_id="test_tenant",
        )

        # Emit 5 events
        for i in range(5):
            event = create_status_event(
                task_id="workflow_reconnect",
                tenant_id="test_tenant",
                state=TaskState.WORKING,
                phase=f"phase_{i}",
            )
            await queue.enqueue(event)

        # First client receives first 2 events
        first_client_events = []
        async for event in queue.subscribe(from_offset=0):
            first_client_events.append(event)
            if len(first_client_events) >= 2:
                break

        assert len(first_client_events) == 2
        assert first_client_events[0].phase == "phase_0"
        assert first_client_events[1].phase == "phase_1"

        # Client "disconnects" and reconnects from offset 2
        reconnect_events = []
        async for event in queue.subscribe(from_offset=2):
            reconnect_events.append(event)
            if len(reconnect_events) >= 3:
                break

        # Should get events 2, 3, 4
        assert len(reconnect_events) == 3
        assert reconnect_events[0].phase == "phase_2"
        assert reconnect_events[1].phase == "phase_3"
        assert reconnect_events[2].phase == "phase_4"
