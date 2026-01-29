"""Tests for InstrumentedRLM with EventQueue integration.

Tests verify:
- Event emission at iteration boundaries
- Cancellation via CancellationToken
- Backward compatibility (works without event_queue)
- Integration with RLMInference
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_agents.inference import (
    InstrumentedRLM,
    RLMCancelledError,
    RLMInference,
)
from cogniverse_core.events.types import (
    EventType,
    TaskState,
)


class TestInstrumentedRLMBasics:
    """Basic tests for InstrumentedRLM initialization."""

    def test_init_without_event_queue(self):
        """InstrumentedRLM works without event_queue."""
        rlm = InstrumentedRLM(
            "context, query -> answer",
            max_iterations=5,
        )
        assert rlm.max_iterations == 5
        assert rlm._event_queue is None
        assert rlm._task_id is None

    def test_init_with_event_queue(self):
        """InstrumentedRLM accepts event_queue parameters."""
        mock_queue = MagicMock()
        rlm = InstrumentedRLM(
            "context, query -> answer",
            max_iterations=5,
            event_queue=mock_queue,
            task_id="task_123",
            tenant_id="tenant_1",
        )
        assert rlm._event_queue is mock_queue
        assert rlm._task_id == "task_123"
        assert rlm._tenant_id == "tenant_1"

    def test_default_tenant_id(self):
        """Default tenant_id is 'default'."""
        rlm = InstrumentedRLM(
            "context, query -> answer",
            tenant_id=None,
        )
        assert rlm._tenant_id == "default"


class TestRLMCancelledError:
    """Tests for RLMCancelledError exception."""

    def test_error_message(self):
        """RLMCancelledError has proper message."""
        error = RLMCancelledError("Custom message")
        assert str(error) == "Custom message"
        assert error.reason is None

    def test_error_with_reason(self):
        """RLMCancelledError stores reason."""
        error = RLMCancelledError("Cancelled", reason="User requested")
        assert error.reason == "User requested"


class TestInstrumentedRLMCancellation:
    """Tests for cancellation support."""

    def test_check_cancelled_no_queue(self):
        """_check_cancelled does nothing without event_queue."""
        rlm = InstrumentedRLM("context, query -> answer")
        # Should not raise
        rlm._check_cancelled()

    def test_check_cancelled_not_cancelled(self):
        """_check_cancelled does nothing when not cancelled."""
        mock_queue = MagicMock()
        mock_queue.cancellation_token.is_cancelled = False

        rlm = InstrumentedRLM(
            "context, query -> answer",
            event_queue=mock_queue,
            task_id="task_123",
        )
        # Should not raise
        rlm._check_cancelled()

    def test_check_cancelled_raises_when_cancelled(self):
        """_check_cancelled raises RLMCancelledError when cancelled."""
        mock_queue = MagicMock()
        mock_queue.cancellation_token.is_cancelled = True
        mock_queue.cancellation_token.reason = "User requested"

        rlm = InstrumentedRLM(
            "context, query -> answer",
            event_queue=mock_queue,
            task_id="task_123",
        )

        with pytest.raises(RLMCancelledError) as exc_info:
            rlm._check_cancelled()

        assert "User requested" in str(exc_info.value)
        assert exc_info.value.reason == "User requested"


class TestInstrumentedRLMEventEmission:
    """Tests for event emission logic."""

    def test_emit_sync_no_queue(self):
        """_emit_sync does nothing without event_queue."""
        rlm = InstrumentedRLM("context, query -> answer")
        # Should not raise
        rlm._emit_sync(MagicMock())

    def test_emit_sync_no_task_id(self):
        """_emit_sync does nothing without task_id."""
        mock_queue = MagicMock()
        rlm = InstrumentedRLM(
            "context, query -> answer",
            event_queue=mock_queue,
            task_id=None,
        )
        # Should not raise
        rlm._emit_sync(MagicMock())

    @pytest.mark.asyncio
    async def test_emit_sync_enqueues_event(self):
        """_emit_sync enqueues event in running loop."""
        mock_queue = AsyncMock()
        mock_event = MagicMock()

        rlm = InstrumentedRLM(
            "context, query -> answer",
            event_queue=mock_queue,
            task_id="task_123",
            tenant_id="tenant_1",
        )

        # Call _emit_sync in an async context
        rlm._emit_sync(mock_event)

        # Allow the task to complete
        await asyncio.sleep(0.01)

        # Verify enqueue was called
        mock_queue.enqueue.assert_called_once_with(mock_event)


class TestRLMInferenceWithEventQueue:
    """Tests for RLMInference using InstrumentedRLM."""

    def test_init_with_event_queue(self):
        """RLMInference accepts event_queue parameters."""
        mock_queue = MagicMock()
        rlm_inference = RLMInference(
            backend="openai",
            model="gpt-4o",
            max_iterations=5,
            event_queue=mock_queue,
            task_id="task_123",
            tenant_id="tenant_1",
        )
        assert rlm_inference._event_queue is mock_queue
        assert rlm_inference._task_id == "task_123"
        assert rlm_inference._tenant_id == "tenant_1"

    def test_get_rlm_without_event_queue(self):
        """_get_rlm returns standard RLM without event_queue."""
        rlm_inference = RLMInference(
            backend="openai",
            model="gpt-4o",
        )

        with patch("cogniverse_agents.inference.rlm_inference.dspy") as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm
            mock_dspy.RLM.return_value = MagicMock()

            rlm_inference._get_rlm()

            # Should use standard dspy.RLM
            mock_dspy.RLM.assert_called_once()

    def test_get_rlm_with_event_queue(self):
        """_get_rlm returns InstrumentedRLM with event_queue."""
        mock_queue = MagicMock()
        rlm_inference = RLMInference(
            backend="openai",
            model="gpt-4o",
            event_queue=mock_queue,
            task_id="task_123",
            tenant_id="tenant_1",
        )

        with patch("cogniverse_agents.inference.rlm_inference.dspy") as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            rlm = rlm_inference._get_rlm()

            # Should be InstrumentedRLM
            assert isinstance(rlm, InstrumentedRLM)
            assert rlm._event_queue is mock_queue
            assert rlm._task_id == "task_123"
            assert rlm._tenant_id == "tenant_1"


class TestRLMAwareMixinWithEventQueue:
    """Tests for RLMAwareMixin with event_queue."""

    def test_get_rlm_creates_new_instance_with_event_queue(self):
        """get_rlm creates new instance when event_queue provided."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        class TestAgent(RLMAwareMixin):
            tenant_id = "test_tenant"

        agent = TestAgent()
        mock_queue = MagicMock()

        # First call without event_queue
        rlm1 = agent.get_rlm(backend="openai", model="gpt-4o")

        # Second call with event_queue should create new instance
        rlm2 = agent.get_rlm(
            backend="openai",
            model="gpt-4o",
            event_queue=mock_queue,
            task_id="task_123",
        )

        # Should be different instances
        assert rlm1 is not rlm2
        assert rlm2._event_queue is mock_queue
        assert rlm2._task_id == "task_123"
        assert rlm2._tenant_id == "test_tenant"

    def test_get_rlm_uses_agent_tenant_id(self):
        """get_rlm uses agent's tenant_id if not specified."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        class TestAgent(RLMAwareMixin):
            tenant_id = "agent_tenant"

        agent = TestAgent()
        mock_queue = MagicMock()

        rlm = agent.get_rlm(
            backend="openai",
            model="gpt-4o",
            event_queue=mock_queue,
            task_id="task_123",
            # tenant_id not specified
        )

        assert rlm._tenant_id == "agent_tenant"


class TestEventTypes:
    """Tests verifying correct event types are used."""

    def test_status_event_has_working_state(self):
        """StatusEvent uses WORKING state for start."""
        from cogniverse_core.events.types import create_status_event

        event = create_status_event(
            task_id="task_123",
            tenant_id="tenant_1",
            state=TaskState.WORKING,
            phase="rlm_start",
        )

        assert event.event_type == EventType.STATUS
        assert event.state == TaskState.WORKING
        assert event.phase == "rlm_start"

    def test_progress_event_has_iteration_info(self):
        """ProgressEvent includes iteration information."""
        from cogniverse_core.events.types import create_progress_event

        event = create_progress_event(
            task_id="task_123",
            tenant_id="tenant_1",
            current=3,
            total=10,
            step="iteration_4",
            details={"iteration": 4},
        )

        assert event.event_type == EventType.PROGRESS
        assert event.current == 3
        assert event.total == 10
        assert event.percentage == 30.0
        assert event.step == "iteration_4"
        assert event.details == {"iteration": 4}
