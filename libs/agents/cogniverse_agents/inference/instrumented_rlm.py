"""Instrumented RLM with EventQueue integration for real-time progress.

Provides InstrumentedRLM, a subclass of dspy.RLM that emits events
at each iteration for real-time progress tracking and cancellation support.

Usage:
    from cogniverse_agents.inference import InstrumentedRLM

    # With event queue for real-time progress
    rlm = InstrumentedRLM(
        "context, query -> answer",
        event_queue=queue,
        task_id="task_123",
        tenant_id="tenant_1",
        max_iterations=10,
    )
    result = rlm(context=large_context, query="Summarize this")

    # Without event queue (behaves like standard dspy.RLM)
    rlm = InstrumentedRLM("context, query -> answer", max_iterations=10)
    result = rlm(context=large_context, query="Summarize this")
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory

if TYPE_CHECKING:
    from cogniverse_core.events import EventQueue

from cogniverse_agents.inference.tolerant_interpreter import TolerantRLM
from cogniverse_core.events.types import (
    TaskState,
    create_progress_event,
    create_status_event,
)

logger = logging.getLogger(__name__)


class RLMCancelledError(Exception):
    """Raised when RLM is cancelled via CancellationToken."""

    def __init__(
        self, message: str = "RLM cancelled by user", reason: Optional[str] = None
    ):
        self.reason = reason
        super().__init__(message)


def _mark_fallback(prediction: Prediction) -> None:
    """Tag a Prediction returned via fallback extraction so callers can detect it.

    Why: when max_iterations is exhausted without a SUBMIT(), the parent class
    falls back to best-effort extraction. Answer quality may be lower; callers
    (RLMInference, agents) need a signal so they can flag the response or
    trigger a re-plan rather than treating it as a clean completion.
    """
    try:
        prediction.was_fallback = True
    except (
        Exception
    ):  # pragma: no cover — defensive against immutable Prediction subclasses
        logger.debug(
            "Could not set was_fallback on prediction; downstream defaults to False"
        )


class InstrumentedRLM(TolerantRLM):
    """RLM with EventQueue integration for real-time progress tracking.

    Subclasses dspy.RLM (via TolerantRLM, which hardens the Deno JSON-RPC
    channel against stale messages) to emit events at each iteration:
    - StatusEvent(WORKING) on start
    - ProgressEvent after each iteration (current/total)
    - StatusEvent(COMPLETED) or ErrorEvent on finish
    - Supports cancellation via CancellationToken

    When no event_queue is provided, behaves identically to dspy.RLM.

    Attributes:
        event_queue: Optional EventQueue for emitting progress events
        task_id: Task identifier for events
        tenant_id: Tenant identifier for multi-tenant isolation
    """

    def __init__(
        self,
        signature,
        event_queue: Optional["EventQueue"] = None,
        task_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize InstrumentedRLM.

        Args:
            signature: DSPy signature (e.g., "context, query -> answer")
            event_queue: Optional EventQueue for emitting events
            task_id: Task identifier for events (required if event_queue provided)
            tenant_id: Tenant identifier for events (required if event_queue provided)
            **kwargs: Additional arguments passed to dspy.RLM
        """
        super().__init__(signature, **kwargs)
        self._event_queue = event_queue
        self._task_id = task_id
        if event_queue is not None and not tenant_id:
            raise ValueError(
                "tenant_id is required when event_queue is provided — "
                "RLM events must be tenant-scoped"
            )
        self._tenant_id = tenant_id
        # Retain references to fire-and-forget enqueue tasks; without a live
        # reference CPython may GC the task and drop the event before it runs.
        self._background_tasks: set[asyncio.Task] = set()

    def _emit_sync(self, event) -> None:
        """Emit event synchronously (fire-and-forget in background).

        Attempts to enqueue the event in the current async loop.
        Silently skips if no loop is running.
        """
        if not self._event_queue or not self._task_id:
            return

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._event_queue.enqueue(event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # No running event loop - skip event emission
            # This can happen in sync contexts
            logger.debug("No async loop available for event emission, skipping")

    def _check_cancelled(self) -> None:
        """Check if cancelled and raise RLMCancelledError if so.

        The orchestrator-level inbound-queue drain (in
        ``OrchestratorAgent._iterative_retrieval_loop``) sets the event
        queue's cancellation token with ``reason="user_stop"`` when
        it sees a ``tags=("stop",)`` message. This method observes the
        token at each REPL iteration so cooperative cancellation
        propagates from the outer loop into the RLM.
        """
        if not self._event_queue:
            return

        if self._event_queue.cancellation_token.is_cancelled:
            reason = self._event_queue.cancellation_token.reason
            raise RLMCancelledError(
                f"RLM cancelled: {reason or 'user requested'}",
                reason=reason,
            )

    def forward(self, **input_args) -> Prediction:
        """Execute RLM with progress event emission.

        Overrides dspy.RLM.forward() to add event emission at each iteration.
        Maintains full compatibility with the parent implementation.

        Args:
            **input_args: Input values matching the signature's input fields

        Returns:
            Prediction with output field(s) and trajectory for debugging

        Raises:
            RLMCancelledError: If cancelled via CancellationToken
            ValueError: If required input fields are missing
        """
        self._validate_inputs(input_args)

        self._emit_sync(
            create_status_event(
                self._task_id,
                self._tenant_id,
                TaskState.WORKING,
                phase="rlm_start",
                message=f"Starting RLM (max {self.max_iterations} iterations)",
            )
        )

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history: REPLHistory = REPLHistory()

            for iteration in range(self.max_iterations):
                self._check_cancelled()

                self._emit_sync(
                    create_progress_event(
                        self._task_id,
                        self._tenant_id,
                        current=iteration,
                        total=self.max_iterations,
                        step=f"iteration_{iteration + 1}",
                        details={
                            "iteration": iteration + 1,
                            "max_iterations": self.max_iterations,
                        },
                    )
                )

                result = self._execute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )

                if isinstance(result, Prediction):
                    self._emit_sync(
                        create_status_event(
                            self._task_id,
                            self._tenant_id,
                            TaskState.COMPLETED,
                            phase="rlm_complete",
                            message=f"Completed in {iteration + 1} iterations",
                        )
                    )
                    return result

                history = result

            self._emit_sync(
                create_status_event(
                    self._task_id,
                    self._tenant_id,
                    TaskState.WORKING,
                    phase="rlm_extracting",
                    message="Max iterations reached, extracting final output",
                )
            )

            result = self._extract_fallback(variables, history, output_field_names)
            _mark_fallback(result)

            self._emit_sync(
                create_status_event(
                    self._task_id,
                    self._tenant_id,
                    TaskState.COMPLETED,
                    phase="rlm_complete",
                    message=f"Completed via extraction after {self.max_iterations} iterations",
                )
            )

            return result
