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

import dspy
from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory

if TYPE_CHECKING:
    from cogniverse_core.events import EventQueue

from cogniverse_core.events.types import (
    TaskState,
    create_artifact_event,
    create_progress_event,
    create_status_event,
)

logger = logging.getLogger(__name__)


class RLMCancelledError(Exception):
    """Raised when RLM is cancelled via CancellationToken."""

    def __init__(self, message: str = "RLM cancelled by user", reason: Optional[str] = None):
        self.reason = reason
        super().__init__(message)


class InstrumentedRLM(dspy.RLM):
    """RLM with EventQueue integration for real-time progress tracking.

    Subclasses dspy.RLM to emit events at each iteration:
    - StatusEvent(WORKING) on start
    - ProgressEvent after each iteration (current/total)
    - ArtifactEvent with iteration reasoning (if emit_artifacts=True)
    - StatusEvent(COMPLETED) or ErrorEvent on finish
    - Supports cancellation via CancellationToken

    When no event_queue is provided, behaves identically to dspy.RLM.

    Attributes:
        event_queue: Optional EventQueue for emitting progress events
        task_id: Task identifier for events
        tenant_id: Tenant identifier for multi-tenant isolation
        emit_artifacts: Whether to emit ArtifactEvents with iteration details
    """

    def __init__(
        self,
        signature,
        event_queue: Optional["EventQueue"] = None,
        task_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        emit_artifacts: bool = False,
        **kwargs,
    ):
        """Initialize InstrumentedRLM.

        Args:
            signature: DSPy signature (e.g., "context, query -> answer")
            event_queue: Optional EventQueue for emitting events
            task_id: Task identifier for events (required if event_queue provided)
            tenant_id: Tenant identifier for events (required if event_queue provided)
            emit_artifacts: Whether to emit iteration reasoning as ArtifactEvents
            **kwargs: Additional arguments passed to dspy.RLM
        """
        super().__init__(signature, **kwargs)
        self._event_queue = event_queue
        self._task_id = task_id
        self._tenant_id = tenant_id or "default"
        self._emit_artifacts = emit_artifacts

    def _emit_sync(self, event) -> None:
        """Emit event synchronously (fire-and-forget in background).

        Attempts to enqueue the event in the current async loop.
        Silently skips if no loop is running.
        """
        if not self._event_queue or not self._task_id:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._event_queue.enqueue(event))
        except RuntimeError:
            # No running event loop - skip event emission
            # This can happen in sync contexts
            logger.debug("No async loop available for event emission, skipping")

    def _check_cancelled(self) -> None:
        """Check if cancelled and raise RLMCancelledError if so."""
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
                        details={"iteration": iteration + 1, "max_iterations": self.max_iterations},
                    )
                )

                result = self._execute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )

                if self._emit_artifacts:
                    if isinstance(result, REPLHistory) and len(result) > 0:
                        last_entry = result[-1]
                        if hasattr(last_entry, "reasoning"):
                            self._emit_sync(
                                create_artifact_event(
                                    self._task_id,
                                    self._tenant_id,
                                    artifact_type="rlm_iteration",
                                    data={
                                        "iteration": iteration + 1,
                                        "reasoning": str(last_entry.reasoning)[:500],
                                        "code": str(getattr(last_entry, "code", ""))[:500],
                                    },
                                    is_partial=True,
                                )
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

    async def aforward(self, **input_args) -> Prediction:
        """Async version of forward() with proper event emission.

        Overrides dspy.RLM.aforward() to add event emission at each iteration.

        Args:
            **input_args: Input values matching the signature's input fields

        Returns:
            Prediction with output field(s) and trajectory for debugging

        Raises:
            RLMCancelledError: If cancelled via CancellationToken
            ValueError: If required input fields are missing
        """
        self._validate_inputs(input_args)

        # Emit start event (can use await directly in async context)
        if self._event_queue and self._task_id:
            await self._event_queue.enqueue(
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
            history = REPLHistory()

            for iteration in range(self.max_iterations):
                # Check cancellation
                self._check_cancelled()

                # Emit progress event
                if self._event_queue and self._task_id:
                    await self._event_queue.enqueue(
                        create_progress_event(
                            self._task_id,
                            self._tenant_id,
                            current=iteration,
                            total=self.max_iterations,
                            step=f"iteration_{iteration + 1}",
                            details={"iteration": iteration + 1},
                        )
                    )

                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )

                if isinstance(result, Prediction):
                    if self._event_queue and self._task_id:
                        await self._event_queue.enqueue(
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

            # Fallback extraction
            if self._event_queue and self._task_id:
                await self._event_queue.enqueue(
                    create_status_event(
                        self._task_id,
                        self._tenant_id,
                        TaskState.WORKING,
                        phase="rlm_extracting",
                        message="Max iterations reached, extracting final output",
                    )
                )

            result = await self._aextract_fallback(variables, history, output_field_names)

            if self._event_queue and self._task_id:
                await self._event_queue.enqueue(
                    create_status_event(
                        self._task_id,
                        self._tenant_id,
                        TaskState.COMPLETED,
                        phase="rlm_complete",
                        message=f"Completed via extraction after {self.max_iterations} iterations",
                    )
                )

            return result
