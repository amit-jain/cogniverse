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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional

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

# The span every promoted recursive-LM call emits, whatever promoted it: the
# orchestrator's sufficiency gate and the ingest path's claim extraction.
RLM_RUN_SPAN_NAME = "InstrumentedRLM.run"
MAX_ITERATIONS_ATTRIBUTE = "max_iterations"
RLM_ITERATIONS_ATTRIBUTE = "rlm_iterations"
# The attributes the seam itself writes. Call sites add their own on top.
RLM_RUN_SPAN_ATTRIBUTES = frozenset(
    {MAX_ITERATIONS_ATTRIBUTE, RLM_ITERATIONS_ATTRIBUTE}
)


def reported_rlm_iterations(prediction: Any) -> int:
    """REPL turns the RLM reports for one call.

    ``dspy.RLM`` fills ``Prediction.trajectory`` with one entry per REPL step.
    A call that produced no prediction reports zero.
    """
    trajectory = getattr(prediction, "trajectory", None)
    return len(trajectory) if isinstance(trajectory, list) else 0


class RLMRunSpanRecorder:
    """Stamps the RLM's reported iteration count onto its run span."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def record(self, prediction: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.set_attribute(
                RLM_ITERATIONS_ATTRIBUTE, reported_rlm_iterations(prediction)
            )
        except Exception as exc:
            logger.debug("%s iteration count not recorded: %s", RLM_RUN_SPAN_NAME, exc)


@contextmanager
def rlm_run_span(
    telemetry_manager: Any,
    *,
    tenant_id: str,
    max_iterations: int,
    attributes: Optional[Mapping[str, Any]] = None,
) -> Iterator[RLMRunSpanRecorder]:
    """Emit ``InstrumentedRLM.run`` around one recursive-LM call.

    Yields a recorder whose ``record`` writes the iteration count the returned
    prediction reports; ``rlm_iterations`` is on the span from the start so the
    attribute set does not depend on whether the call reached that point.

    Telemetry never gates the call: with no manager, or when the span cannot be
    opened, the recorder drops its writes and the body still runs. Exceptions
    raised by the body propagate.
    """
    scope = None
    span = None
    if telemetry_manager is not None:
        span_attributes = dict(attributes or {})
        span_attributes[MAX_ITERATIONS_ATTRIBUTE] = int(max_iterations)
        span_attributes[RLM_ITERATIONS_ATTRIBUTE] = 0
        try:
            scope = telemetry_manager.span(
                name=RLM_RUN_SPAN_NAME,
                tenant_id=tenant_id,
                attributes=span_attributes,
            )
            span = scope.__enter__()
        except Exception as exc:
            logger.debug("%s span not opened: %s", RLM_RUN_SPAN_NAME, exc)
            scope = None
            span = None
    try:
        yield RLMRunSpanRecorder(span)
    except BaseException as exc:
        _close_run_span(scope, exc)
        raise
    _close_run_span(scope, None)


def _close_run_span(scope: Any, exc: Optional[BaseException]) -> None:
    """End the run span, letting nothing telemetry raises reach the caller.

    The scope re-raises whatever the body raised; that one propagates from
    ``rlm_run_span`` itself and is not a close failure.
    """
    if scope is None:
        return
    try:
        if exc is None:
            scope.__exit__(None, None, None)
        else:
            scope.__exit__(type(exc), exc, exc.__traceback__)
    except BaseException as close_exc:
        if close_exc is exc:
            return
        logger.debug("%s span not closed: %s", RLM_RUN_SPAN_NAME, close_exc)


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

    def _emit_sync(self, build_event: Callable[[], Any]) -> None:
        """Emit an event synchronously (fire-and-forget in background).

        The event is BUILT only once there is somewhere to send it: every event
        type requires a task id and a tenant id, and those exist only alongside
        a queue, so building one unconditionally raises and takes the RLM call
        down with it.

        Attempts to enqueue the event in the current async loop. Silently skips
        if no loop is running.
        """
        if not self._event_queue or not self._task_id:
            return

        event = build_event()
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
            lambda: create_status_event(
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
                    lambda iteration=iteration: create_progress_event(
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
                        lambda iteration=iteration: create_status_event(
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
                lambda: create_status_event(
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
                lambda: create_status_event(
                    self._task_id,
                    self._tenant_id,
                    TaskState.COMPLETED,
                    phase="rlm_complete",
                    message=f"Completed via extraction after {self.max_iterations} iterations",
                )
            )

            return result
