"""Behavioural contract — inbound stop propagates into running RLM.

The orchestrator's iterative loop drains the inbound queue between
iterations. When a stop message arrives, the loop also calls
``event_queue.cancel(reason="user_stop")`` — that signal must be
visible to any ``InstrumentedRLM`` currently mid-execution inside a
sub-agent's chain so the RLM exits at its next REPL iteration with
``RLMCancelledError(reason="user_stop")`` rather than running to
completion.

This test locks that propagation: cancellation reason
``"user_stop"`` (specifically — not the default ``"user requested"``
nor the existing ``"external_timeout"``) MUST surface from the
RLM. The taxonomy distinguishes user-driven cooperative cancel from
operator-driven external cancel from natural max-iter exit.
"""

from __future__ import annotations

import dspy
import pytest

from cogniverse_agents.inference.instrumented_rlm import (
    InstrumentedRLM,
    RLMCancelledError,
)
from cogniverse_core.events.backends.memory import InMemoryEventQueue

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _StopSignature(dspy.Signature):
    """Trivial signature so the RLM has something to load — the test
    never actually runs the LM (cancellation fires first)."""

    query: str = dspy.InputField()
    answer: str = dspy.OutputField()


# --------------------------------------------------------------------- #
# Pre-cancelled token → RLM exits with user_stop reason at first check    #
# --------------------------------------------------------------------- #


async def test_rlm_raises_user_stop_when_token_pre_cancelled():
    """When the event queue's cancellation token is set with
    ``reason="user_stop"`` BEFORE the RLM iterates, the first
    ``_check_cancelled`` call inside ``forward()`` raises
    ``RLMCancelledError`` carrying exactly that reason — not a
    generic "cancelled" string, not the default ``"user requested"``.
    The taxonomy is the contract the orchestrator-level stop handler
    relies on.
    """
    event_queue = InMemoryEventQueue(task_id="t-1", tenant_id="test_tenant")
    event_queue.cancel(reason="user_stop")

    rlm = InstrumentedRLM(
        _StopSignature,
        event_queue=event_queue,
        task_id="t-1",
        tenant_id="test_tenant",
    )

    with pytest.raises(RLMCancelledError) as exc_info:
        rlm._check_cancelled()
    assert exc_info.value.reason == "user_stop"
    # Exception message includes the reason verbatim.
    assert "user_stop" in str(exc_info.value)


# --------------------------------------------------------------------- #
# External-timeout reason produces a distinct exception                   #
# --------------------------------------------------------------------- #


async def test_rlm_external_timeout_reason_distinct_from_user_stop():
    """``"external_timeout"`` (admin abort / wall-clock cap) MUST
    surface as a distinct reason — not aliased to user_stop. Locks
    the taxonomy across three independent cancellation sources.
    """
    event_queue = InMemoryEventQueue(task_id="t-2", tenant_id="test_tenant")
    event_queue.cancel(reason="external_timeout")

    rlm = InstrumentedRLM(
        _StopSignature,
        event_queue=event_queue,
        task_id="t-2",
        tenant_id="test_tenant",
    )

    with pytest.raises(RLMCancelledError) as exc_info:
        rlm._check_cancelled()
    assert exc_info.value.reason == "external_timeout"
    assert exc_info.value.reason != "user_stop"


# --------------------------------------------------------------------- #
# Token cancelled mid-iteration → next _check_cancelled raises             #
# --------------------------------------------------------------------- #


async def test_rlm_observes_cancellation_set_after_construction():
    """Cancellation set AFTER the RLM is constructed must propagate at
    the next ``_check_cancelled`` — proves the RLM observes the
    cancellation TOKEN's current state, not a snapshot captured at
    construction time. This is the path the orchestrator's stop
    handler relies on: the token is cancelled mid-RLM-iteration and
    the next iter exits.
    """
    event_queue = InMemoryEventQueue(task_id="t-3", tenant_id="test_tenant")

    rlm = InstrumentedRLM(
        _StopSignature,
        event_queue=event_queue,
        task_id="t-3",
        tenant_id="test_tenant",
    )

    # First check sees no cancellation — succeeds.
    rlm._check_cancelled()
    assert event_queue.cancellation_token.is_cancelled is False

    # Cancel via the queue (mirroring what orchestrator's stop handler
    # does), then re-check — must raise.
    event_queue.cancel(reason="user_stop")
    with pytest.raises(RLMCancelledError) as exc_info:
        rlm._check_cancelled()
    assert exc_info.value.reason == "user_stop"


# --------------------------------------------------------------------- #
# No event queue → no cancellation observable (back-compat)              #
# --------------------------------------------------------------------- #


async def test_rlm_without_event_queue_does_not_check_cancellation():
    """RLM instantiated without an event_queue has no cancellation
    signal source — ``_check_cancelled`` is a no-op. Locks back-compat
    for callers that don't wire the event queue.
    """
    rlm = InstrumentedRLM(_StopSignature)
    # Must not raise.
    rlm._check_cancelled()


# --------------------------------------------------------------------- #
# Default reason ("user requested") still works                          #
# --------------------------------------------------------------------- #


async def test_rlm_default_cancellation_reason_when_unspecified():
    """``cancel()`` without an explicit ``reason`` produces a fallback
    string in the exception message — the exact existing
    ``"user requested"`` text. This is the back-compat path for
    callers that don't pass a reason. The user_stop reason is the
    DIFFERENTIATED case the new orchestrator handler emits.
    """
    event_queue = InMemoryEventQueue(task_id="t-4", tenant_id="test_tenant")
    event_queue.cancel()  # no reason

    rlm = InstrumentedRLM(
        _StopSignature,
        event_queue=event_queue,
        task_id="t-4",
        tenant_id="test_tenant",
    )

    with pytest.raises(RLMCancelledError) as exc_info:
        rlm._check_cancelled()
    # When no reason was passed to cancel(), the property returns None,
    # but the exception message falls back to "user requested".
    assert exc_info.value.reason is None
    assert "user requested" in str(exc_info.value)


# --------------------------------------------------------------------- #
# End-to-end via orchestrator-style cancel → propagates to RLM            #
# --------------------------------------------------------------------- #


async def test_orchestrator_style_user_stop_cancel_propagates_to_rlm():
    """Simulates the wired path: orchestrator's iterative-loop stop
    handler calls ``event_queue.cancel(reason="user_stop")`` (the new
    code added in this change), and an InstrumentedRLM sharing that
    event_queue observes the cancellation at its next iteration.

    This is the integration test that ties the two changes together —
    the orchestrator-side stop AND the RLM-side observation must
    agree on the ``"user_stop"`` taxonomy string.
    """
    event_queue = InMemoryEventQueue(task_id="t-5", tenant_id="test_tenant")
    rlm = InstrumentedRLM(
        _StopSignature,
        event_queue=event_queue,
        task_id="t-5",
        tenant_id="test_tenant",
    )

    # Mid-RLM-iteration the outer orchestrator fires the cancel —
    # mirrors orchestrator_agent.py:1973 "event_queue.cancel(reason='user_stop')".
    event_queue.cancel(reason="user_stop")

    with pytest.raises(RLMCancelledError) as exc_info:
        rlm._check_cancelled()
    assert exc_info.value.reason == "user_stop"
