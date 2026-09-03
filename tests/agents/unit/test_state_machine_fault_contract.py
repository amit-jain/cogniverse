"""Workflow state machine fault contracts.

A raising transition condition or entry callback is a programming error,
not "condition not met": swallowing the condition returned False (which
the approval orchestrator reads as no-matching-transition, stalling an
approval in place), and swallowing the callback advanced the machine
into a state whose entry side effect never ran.
"""

import pytest

from cogniverse_agents.workflow.state_machine import (
    WorkflowState,
    WorkflowStateMachine,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_raising_condition_propagates():
    machine = WorkflowStateMachine(initial_state=WorkflowState.RUNNING)

    def _broken_condition(ctx):
        raise KeyError("pending_items")

    machine.register_transition(
        from_state=WorkflowState.RUNNING,
        to_state=WorkflowState.AWAITING_APPROVAL,
        condition=_broken_condition,
        description="broken condition",
    )

    with pytest.raises(KeyError, match="pending_items"):
        machine.transition()
    assert machine.current_state is WorkflowState.RUNNING
    assert [h.transition_reason for h in machine.history] == ["initialization"]


def test_raising_entry_callback_propagates_without_advancing_state():
    machine = WorkflowStateMachine(initial_state=WorkflowState.RUNNING)

    def _broken_callback(ctx):
        raise RuntimeError("entry side effect failed")

    machine.register_transition(
        from_state=WorkflowState.RUNNING,
        to_state=WorkflowState.AWAITING_APPROVAL,
        condition=lambda ctx: True,
        on_transition=_broken_callback,
        description="broken callback",
    )

    with pytest.raises(RuntimeError, match="entry side effect failed"):
        machine.transition()
    assert machine.current_state is WorkflowState.RUNNING
    assert [h.transition_reason for h in machine.history] == ["initialization"]
