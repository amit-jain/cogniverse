"""Real-boundary tests for the approval DecisionOrchestrator state loop.

These drive the real DecisionOrchestrator + real WorkflowStateMachine + real
HumanApprovalAgent (with a real confidence extractor injected — the intended
DI, not a boundary mock). Each execute() is bounded by asyncio.wait_for so the
C2 infinite-loop regression fails fast instead of hanging the suite.

Regression target (C2): an approval step that yields zero pending items (every
item auto-approved, an empty result, or a non-list result) used to spin forever
because no RUNNING transition matched and the step index never advanced.
"""

import asyncio

import pytest

from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.approval.interfaces import ConfidenceExtractor
from cogniverse_agents.approval.orchestrator import DecisionOrchestrator
from cogniverse_agents.workflow.state_machine import WorkflowState

THRESHOLD = 0.85


class _KeyConfidenceExtractor(ConfidenceExtractor):
    """Read the confidence straight off the item's ``confidence`` key."""

    def extract(self, data: dict) -> float:
        return float(data.get("confidence", 0.0))


def _orchestrator() -> DecisionOrchestrator:
    agent = HumanApprovalAgent(
        confidence_extractor=_KeyConfidenceExtractor(),
        confidence_threshold=THRESHOLD,
        storage=None,
    )
    return DecisionOrchestrator(approval_agent=agent, workflow_id="wf_c2")


async def _run(orch: DecisionOrchestrator):
    return await asyncio.wait_for(orch.execute(), timeout=5.0)


@pytest.mark.asyncio
async def test_all_auto_approved_step_completes_without_looping():
    orch = _orchestrator()
    calls = {"n": 0}

    def executor(ctx):
        calls["n"] += 1
        return [{"confidence": 0.95}, {"confidence": 0.9}]

    orch.register_step("generate", executor, requires_approval=True)

    await _run(orch)

    # Step ran exactly once, advanced past, and the workflow terminated.
    assert calls["n"] == 1
    assert orch.current_step_index == 1
    assert orch.state_machine.current_state == WorkflowState.COMPLETED


@pytest.mark.asyncio
async def test_empty_result_approval_step_completes():
    orch = _orchestrator()
    calls = {"n": 0}

    def executor(ctx):
        calls["n"] += 1
        return []

    orch.register_step("generate", executor, requires_approval=True)

    await _run(orch)

    assert calls["n"] == 1
    assert orch.current_step_index == 1
    assert orch.state_machine.current_state == WorkflowState.COMPLETED


@pytest.mark.asyncio
async def test_non_list_result_approval_step_completes():
    orch = _orchestrator()
    calls = {"n": 0}

    def executor(ctx):
        calls["n"] += 1
        return {"unexpected": "shape"}

    orch.register_step("generate", executor, requires_approval=True)

    await _run(orch)

    assert calls["n"] == 1
    assert orch.current_step_index == 1
    assert orch.state_machine.current_state == WorkflowState.COMPLETED


@pytest.mark.asyncio
async def test_auto_approved_then_sequential_step_runs_both_in_order():
    orch = _orchestrator()
    order = []

    orch.register_step(
        "approve_me",
        lambda ctx: order.append("approve_me") or [{"confidence": 0.99}],
        requires_approval=True,
    )
    orch.register_step(
        "finalize",
        lambda ctx: order.append("finalize") or {"done": True},
        requires_approval=False,
    )

    await _run(orch)

    assert order == ["approve_me", "finalize"]
    assert orch.current_step_index == 2
    assert orch.state_machine.current_state == WorkflowState.COMPLETED


@pytest.mark.asyncio
async def test_pending_items_still_pause_for_human_review():
    """No regression: a step with low-confidence items must pause, not advance."""
    orch = _orchestrator()
    calls = {"n": 0}

    def executor(ctx):
        calls["n"] += 1
        return [{"confidence": 0.10}, {"confidence": 0.20}]

    orch.register_step("generate", executor, requires_approval=True)

    await _run(orch)

    assert calls["n"] == 1
    # Index NOT advanced — the workflow is parked awaiting a human decision.
    assert orch.current_step_index == 0
    assert orch.state_machine.current_state == WorkflowState.AWAITING_APPROVAL
    assert orch.state_machine.context["pending_review_count"] == 2
