"""Real-boundary tests for the approval DecisionOrchestrator state loop.

These drive the real DecisionOrchestrator + real WorkflowStateMachine + real
HumanApprovalAgent (with a real confidence extractor injected — the intended
DI, not a boundary mock). Each execute() is bounded by asyncio.wait_for so the
infinite-loop regression fails fast instead of hanging the suite.

Regression: an approval step that yields zero pending items (every
item auto-approved, an empty result, or a non-list result) used to spin forever
because no RUNNING transition matched and the step index never advanced.
"""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.approval.orchestrator import DecisionOrchestrator
from cogniverse_agents.workflow.state_machine import WorkflowState
from cogniverse_core.approval.interfaces import (
    ApprovalStatus,
    ConfidenceExtractor,
    ReviewDecision,
    ReviewItem,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

THRESHOLD = 0.85


class _KeyConfidenceExtractor(ConfidenceExtractor):
    """Read the confidence straight off the item's ``confidence`` key."""

    def extract(self, data: dict) -> float:
        return float(data.get("confidence", 0.0))


def _orchestrator(tenant_id: str = "acme") -> DecisionOrchestrator:
    agent = HumanApprovalAgent(
        confidence_extractor=_KeyConfidenceExtractor(),
        confidence_threshold=THRESHOLD,
        storage=None,
    )
    return DecisionOrchestrator(
        approval_agent=agent,
        workflow_id="wf_c2",
        initial_context={"agent_type": "routing", "tenant_id": tenant_id},
    )


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
    assert [
        item.metadata["agent_type"]
        for item in orch.state_machine.context["current_batch"].items
    ] == ["routing", "routing"]
    assert orch.state_machine.context["current_batch"].context == {
        "agent_type": "routing",
        "tenant_id": "acme:acme",
        "workflow_id": "wf_c2",
        "step_name": "generate",
        "step_index": 0,
    }


@pytest.mark.parametrize("tenant_id", [None, "", "  ", 7])
def test_constructor_rejects_missing_tenant_id(tenant_id):
    agent = HumanApprovalAgent(
        confidence_extractor=_KeyConfidenceExtractor(),
        confidence_threshold=THRESHOLD,
        storage=None,
    )
    initial_context = {"agent_type": "routing"}
    if tenant_id is not None:
        initial_context["tenant_id"] = tenant_id

    with pytest.raises(ValueError, match="tenant_id"):
        DecisionOrchestrator(
            approval_agent=agent,
            workflow_id="wf_missing_tenant",
            initial_context=initial_context,
        )


def test_constructor_rejects_tenant_that_differs_from_approval_storage():
    agent = HumanApprovalAgent(
        confidence_extractor=_KeyConfidenceExtractor(),
        confidence_threshold=THRESHOLD,
        storage=SimpleNamespace(tenant_id="globex:production"),
    )

    with pytest.raises(
        ValueError,
        match=(
            "DecisionOrchestrator tenant_id does not match approval storage: "
            "context=acme:acme storage=globex:production"
        ),
    ):
        DecisionOrchestrator(
            approval_agent=agent,
            workflow_id="wf_storage_tenant_mismatch",
            initial_context={"agent_type": "routing", "tenant_id": "acme"},
        )


@pytest.mark.asyncio
async def test_execute_rejects_tenant_change_before_running_executor():
    orch = _orchestrator("acme")
    calls = []
    orch.register_step(
        "generate",
        lambda context: calls.append(context) or [{"confidence": 0.1}],
        requires_approval=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            "DecisionOrchestrator tenant_id cannot change: "
            "initial=acme:acme update=globex:production"
        ),
    ):
        await orch.execute(context_updates={"tenant_id": "globex:production"})

    assert calls == []
    assert orch.state_machine.context["tenant_id"] == "acme:acme"
    assert orch.state_machine.current_state is WorkflowState.INITIALIZING


@pytest.mark.asyncio
async def test_async_executor_result_is_awaited_before_approval_processing():
    orch = _orchestrator("acme")

    async def executor(context):
        await asyncio.sleep(0)
        return [
            {
                "query": "find the launch recording",
                "chosen_agent": "video_search_agent",
                "confidence": 0.2,
                "observed_tenant": context["tenant_id"],
            }
        ]

    orch.register_step("generate", executor, requires_approval=True)

    context = await _run(orch)

    assert context["step_generate_result"] == [
        {
            "query": "find the launch recording",
            "chosen_agent": "video_search_agent",
            "confidence": 0.2,
            "observed_tenant": "acme:acme",
        }
    ]
    assert context["current_batch"].context["tenant_id"] == "acme:acme"
    assert context["pending_review_count"] == 1
    assert orch.state_machine.current_state is WorkflowState.AWAITING_APPROVAL


@pytest.mark.asyncio
async def test_concurrent_async_executors_keep_tenants_isolated():
    first = _orchestrator("acme")
    second = _orchestrator("globex:production")
    entered = 0
    both_entered = asyncio.Event()

    async def executor(context):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        return [{"confidence": 0.1, "tenant": context["tenant_id"]}]

    first.register_step("generate", executor, requires_approval=True)
    second.register_step("generate", executor, requires_approval=True)

    first_context, second_context = await asyncio.gather(_run(first), _run(second))

    assert first_context["current_batch"].context["tenant_id"] == "acme:acme"
    assert second_context["current_batch"].context["tenant_id"] == ("globex:production")
    assert first_context["step_generate_result"] == [
        {"confidence": 0.1, "tenant": "acme:acme"}
    ]
    assert second_context["step_generate_result"] == [
        {"confidence": 0.1, "tenant": "globex:production"}
    ]


@pytest.mark.asyncio
async def test_async_executor_failure_marks_workflow_failed_without_batch():
    orch = _orchestrator()

    async def executor(_context):
        await asyncio.sleep(0)
        raise RuntimeError("model endpoint refused the request")

    orch.register_step("generate", executor, requires_approval=True)

    context = await _run(orch)

    assert orch.state_machine.current_state is WorkflowState.FAILED
    assert orch.state_machine.history[-1].transition_reason == (
        "forced: model endpoint refused the request"
    )
    assert "step_generate_result" not in context
    assert "current_batch" not in context


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [None, "", "  ", 7])
async def test_execute_rejects_missing_agent_type_before_running_steps(agent_type):
    agent = HumanApprovalAgent(
        confidence_extractor=_KeyConfidenceExtractor(),
        confidence_threshold=THRESHOLD,
        storage=None,
    )
    initial_context = {"tenant_id": "acme"}
    if agent_type is not None:
        initial_context["agent_type"] = agent_type
    orch = DecisionOrchestrator(
        approval_agent=agent,
        workflow_id="wf_missing_agent_type",
        initial_context=initial_context,
    )
    calls = []
    orch.register_step(
        "generate",
        lambda context: calls.append(context) or [{"confidence": 0.1}],
        requires_approval=True,
    )

    with pytest.raises(
        ValueError,
        match=r"^context\.agent_type must be a non-empty string$",
    ):
        await orch.execute()

    assert calls == []
    assert orch.state_machine.current_state is WorkflowState.INITIALIZING
    assert "failure_reason" not in orch.state_machine.context


@pytest.mark.asyncio
async def test_stale_reload_overlay_rejects_original_when_replacement_is_returned():
    orch = _orchestrator()
    orch.register_step(
        "generate",
        lambda _context: [
            {
                "query": "find the product launch",
                "chosen_agent": "video_search_agent",
                "confidence": 0.2,
            }
        ],
        requires_approval=True,
    )
    await _run(orch)
    pending_batch = orch.state_machine.context["current_batch"]
    original = pending_batch.items[0]
    stale_batch = copy.deepcopy(pending_batch)
    replacement = ReviewItem(
        item_id=f"{original.item_id}_regenerated",
        data={
            "query": "find the exact product launch recording",
            "chosen_agent": "video_search_agent",
        },
        confidence=0.4,
        status=ApprovalStatus.REGENERATED,
        metadata={
            "agent_type": "routing",
            "original_item_id": original.item_id,
        },
    )
    orch.approval_agent.apply_decision = AsyncMock(return_value=replacement)
    orch.approval_agent.storage = SimpleNamespace(
        tenant_id="acme:acme",
        get_batch=AsyncMock(return_value=stale_batch),
    )

    await orch.apply_approvals(
        [
            ReviewDecision(
                item_id=original.item_id,
                approved=False,
                reviewer="reviewer@example.com",
                feedback="Use the exact launch wording.",
            )
        ]
    )

    overlaid = orch.state_machine.context["current_batch"]
    assert [(item.item_id, item.status) for item in overlaid.items] == [
        (original.item_id, ApprovalStatus.REJECTED),
        (replacement.item_id, ApprovalStatus.REGENERATED),
    ]
    assert [item.item_id for item in overlaid.pending_review] == [replacement.item_id]
    assert [item.item_id for item in overlaid.rejected] == [original.item_id]
    assert orch.state_machine.context["pending_review_count"] == 1
    assert orch.state_machine.context["rejection_count"] == 0
    assert orch.state_machine.current_state is WorkflowState.AWAITING_APPROVAL


@pytest.mark.asyncio
async def test_decision_storage_failure_leaves_workflow_awaiting_same_batch():
    orch = _orchestrator()
    calls = []
    orch.register_step(
        "generate",
        lambda _context: calls.append("generate") or [{"confidence": 0.2}],
        requires_approval=True,
    )
    await _run(orch)
    batch = orch.state_machine.context["current_batch"]
    item = batch.pending_review[0]
    orch.approval_agent.apply_decision = AsyncMock(
        side_effect=ConnectionError("Phoenix decision write interrupted")
    )

    with pytest.raises(ConnectionError, match="^Phoenix decision write interrupted$"):
        await orch.apply_approvals(
            [
                ReviewDecision(
                    item_id=item.item_id,
                    approved=True,
                    reviewer="reviewer@example.com",
                )
            ]
        )

    assert calls == ["generate"]
    assert orch.current_step_index == 0
    assert orch.state_machine.current_state is WorkflowState.AWAITING_APPROVAL
    assert orch.state_machine.context["current_batch"] is batch
    assert orch.state_machine.context["pending_review_count"] == 1
    assert orch.state_machine.context["rejection_count"] == 0
