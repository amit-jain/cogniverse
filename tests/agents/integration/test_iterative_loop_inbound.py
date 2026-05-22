"""Behavioural contract — inbound messages reach the iterative loop.

Pins the observable behaviour the inbound channel promises against
``OrchestratorAgent._iterative_retrieval_loop``:

* A constraint message enqueued before iter N appears as the first
  entry of iter N's ``missing_aspects`` — it's the steering the
  reformulator sees, not just a buffered side-channel.
* Multiple constraint messages compose in submission order (not just
  the newest, not just the oldest).
* A stop message exits the loop with ``exit_reason="user_stop"`` and
  returns the partial evidence accumulated up to the drain point —
  cooperative cancellation, not a fresh recompute.
* Untagged / empty messages are drained but do not steer the loop.

These tests stub ``_reformulate_query`` + ``_execute_plan`` +
``_run_sufficiency_gate`` to keep the loop deterministic and fast
(no live LM, no live Vespa). The wiring under test is the inbound
drain itself; LM-driven assertions (constraint actually steers the
LM's reformulated text byte-equal to a golden) belong to the live-
cluster E2E test, not here.

The stubbed loop's exit path is ``"sufficient"`` at iter 1 because
``_iterative_retrieval_loop`` has a convergence heuristic
(``orchestrator_agent.py:1990``) that exits when every sub-agent
returns evidence at ``iter_idx >= 1``. The stub returns one snippet
per iteration so convergence fires; the assertions reflect this
real loop behaviour rather than fighting it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_runtime.messaging import (
    InboundMessage,
    get_inbound_queue_registry,
    reset_inbound_queue_registry_for_testing,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _msg(content: str, *, tags: tuple[str, ...] = ("constraint",)) -> InboundMessage:
    return InboundMessage(
        session_id="sess-1",
        role="user",
        content=content,
        tags=tags,
        created_at=datetime.now(timezone.utc).isoformat(),
        deadline_ms=None,
    )


class _StubOrchestrator:
    """Drop-in stand-in for ``OrchestratorAgent`` exposing exactly the
    methods ``_iterative_retrieval_loop`` calls.

    Records every ``missing_aspects`` value the loop computes so the
    test can assert against the per-iteration trajectory.
    """

    def __init__(self) -> None:
        self._reformulate_calls: List[List[str]] = []
        self._execute_calls: int = 0

    async def _reformulate_query(
        self, query: str, missing_aspects: List[str]
    ) -> tuple[str, str]:
        self._reformulate_calls.append(list(missing_aspects))
        return (f"{query} :: {','.join(missing_aspects)}", "stub-rationale")

    async def _execute_plan(self, plan, **_kwargs) -> Dict[str, Any]:
        self._execute_calls += 1
        return {
            "stub-agent": {
                "results": [
                    {
                        "source_doc_id": f"doc-{self._execute_calls}",
                        "segment_id": f"seg-{self._execute_calls}",
                        "text": f"evidence-{self._execute_calls}",
                        "score": 0.9,
                    }
                ],
            }
        }

    def _extract_evidence_from_results(
        self, results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        snippets: List[Dict[str, Any]] = []
        for v in results.values():
            snippets.extend(v.get("results", []))
        return snippets

    def _deduplicate_evidence(
        self, snippets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        out: List[Dict[str, Any]] = []
        for s in snippets:
            key = (s.get("source_doc_id", ""), s.get("segment_id", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    async def _run_sufficiency_gate(
        self, *, original_query: str, accumulated_evidence, iteration_idx: int
    ) -> Dict[str, Any]:
        return {
            "sufficient": False,
            "missing_aspects": [],
            "confidence": 0.3,
            "rationale": "stub",
        }

    def _emit_retrieval_iteration_span(self, **_kwargs) -> None:
        pass

    def _evidence_token_estimate(self, snippets) -> int:
        return 0

    async def _expand_via_kg_traversal(
        self, accumulated, missing_aspects, **_kwargs
    ) -> List[Dict[str, Any]]:
        return []

    @property
    def _colbert_query_encoder(self):
        return None


async def _run_loop(orch, plan, *, inbound_queue):
    """Bind ``_iterative_retrieval_loop`` from the real class so the
    test exercises the SHIPPED loop body, not a copy.
    """
    return await OrchestratorAgent._iterative_retrieval_loop(
        orch,
        query="test query",
        plan=plan,
        tenant_id="test_tenant",
        workflow_id="wf-1",
        session_id="sess-1",
        agent_results_sink={},
        inbound_queue=inbound_queue,
    )


class _StubPlanStep:
    def __init__(self) -> None:
        self.input_data: Dict[str, Any] = {}


class _StubPlan:
    def __init__(self) -> None:
        self.steps = [_StubPlanStep()]


# --------------------------------------------------------------------- #
# Constraint reaches the reformulator at iter 0                          #
# --------------------------------------------------------------------- #


async def test_constraint_enqueued_before_iter_0_appears_in_iter_0_missing_aspects():
    """A constraint enqueued before the loop starts MUST appear as
    the first element of iter 0's ``missing_aspects`` — the
    reformulator sees the caller's steering on the very first call.
    Convergence heuristic exits at iter 1 with
    ``exit_reason="sufficient"``; the constraint persists across
    iter 1 too.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")
    await queue.enqueue(_msg("only sources from 2024"))

    stub = _StubOrchestrator()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    # Convergence at iter 1 (every sub-agent returned evidence at
    # iter_idx >= 1).
    assert result.exit_reason == "sufficient"
    assert result.iterations_executed == 2
    # Reformulate called twice — once per iteration.
    assert len(stub._reformulate_calls) == 2
    # Iter 0 reformulation got the constraint at position 0.
    assert stub._reformulate_calls[0] == ["only sources from 2024"]
    # Iter 1 still has it — constraints are monotonic across iterations.
    assert stub._reformulate_calls[1] == ["only sources from 2024"]


# --------------------------------------------------------------------- #
# Constraint enqueued mid-loop appears only iter 1 onward                #
# --------------------------------------------------------------------- #


async def test_constraint_enqueued_after_iter_0_appears_in_iter_1_not_iter_0():
    """A constraint that lands between iter 0 and iter 1 must not
    retroactively appear in iter 0's reformulation. The per-iteration
    drain at the top is the boundary.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")

    enqueued = {"done": False}

    class _MidLoopEnqueuer(_StubOrchestrator):
        async def _execute_plan(self, plan, **kwargs):
            result = await super()._execute_plan(plan, **kwargs)
            if not enqueued["done"]:
                await queue.enqueue(_msg("mid-loop constraint"))
                enqueued["done"] = True
            return result

    stub = _MidLoopEnqueuer()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    assert result.exit_reason == "sufficient"
    assert result.iterations_executed == 2
    # Iter 0 saw no constraints (queue was empty when iter 0 drained,
    # constraint enqueued during iter 0's execute).
    assert stub._reformulate_calls[0] == []
    # Iter 1 picks up the constraint from iter 0's enqueue.
    assert stub._reformulate_calls[1] == ["mid-loop constraint"]


# --------------------------------------------------------------------- #
# Multiple constraints compose in submission order                        #
# --------------------------------------------------------------------- #


async def test_multiple_constraints_compose_in_submission_order():
    """Three constraints enqueued before iter 0 must all appear in
    iter 0's ``missing_aspects`` in the same order they were sent.
    Last-write-wins semantics would silently drop earlier
    constraints — this test catches that.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")
    await queue.enqueue(_msg("constraint A"))
    await queue.enqueue(_msg("constraint B"))
    await queue.enqueue(_msg("constraint C"))

    stub = _StubOrchestrator()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    assert result.exit_reason == "sufficient"
    assert result.iterations_executed == 2
    assert stub._reformulate_calls[0] == [
        "constraint A",
        "constraint B",
        "constraint C",
    ]


# --------------------------------------------------------------------- #
# Stop tag — cooperative cancellation returns partial evidence           #
# --------------------------------------------------------------------- #


async def test_stop_mid_loop_exits_user_stop_with_partial_evidence():
    """A ``stop`` message enqueued during iter 0 must:
    1. Be drained at iter 1's top → exit with
       ``exit_reason="user_stop"`` exactly.
    2. Report ``iterations_executed == 1`` (iter 0 ran; iter 1
       never executed past the drain).
    3. Return iter 0's evidence — partial state, not a fresh
       computation.
    Stop drains BEFORE the convergence check fires at iter 1, so
    the partial-state semantics are visible.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")

    class _StopAfterIter0(_StubOrchestrator):
        async def _execute_plan(self, plan, **kwargs):
            result = await super()._execute_plan(plan, **kwargs)
            # Enqueue stop after iter 0 completes its execute_plan,
            # before iter 1's drain.
            if self._execute_calls == 1:
                await queue.enqueue(_msg("", tags=("stop",)))
            return result

    stub = _StopAfterIter0()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    assert result.exit_reason == "user_stop"
    assert result.iterations_executed == 1
    # Partial evidence: exactly iter 0's snippet, not iter 1's.
    assert len(result.evidence) == 1
    assert result.evidence[0]["segment_id"] == "seg-1"
    assert result.evidence[0]["source_doc_id"] == "doc-1"


# --------------------------------------------------------------------- #
# Stop wins over constraint when both arrive in same drain batch         #
# --------------------------------------------------------------------- #


async def test_stop_and_constraint_in_same_batch_stop_takes_precedence():
    """When the queue holds both a stop AND a constraint at the same
    drain, the loop must honour stop — partial cancellation beats
    steering. Stop at iter 0's drain → zero iterations executed.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")

    # Enqueue constraint BEFORE stop — the constraint is older. The
    # stop must still win.
    await queue.enqueue(_msg("late-constraint"))
    await queue.enqueue(_msg("", tags=("stop",)))

    stub = _StubOrchestrator()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    assert result.exit_reason == "user_stop"
    # Zero iterations executed — stop drained at iter 0's drain
    # boundary before any retrieval ran.
    assert result.iterations_executed == 0
    assert result.evidence == []
    # The constraint was never applied to any reformulator call.
    assert stub._reformulate_calls == []


# --------------------------------------------------------------------- #
# No inbound queue → loop runs unchanged (regression guard)               #
# --------------------------------------------------------------------- #


async def test_no_inbound_queue_loop_runs_unchanged():
    """Sessions without an inbound queue MUST run the iterative
    loop exactly as before this feature shipped — convergence at
    iter 1, ``exit_reason="sufficient"``, no inbound steering.
    """
    reset_inbound_queue_registry_for_testing()

    stub = _StubOrchestrator()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=None)

    assert result.exit_reason == "sufficient"
    assert result.iterations_executed == 2
    # No constraints anywhere — every reformulate call's missing_aspects
    # comes from the (empty) gate output only.
    assert stub._reformulate_calls == [[], []]


# --------------------------------------------------------------------- #
# Interrupt tag → same path as constraint                                 #
# --------------------------------------------------------------------- #


async def test_interrupt_tag_treated_as_constraint():
    """``"interrupt"`` and ``"constraint"`` are synonyms — both
    prepend content to missing_aspects. Locking the alias prevents
    silent divergence between the two tag names.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")
    await queue.enqueue(_msg("urgent-replan", tags=("interrupt",)))

    stub = _StubOrchestrator()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    assert result.exit_reason == "sufficient"
    assert stub._reformulate_calls[0] == ["urgent-replan"]
    assert stub._reformulate_calls[1] == ["urgent-replan"]


# --------------------------------------------------------------------- #
# Untagged content message → ignored                                      #
# --------------------------------------------------------------------- #


async def test_untagged_message_does_not_steer_loop():
    """Messages without any of the recognised tags (stop, constraint,
    interrupt) are drained but not applied to ``missing_aspects``.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")
    await queue.enqueue(_msg("just an FYI", tags=("system",)))

    stub = _StubOrchestrator()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    assert result.exit_reason == "sufficient"
    assert stub._reformulate_calls == [[], []]


# --------------------------------------------------------------------- #
# Empty-content constraint → not injected as blank aspect                 #
# --------------------------------------------------------------------- #


async def test_empty_content_constraint_does_not_inject_blank_aspect():
    """A constraint message with empty content must not pollute
    ``missing_aspects`` with an empty string — that would feed the
    reformulator a no-op blank constraint.
    """
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    queue = await registry.get_or_create_queue("sess-1", "test_tenant")
    await queue.enqueue(_msg("", tags=("constraint",)))

    stub = _StubOrchestrator()
    plan = _StubPlan()
    result = await _run_loop(stub, plan, inbound_queue=queue)

    assert result.exit_reason == "sufficient"
    assert stub._reformulate_calls == [[], []]
