"""The online span-scoring loop scores spans concurrently, bounded and correct.

``run_online_evaluation`` used to score spans one ``await evaluate_span`` at a
time, serialising every per-span Phoenix annotation write. ``_score_spans_bounded``
overlaps them through a semaphore. This drives a REAL ``OnlineEvaluator`` (real
structural evaluators) through the helper and proves, by executing the
interleaving, that:

* concurrency is bounded to ``_ONLINE_EVAL_CONCURRENCY`` (never a fan-out storm),
* every span is counted exactly once (no lost counter increment under overlap),
* every produced score is persisted (the returned total equals the write count).

The provider is instrumented to force interleaving and observe peak concurrency —
the barrier+counter technique for a concurrency invariant — not to assert a
payload shape.
"""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_evaluation.online_evaluator import OnlineEvaluator
from cogniverse_runtime.optimization_cli import (
    _ONLINE_EVAL_CONCURRENCY,
    _score_spans_bounded,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _InstrumentedAnnotations:
    """Records peak concurrency across add_annotation calls and yields the event
    loop mid-call so overlapping tasks genuinely interleave."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.writes = 0

    async def add_annotation(self, **kwargs) -> str:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.writes += 1
        await asyncio.sleep(0.01)  # yield so concurrent writers pile up
        self.in_flight -= 1
        return kwargs.get("span_id", "")


class _InstrumentedProvider:
    def __init__(self) -> None:
        self.annotations = _InstrumentedAnnotations()


def _span(i: int) -> dict:
    return {
        "context.span_id": f"span-{i}",
        "attributes.routing": {"confidence": "high"},
        "status_code": "OK",
    }


@pytest.mark.asyncio
async def test_scoring_is_bounded_counted_and_fully_persisted():
    provider = _InstrumentedProvider()
    evaluator = OnlineEvaluator(
        provider=provider, project_name="p", config=None, agent_type="routing"
    )
    n = 20
    rows = [_span(i) for i in range(n)]

    scores_persisted = await _score_spans_bounded(evaluator, rows)

    ann = provider.annotations
    # Genuine overlap occurred (more than one writer in flight at the peak)...
    assert ann.max_in_flight > 1
    # ...but never beyond the semaphore bound.
    assert ann.max_in_flight <= _ONLINE_EVAL_CONCURRENCY
    # Every span was counted exactly once despite the interleaving.
    stats = evaluator.get_statistics()
    assert stats["total_evaluated"] == n
    assert stats["total_skipped"] == 0
    # The returned total matches the writes actually issued — no score dropped,
    # none double-counted.
    assert scores_persisted == ann.writes
    assert scores_persisted > 0


@pytest.mark.asyncio
async def test_empty_rows_is_a_noop():
    provider = _InstrumentedProvider()
    evaluator = OnlineEvaluator(
        provider=provider, project_name="p", config=None, agent_type="routing"
    )
    assert await _score_spans_bounded(evaluator, []) == 0
    assert provider.annotations.writes == 0
    assert evaluator.get_statistics()["total_evaluated"] == 0
