"""The routing-evaluation tab actually runs its async annotation calls.

The tab used to call the async ``identify_spans_needing_annotation`` /
``store_human_annotation`` / ``approve_llm_annotation`` without awaiting them:
each returned an un-executed coroutine (truthy!), so the UI reported
"✅ Annotation saved!" while nothing persisted. The actions now go through
module helpers that resolve the coroutine via ``asyncio.run``.
"""

from __future__ import annotations

import pytest

from cogniverse_dashboard.tabs.routing_evaluation import (
    approve_llm_annotation_sync,
    identify_annotation_spans_sync,
    submit_human_annotation_sync,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubAgent:
    def __init__(self):
        self.calls = []

    async def identify_spans_needing_annotation(self, lookback_hours=None):
        self.calls.append(lookback_hours)
        return ["req-1", "req-2"]


class _StubStorage:
    def __init__(self):
        self.stored = []
        self.approved = []

    async def store_human_annotation(
        self, span_id, label, reasoning, suggested_agent=None
    ):
        self.stored.append((span_id, label, reasoning, suggested_agent))
        return True

    async def approve_llm_annotation(self, span_id):
        self.approved.append(span_id)
        return True


def test_identify_runs_the_coroutine_and_returns_requests():
    agent = _StubAgent()
    requests = identify_annotation_spans_sync(agent, lookback_hours=6)

    # The coroutine actually executed and the resolved value came back —
    # not a coroutine object.
    assert requests == ["req-1", "req-2"]
    assert agent.calls == [6]


def test_submit_persists_and_returns_bool():
    storage = _StubStorage()
    ok = submit_human_annotation_sync(
        storage, span_id="s1", label="correct", reasoning="fine", suggested_agent=None
    )

    assert ok is True
    assert storage.stored == [("s1", "correct", "fine", None)]


def test_approve_persists_and_returns_bool():
    storage = _StubStorage()
    ok = approve_llm_annotation_sync(storage, span_id="s2")

    assert ok is True
    assert storage.approved == ["s2"]
