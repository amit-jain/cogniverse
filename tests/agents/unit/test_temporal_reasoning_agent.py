"""Unit tests for TemporalReasoningAgent."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.temporal_reasoning_agent import (
    TemporalReasoningAgent,
    TemporalReasoningDeps,
    TemporalReasoningInput,
    TimeWindow,
    _content_signature,
    _parse_iso,
)


def _row(
    mid: str,
    content: str,
    *,
    written_at: str | None = "2026-01-01T00:00:00Z",
    subject_key: str = "policy:refunds",
):
    meta: Dict[str, Any] = {"kind": "external_doc", "subject_key": subject_key}
    if written_at is not None:
        # Real memories carry written_at under the provenance payload that
        # attach_to_metadata writes (metadata["provenance"]["written_at"]).
        meta["provenance"] = {"written_at": written_at}
    return {"id": mid, "memory": content, "metadata": meta}


def _factory_for(rows: List[Dict[str, Any]]):
    def _factory(tenant_id: str):
        mm = MagicMock()
        mm.memory = MagicMock()
        mm.get_all_memories = lambda *, tenant_id=tenant_id, agent_name: list(rows)
        return mm

    return _factory


def _build(rows: List[Dict[str, Any]]):
    return TemporalReasoningAgent(
        deps=TemporalReasoningDeps(tenant_id="acme"),
        memory_manager_factory=_factory_for(rows),
    )


class TestParseIso:
    def test_z_suffix(self):
        assert _parse_iso("2026-01-01T00:00:00Z") is not None

    def test_offset(self):
        assert _parse_iso("2026-01-01T00:00:00+00:00") is not None

    def test_naive_assumed_utc(self):
        out = _parse_iso("2026-01-01T00:00:00")
        assert out is not None and out.tzinfo is not None

    def test_returns_none_on_garbage(self):
        assert _parse_iso("garbage") is None
        assert _parse_iso(None) is None
        assert _parse_iso("") is None


class TestContentSignature:
    def test_stable_for_same_content(self):
        a = _content_signature([{"memory": "x"}, {"memory": "y"}])
        b = _content_signature([{"memory": "y"}, {"memory": "x"}])
        assert a == b

    def test_changes_with_content(self):
        a = _content_signature([{"memory": "x"}])
        b = _content_signature([{"memory": "y"}])
        assert a != b


class TestTimeWindowValidation:
    def test_unparseable_start_rejected(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            TimeWindow(label="x", start="garbage")

    def test_unparseable_end_rejected(self):
        with pytest.raises(Exception):
            TimeWindow(label="x", start="2026-01-01T00:00:00Z", end="bad")

    def test_open_ended_window(self):
        w = TimeWindow(label="x", start="2026-01-01T00:00:00Z", end=None)
        assert w.end is None


@pytest.mark.asyncio
class TestBucketing:
    async def test_two_windows_with_clear_split(self):
        rows = [
            _row("a", "v1 policy", written_at="2026-01-15T00:00:00Z"),
            _row("b", "v2 policy", written_at="2026-04-15T00:00:00Z"),
        ]
        agent = _build(rows)
        out = await agent._process_impl(
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="policy:refunds",
                windows=[
                    TimeWindow(
                        label="Q1",
                        start="2026-01-01T00:00:00Z",
                        end="2026-04-01T00:00:00Z",
                    ),
                    TimeWindow(
                        label="Q2",
                        start="2026-04-01T00:00:00Z",
                        end="2026-07-01T00:00:00Z",
                    ),
                ],
            )
        )
        q1 = next(v for v in out.window_views if v.label == "Q1")
        q2 = next(v for v in out.window_views if v.label == "Q2")
        assert q1.matching_memory_ids == ["a"]
        assert q2.matching_memory_ids == ["b"]
        # Two distinct content signatures → knowledge evolved.
        assert out.distinct_signatures_count == 2

    async def test_unchanged_signatures_count_one(self):
        rows = [
            _row("a", "same content", written_at="2026-01-15T00:00:00Z"),
            _row("b", "same content", written_at="2026-04-15T00:00:00Z"),
        ]
        agent = _build(rows)
        out = await agent._process_impl(
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="policy:refunds",
                windows=[
                    TimeWindow(
                        label="Q1",
                        start="2026-01-01T00:00:00Z",
                        end="2026-04-01T00:00:00Z",
                    ),
                    TimeWindow(
                        label="Q2",
                        start="2026-04-01T00:00:00Z",
                        end="2026-07-01T00:00:00Z",
                    ),
                ],
            )
        )
        # Both windows have identical content → 1 distinct signature.
        assert out.distinct_signatures_count == 1

    async def test_undated_memories_isolated(self):
        rows = [
            _row("a", "dated", written_at="2026-01-15T00:00:00Z"),
            _row("b", "undated", written_at=None),
        ]
        agent = _build(rows)
        out = await agent._process_impl(
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="policy:refunds",
                windows=[
                    TimeWindow(
                        label="Q1",
                        start="2026-01-01T00:00:00Z",
                        end="2026-04-01T00:00:00Z",
                    ),
                    TimeWindow(
                        label="Q2",
                        start="2026-04-01T00:00:00Z",
                        end="2026-07-01T00:00:00Z",
                    ),
                ],
            )
        )
        assert out.undated_count == 1
        # Undated does not appear in any window bucket.
        for v in out.window_views:
            assert "b" not in v.matching_memory_ids

    async def test_open_ended_window_includes_recent(self):
        rows = [
            _row("a", "early", written_at="2026-01-01T00:00:00Z"),
            _row("b", "later", written_at="2026-06-01T00:00:00Z"),
            _row("c", "future", written_at="2027-01-01T00:00:00Z"),
        ]
        agent = _build(rows)
        out = await agent._process_impl(
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="policy:refunds",
                windows=[
                    TimeWindow(
                        label="early",
                        start="2026-01-01T00:00:00Z",
                        end="2026-06-01T00:00:00Z",
                    ),
                    # Open-ended: matches everything at-or-after 2026-06-01.
                    TimeWindow(
                        label="from_june",
                        start="2026-06-01T00:00:00Z",
                        end=None,
                    ),
                ],
            )
        )
        early = next(v for v in out.window_views if v.label == "early")
        later = next(v for v in out.window_views if v.label == "from_june")
        assert early.matching_memory_ids == ["a"]
        assert sorted(later.matching_memory_ids) == ["b", "c"]

    async def test_subject_filter_isolates_unrelated_subjects(self):
        rows = [
            _row("a", "right subject", subject_key="policy:refunds"),
            _row("b", "wrong subject", subject_key="policy:returns"),
        ]
        agent = _build(rows)
        out = await agent._process_impl(
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="policy:refunds",
                windows=[
                    TimeWindow(
                        label="Q1",
                        start="2025-01-01T00:00:00Z",
                        end="2027-01-01T00:00:00Z",
                    ),
                    TimeWindow(
                        label="Q2",
                        start="2027-01-01T00:00:00Z",
                        end=None,
                    ),
                ],
            )
        )
        # 'b' is filtered before bucketing.
        all_ids = [mid for v in out.window_views for mid in v.matching_memory_ids]
        assert "b" not in all_ids
        assert "a" in all_ids


@pytest.mark.asyncio
class TestEdgeCases:
    async def test_single_window_rejected_by_validation(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="x",
                windows=[
                    TimeWindow(label="only", start="2026-01-01T00:00:00Z", end=None)
                ],
            )

    async def test_no_subject_key_rejected(self):
        with pytest.raises(Exception):
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="",
                windows=[
                    TimeWindow(label="a", start="2026-01-01T00:00:00Z"),
                    TimeWindow(label="b", start="2026-04-01T00:00:00Z"),
                ],
            )

    async def test_empty_windows_yield_zero_signatures(self):
        # A window with no matches still gets a signature (the empty hash).
        rows: List[Dict[str, Any]] = []
        agent = _build(rows)
        out = await agent._process_impl(
            TemporalReasoningInput(
                tenant_id="acme",
                subject_key="policy:refunds",
                windows=[
                    TimeWindow(label="x", start="2026-01-01T00:00:00Z"),
                    TimeWindow(label="y", start="2026-04-01T00:00:00Z"),
                ],
            )
        )
        # Both empty → identical empty-hash signatures → 1 distinct.
        assert out.distinct_signatures_count == 1
        assert all(v.matching_memory_ids == [] for v in out.window_views)


def test_agent_capabilities_advertised():
    agent = TemporalReasoningAgent(deps=TemporalReasoningDeps(tenant_id="acme"))
    assert agent.agent_name == "temporal_reasoning_agent"
    assert "temporal_reasoning" in agent.capabilities
    assert agent.port == 8025
