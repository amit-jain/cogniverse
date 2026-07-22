"""Query temporal-range extraction and its wiring into multi_modal reranking."""

from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_agents.search.rerank_service import rerank_result_dicts
from cogniverse_agents.search.temporal_query import extract_time_range

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestExtractTimeRange:
    def test_relative_n_days(self):
        start, end = extract_time_range("videos from the last 7 days", now=_NOW)
        assert start == _NOW - timedelta(days=7)
        assert end == _NOW

    def test_relative_single_unit(self):
        start, end = extract_time_range("clips from last month", now=_NOW)
        assert start == _NOW - timedelta(days=30)
        assert end == _NOW

    def test_huge_relative_window_clamps_instead_of_overflowing(self):
        # "last 3000 years" -> 1,095,000 days would underflow past datetime.min;
        # "last 9999999999 days" would overflow timedelta (max 999999999 days).
        # Both must clamp to the earliest representable instant, not raise
        # OverflowError (which the search route surfaced as a 500).
        for query in ("fossils from the last 3000 years", "the last 9999999999 days"):
            start, end = extract_time_range(query, now=_NOW)
            assert end == _NOW
            assert start.year == 1  # clamped to datetime.min
            assert start < end

    def test_explicit_year(self):
        start, end = extract_time_range("footage from 2023", now=_NOW)
        assert start == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert end == datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_this_year(self):
        start, end = extract_time_range("highlights this year", now=_NOW)
        assert start == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert end == _NOW

    def test_yesterday(self):
        start, end = extract_time_range("what happened yesterday", now=_NOW)
        assert start == datetime(2024, 6, 14, 0, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2024, 6, 14, 23, 59, 59, 999999, tzinfo=timezone.utc)

    def test_no_temporal_intent_returns_none(self):
        assert extract_time_range("mountain landscape photography", now=_NOW) is None

    def test_fuzzy_recent_not_treated_as_temporal(self):
        # The safe gate excludes vague words like "recent" — no false window.
        assert extract_time_range("recent advances in robotics", now=_NOW) is None


def _make_results():
    now = datetime.now(timezone.utc)

    def ms(dt):
        return int(dt.timestamp() * 1000)

    return [
        {
            "id": "old",
            "score": 0.9,
            "content": "old clip",
            "creation_timestamp": ms(now - timedelta(days=400)),
        },
        {
            "id": "fresh",
            "score": 0.9,
            "content": "fresh clip",
            "creation_timestamp": ms(now - timedelta(days=2)),
        },
    ]


@pytest.mark.asyncio
async def test_temporal_query_feeds_time_range_to_reranker():
    out = await rerank_result_dicts(
        "videos from the last 7 days",
        _make_results(),
        strategy="multi_modal",
        tenant_id="acme:acme",
    )
    by_id = {r["id"]: r for r in out}
    fresh = by_id["fresh"]["metadata"]["score_components"]["temporal"]
    old = by_id["old"]["metadata"]["score_components"]["temporal"]
    assert fresh >= 0.7  # 2 days ago — inside the last-7-days window
    assert old <= 0.2  # 400 days ago — >365 days out of range


@pytest.mark.asyncio
async def test_non_temporal_query_keeps_neutral_temporal_score():
    out = await rerank_result_dicts(
        "mountain landscape",
        _make_results(),
        strategy="multi_modal",
        tenant_id="acme:acme",
    )
    for r in out:
        assert r["metadata"]["score_components"]["temporal"] == 0.5
