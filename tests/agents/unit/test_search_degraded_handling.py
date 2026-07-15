"""A Vespa soft-timeout (HTTP 200 + root.errors) is surfaced, not swallowed.

The media search agents read root.children after only a status check, so a
degraded query (200 + root.errors code 12 + partial children) returned
incomplete results as if complete. vespa_search_children raises instead —
mirroring the base search_backend._process_results contract.
"""

from __future__ import annotations

import pytest

from cogniverse_agents.search.vespa_query import (
    VespaSearchDegraded,
    vespa_search_children,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_soft_timeout_body_raises():
    degraded = {
        "root": {
            "errors": [{"code": 12, "summary": "Timeout", "message": "timed out"}],
            "children": [],  # partial/empty on a soft timeout
        }
    }
    with pytest.raises(VespaSearchDegraded, match="errors"):
        vespa_search_children(degraded)


def test_clean_body_returns_children():
    ok = {"root": {"children": [{"fields": {"id": "d1"}}, {"fields": {"id": "d2"}}]}}
    hits = vespa_search_children(ok)
    assert [h["fields"]["id"] for h in hits] == ["d1", "d2"]


def test_empty_result_is_not_an_error():
    assert vespa_search_children({"root": {"children": []}}) == []
    assert vespa_search_children({}) == []


def test_degraded_coverage_warns_but_returns_hits(caplog):
    import logging

    body = {
        "root": {
            "coverage": {"degraded": {"timeout": True}},
            "children": [{"fields": {"id": "d1"}}],
        }
    }
    with caplog.at_level(logging.WARNING):
        hits = vespa_search_children(body)
    assert len(hits) == 1
    assert any("degraded" in r.getMessage().lower() for r in caplog.records)
