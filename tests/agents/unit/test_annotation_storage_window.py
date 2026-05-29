"""get_annotation_statistics must build a UTC-aware 30-day query window.

A naive datetime.now() window is reinterpreted as UTC by the Phoenix
get_spans boundary, shifting the window by the host's local offset on a
non-UTC host (the IST-host class of bug).
"""

from datetime import timezone

import pytest

from cogniverse_agents.routing.annotation_storage import RoutingAnnotationStorage


@pytest.mark.asyncio
async def test_statistics_window_is_utc_aware_30_days():
    mgr = object.__new__(RoutingAnnotationStorage)
    captured = {}

    async def fake_query(start_time, end_time, only_human_reviewed):
        captured["start"] = start_time
        captured["end"] = end_time
        captured["only_human_reviewed"] = only_human_reviewed
        return []

    mgr.query_annotated_spans = fake_query

    await mgr.get_annotation_statistics()

    assert captured["start"].tzinfo == timezone.utc
    assert captured["end"].tzinfo == timezone.utc
    assert (captured["end"] - captured["start"]).days == 30
    assert captured["only_human_reviewed"] is False
