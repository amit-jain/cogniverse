"""Annotation storage query behavior against the telemetry provider.

- get_annotation_statistics must build a UTC-aware 30-day query window:
  a naive datetime.now() window is reinterpreted as UTC by the Phoenix
  get_spans boundary, shifting the window by the host's local offset on a
  non-UTC host (the IST-host class of bug).
- A telemetry-backend failure must raise, not read as "no annotations":
  the storages previously flattened every exception to []/zero stats, so a
  Phoenix outage looked identical to an empty annotation queue.
- query_annotated_spans(spans_df=...) must reuse a caller-provided project
  frame instead of re-pulling the whole project window per call.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from cogniverse_agents.routing.annotation_storage import RoutingAnnotationStorage
from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotationStorage,
)


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


class TestBackendFailurePropagates:
    """A telemetry-backend failure must raise, not read as 'no annotations'."""

    @staticmethod
    def _failing_provider():
        provider = MagicMock()
        provider.traces.get_spans = AsyncMock(
            side_effect=TimeoutError("phoenix query timed out")
        )
        return provider

    def _routing_storage(self):
        storage = object.__new__(RoutingAnnotationStorage)
        storage.project_name = "cogniverse-acme"
        storage.provider = self._failing_provider()
        return storage

    def _orchestration_storage(self):
        storage = object.__new__(OrchestrationAnnotationStorage)
        storage.project_name = "cogniverse-acme"
        storage.provider = self._failing_provider()
        return storage

    @pytest.mark.asyncio
    async def test_routing_query_annotated_spans_raises_on_backend_failure(self):
        storage = self._routing_storage()
        with pytest.raises(TimeoutError, match="phoenix query timed out"):
            await storage.query_annotated_spans(
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_routing_statistics_raise_on_backend_failure(self):
        storage = self._routing_storage()
        with pytest.raises(TimeoutError, match="phoenix query timed out"):
            await storage.get_annotation_statistics()

    @pytest.mark.asyncio
    async def test_orchestration_query_annotated_spans_raises_on_backend_failure(self):
        storage = self._orchestration_storage()
        with pytest.raises(TimeoutError, match="phoenix query timed out"):
            await storage.query_annotated_spans(
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )


class TestPrefetchedSpansSkipTheProviderPull:
    """query_annotated_spans(spans_df=...) reuses the caller's pre-fetched
    project frame. The quality-monitor cycles call the method once per agent
    type over an identical window; without the parameter every call re-pulled
    the whole project (limit=10000) from the provider."""

    @staticmethod
    def _spans_df():
        return pd.DataFrame(
            [
                {
                    "context.span_id": "s1",
                    "name": "cogniverse.routing",
                    "attributes.input.value": "find robot videos",
                    "attributes.output.value": json.dumps(
                        {"chosen_agent": "video_search", "confidence": 0.42}
                    ),
                }
            ]
        )

    @staticmethod
    def _annotations_df():
        return pd.DataFrame(
            [
                {
                    "result.label": "correct",
                    "result.score": 0.9,
                    "metadata": {
                        "human_reviewed": True,
                        "reasoning": "matches intent",
                        "timestamp": "2026-07-17T10:00:00+00:00",
                    },
                }
            ],
            index=["s1"],
        )

    def _storage(self):
        storage = object.__new__(RoutingAnnotationStorage)
        storage.project_name = "cogniverse-acme"
        storage.agent_type = "routing"
        storage.annotation_name = "routing_annotation"
        provider = MagicMock()
        provider.traces.get_spans = AsyncMock(
            side_effect=AssertionError("provider span pull must be skipped")
        )
        provider.annotations.get_annotations = AsyncMock(
            return_value=self._annotations_df()
        )
        storage.provider = provider
        return storage

    @pytest.mark.asyncio
    async def test_prefetched_frame_is_used_without_a_provider_span_pull(self):
        storage = self._storage()

        rows = await storage.query_annotated_spans(
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            only_human_reviewed=True,
            spans_df=self._spans_df(),
        )

        storage.provider.traces.get_spans.assert_not_called()
        assert rows == [
            {
                "span_id": "s1",
                "agent_type": "routing",
                "query": "find robot videos",
                "chosen_agent": "video_search",
                "routing_confidence": 0.42,
                "output": {"chosen_agent": "video_search", "confidence": 0.42},
                "annotation_label": "correct",
                "annotation_confidence": 0.9,
                "annotation_reasoning": "matches intent",
                "annotation_timestamp": "2026-07-17T10:00:00+00:00",
                "suggested_agent": None,
                "human_reviewed": True,
                "context": {},
            }
        ]

    @pytest.mark.asyncio
    async def test_omitting_the_frame_still_pulls_from_the_provider(self):
        storage = self._storage()
        storage.provider.traces.get_spans = AsyncMock(return_value=self._spans_df())

        rows = await storage.query_annotated_spans(
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            only_human_reviewed=True,
        )

        storage.provider.traces.get_spans.assert_awaited_once()
        assert len(rows) == 1
        assert rows[0]["span_id"] == "s1"
