"""Annotation storage query behavior against the telemetry provider.

- get_annotation_statistics must build a UTC-aware 30-day query window:
  a naive datetime.now() window is reinterpreted as UTC by the Phoenix
  get_spans boundary, shifting the window by the host's local offset on a
  non-UTC host (the IST-host class of bug).
- A telemetry-backend failure must raise, not read as "no annotations":
  the storages previously flattened every exception to []/zero stats, so a
  Phoenix outage looked identical to an empty annotation queue.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

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
