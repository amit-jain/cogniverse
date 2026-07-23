"""Segmentation and transcription run concurrently, correctly, and as siblings.

The two are independent (keyframe decode vs audio transcription — neither reads
the other's output), so ``ProcessingStrategySet.process`` overlaps them and runs
description/embedding serially after. These pin three things by executing the
interleaving:

* the two stages genuinely overlap (a serial run would not),
* the downstream serial stages still see the upstream results in order, and
* each stage's telemetry span is a SIBLING under the surrounding pipeline span —
  ``asyncio.gather`` gives each task its own context copy, so transcription's
  span is not accidentally nested inside segmentation's.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id="acme:acme",
        schema_name="video_colpali",
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )


_STAGE_RESULTS = {
    "segmentation": {"keyframes": [1, 2, 3]},
    "transcription": {"transcript": {"text": "hi"}},
    "description": {"descriptions": {"0": "d"}},
    "embedding": {"embeddings_count": 3},
}


@pytest.mark.asyncio
async def test_segmentation_and_transcription_overlap_and_ordering(monkeypatch):
    sset = ProcessingStrategySet()
    monkeypatch.setattr(sset, "get_strategy", lambda name: SimpleNamespace())

    # Isolate stage concurrency from telemetry: a no-op span.
    class _Span:
        def set_attribute(self, *a, **k):
            pass

    @contextmanager
    def _span(*a, **k):
        yield _Span()

    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: SimpleNamespace(span=_span),
    )

    events: list[tuple[str, str, float]] = []
    seen_by: dict[str, dict] = {}

    async def timed(strategy_name, strategy, vp, pm, ctx, results):
        events.append((strategy_name, "enter", time.monotonic()))
        if strategy_name in ("description", "embedding"):
            seen_by[strategy_name] = dict(results)
        await asyncio.sleep(0.1)
        events.append((strategy_name, "exit", time.monotonic()))
        return _STAGE_RESULTS[strategy_name]

    monkeypatch.setattr(sset, "_process_strategy", timed)

    result = await sset.process(Path("/data/vid.mp4"), SimpleNamespace(), _ctx())

    def when(stage: str, kind: str) -> float:
        return next(ts for (s, k, ts) in events if s == stage and k == kind)

    # The two intervals overlap — a serial run would have segmentation exit
    # before transcription enters.
    assert when("segmentation", "enter") < when("transcription", "exit")
    assert when("transcription", "enter") < when("segmentation", "exit")

    # Description ran after the concurrent pair and saw BOTH their outputs.
    assert seen_by["description"].get("keyframes") == [1, 2, 3]
    assert seen_by["description"].get("transcript") == {"text": "hi"}
    # Embedding saw description too — the serial dependency chain is intact.
    assert "descriptions" in seen_by["embedding"]

    # Final merged results carry every stage's keys.
    assert result.get("keyframes") == [1, 2, 3]
    assert result.get("transcript") == {"text": "hi"}
    assert "descriptions" in result
    assert result.get("embeddings_count") == 3


@pytest.mark.asyncio
async def test_concurrent_stage_spans_are_siblings_not_nested(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    # A tm whose span() does exactly what the real TelemetryManager.span does at
    # its core — tracer.start_as_current_span(name) — so this exercises the
    # identical context-var path under asyncio.gather without the full manager
    # (which needs BACKEND_URL wiring).
    @contextmanager
    def _span(name, tenant_id=None, attributes=None, component=None):
        with tracer.start_as_current_span(name) as s:
            yield s

    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: SimpleNamespace(span=_span),
    )

    sset = ProcessingStrategySet()
    monkeypatch.setattr(sset, "get_strategy", lambda name: SimpleNamespace())

    async def stage(strategy_name, strategy, vp, pm, ctx, results):
        await asyncio.sleep(0.02)  # force interleaving of the two spans
        return _STAGE_RESULTS[strategy_name]

    monkeypatch.setattr(sset, "_process_strategy", stage)

    # A parent span active while process() runs — both stage spans must parent
    # to IT, proving they are siblings, not nested one inside the other.
    with tracer.start_as_current_span("pipeline.root"):
        await sset.process(Path("/data/v.mp4"), SimpleNamespace(), _ctx())

    spans = {s.name: s for s in exporter.get_finished_spans()}
    seg = spans["pipeline.keyframes"]
    trans = spans["pipeline.transcription"]
    root = spans["pipeline.root"]

    # Both concurrent stage spans parent to the same root span...
    assert seg.parent is not None and seg.parent.span_id == root.context.span_id
    assert trans.parent is not None and trans.parent.span_id == root.context.span_id
    # ...and transcription is NOT nested under segmentation — the exact bug
    # concurrent tm.span context managers would introduce without per-task
    # context isolation.
    assert trans.parent.span_id != seg.context.span_id
