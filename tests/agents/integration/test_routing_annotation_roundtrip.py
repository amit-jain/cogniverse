"""Real-Phoenix round-trip for RoutingAnnotationStorage.

Stores a human routing annotation via the public API, then reads it back via
query_annotated_spans — no mocks. The old read looked for
``attributes.annotation.label`` on the span (annotations actually live in
Phoenix's separate annotation store), so the feedback loop and dashboard always
saw zero annotations.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cogniverse_agents.routing.annotation_storage import RoutingAnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_human_annotation_round_trip(real_telemetry):
    # Unique tenant so the 30-day get_annotation_statistics window can't be
    # polluted by sibling modules sharing the session Phoenix container.
    tenant_id = f"annrt{uuid4().hex[:8]}"
    project = real_telemetry.config.get_project_name(tenant_id)

    with real_telemetry.span(
        name="GatewayAgent.process",
        tenant_id=tenant_id,
        attributes={
            "routing.query": "find robot videos",
            "routing.chosen_agent": "video_search",
            "routing.confidence": 0.9,
        },
    ) as span:
        span_id = format(span.get_span_context().span_id, "016x")
    real_telemetry.force_flush(timeout_millis=10000)

    storage = RoutingAnnotationStorage(tenant_id=tenant_id)

    # Wait for the span to be queryable before annotating it.
    found = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await storage.provider.traces.get_spans(
            project=project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=10000,
        )
        if (
            spans is not None
            and not spans.empty
            and "context.span_id" in spans.columns
            and (spans["context.span_id"] == span_id).any()
        ):
            found = True
            break
        await asyncio.sleep(2)
    assert found, f"routing span {span_id} not indexed in {project}"

    assert await storage.store_human_annotation(
        span_id, AnnotationLabel.CORRECT_ROUTING, "looks right"
    )

    # Poll the read path until the annotation is joined back.
    result = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        result = await storage.query_annotated_spans(
            start_time=end - timedelta(hours=1),
            end_time=end,
            only_human_reviewed=False,
        )
        if len(result) >= 1:
            break
        await asyncio.sleep(2)

    assert len(result) == 1
    r = result[0]
    assert r["span_id"] == span_id
    assert r["annotation_label"] == "correct_routing"
    assert r["annotation_reasoning"] == "looks right"
    assert float(r["annotation_confidence"]) == 1.0
    assert r["human_reviewed"] is True
    assert r["query"] == "find robot videos"
    assert r["chosen_agent"] == "video_search"

    stats = await storage.get_annotation_statistics()
    assert stats["total"] == 1
    assert stats["human_reviewed"] == 1
    assert stats["pending_review"] == 0
    assert stats["by_label"] == {"correct_routing": 1}

    # The human annotation is human_reviewed, so the filtered path returns it.
    human_only = await storage.query_annotated_spans(
        start_time=datetime.now(timezone.utc) - timedelta(hours=1),
        end_time=datetime.now(timezone.utc) + timedelta(hours=1),
        only_human_reviewed=True,
    )
    assert len(human_only) == 1
    assert human_only[0]["span_id"] == span_id
