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

from cogniverse_agents.routing.annotation_storage import (
    AnnotationStorage,
    RoutingAnnotationStorage,
)
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


async def _wait_for_span(storage, project, span_id):
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
            return True
        await asyncio.sleep(2)
    return False


async def _query_until(storage, minimum=1):
    result = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        result = await storage.query_annotated_spans(
            start_time=end - timedelta(hours=1),
            end_time=end,
            only_human_reviewed=False,
        )
        if len(result) >= minimum:
            break
        await asyncio.sleep(2)
    return result


@pytest.mark.asyncio
async def test_canonical_routing_span_fields_extracted(real_telemetry):
    """A routing span written the way the gateway ACTUALLY emits it — only the
    canonical input.value/output.value slots, no attributes.routing — must
    come back from query_annotated_spans with query/chosen_agent/confidence
    populated. The old read used only the legacy attribute, so every real
    span's fields were empty in the feedback rows."""
    from cogniverse_foundation.telemetry.span_contract import record_span_io

    tenant_id = f"anncanon{uuid4().hex[:8]}"
    project = real_telemetry.config.get_project_name(tenant_id)

    with real_telemetry.span(
        name="cogniverse.routing",
        tenant_id=tenant_id,
    ) as span:
        record_span_io(
            span,
            input_value="find robot videos",
            output={"chosen_agent": "video_search", "confidence": 0.9},
            operation="routing",
        )
        span_id = format(span.get_span_context().span_id, "016x")
    real_telemetry.force_flush(timeout_millis=10000)

    storage = AnnotationStorage(tenant_id=tenant_id)
    assert await _wait_for_span(storage, project, span_id)

    assert await storage.store_human_annotation(
        span_id, AnnotationLabel.CORRECT, "right agent"
    )

    result = await _query_until(storage)
    assert len(result) == 1
    r = result[0]
    assert r["span_id"] == span_id
    assert r["query"] == "find robot videos"
    assert r["chosen_agent"] == "video_search"
    assert float(r["routing_confidence"]) == 0.9
    assert r["annotation_label"] == "correct"
    assert r["agent_type"] == "routing"


@pytest.mark.asyncio
async def test_per_agent_annotations_isolated_by_name(real_telemetry):
    """A summary annotation persists under ``summary_annotation`` and is
    invisible to the routing-typed storage on the same tenant — per-agent
    human feedback streams don't bleed into each other."""
    from cogniverse_foundation.telemetry.span_contract import record_span_io

    tenant_id = f"annsumm{uuid4().hex[:8]}"
    project = real_telemetry.config.get_project_name(tenant_id)

    with real_telemetry.span(
        name="SummarizerAgent.process",
        tenant_id=tenant_id,
    ) as span:
        record_span_io(
            span,
            input_value="summarize the meeting",
            output="A fine summary.",
            operation="summarization",
        )
        span_id = format(span.get_span_context().span_id, "016x")
    real_telemetry.force_flush(timeout_millis=10000)

    summary_storage = AnnotationStorage(tenant_id=tenant_id, agent_type="summary")
    assert await _wait_for_span(summary_storage, project, span_id)

    assert await summary_storage.store_human_annotation(
        span_id, AnnotationLabel.WRONG, "missed the decision items"
    )

    result = await _query_until(summary_storage)
    assert len(result) == 1
    assert result[0]["annotation_label"] == "wrong"
    assert result[0]["agent_type"] == "summary"
    assert result[0]["query"] == "summarize the meeting"

    # The routing-typed storage must NOT see the summary annotation.
    routing_storage = AnnotationStorage(tenant_id=tenant_id)
    routing_rows = await routing_storage.query_annotated_spans(
        start_time=datetime.now(timezone.utc) - timedelta(hours=1),
        end_time=datetime.now(timezone.utc) + timedelta(hours=1),
        only_human_reviewed=False,
    )
    assert routing_rows == []
