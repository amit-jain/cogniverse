"""Full annotation worklist loop against real services — no mocks.

One test walks the whole loop the scheduled cycles run in production:

1. a real low-confidence routing span lands in real Phoenix (canonical
   ``input.value``/``output.value`` slots, exactly as the gateway emits);
2. ``run_annotation_cycle`` — real ``AnnotationAgent`` + real
   ``AnnotationStorage`` — identifies it and enqueues it through the REAL
   mounted runtime router;
3. a reviewer completes it over the REAL endpoint; the label persists into
   real Phoenix BEFORE the in-memory completion (the response says so);
4. the label is readable back through ``query_annotated_spans``;
5. a second cycle run (fresh queue, as after a runtime restart) drops the
   span as already-annotated instead of re-asking for review.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_runtime.quality_monitor_cli import run_annotation_cycle
from cogniverse_runtime.routers import agents as agents_router

pytestmark = pytest.mark.integration


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


@pytest.mark.asyncio
async def test_identify_enqueue_complete_persist_dedupe(real_telemetry):
    from cogniverse_foundation.telemetry.span_contract import record_span_io

    tenant_id = f"annloop{uuid4().hex[:8]}"
    project = real_telemetry.config.get_project_name(tenant_id)

    # 1. A real low-confidence routing decision, canonical slots only.
    with real_telemetry.span(name="cogniverse.routing", tenant_id=tenant_id) as span:
        record_span_io(
            span,
            input_value="play something relaxing",
            output={"chosen_agent": "search_agent", "confidence": 0.2},
            operation="routing",
        )
        span_id = format(span.get_span_context().span_id, "016x")
    real_telemetry.force_flush(timeout_millis=10000)

    storage = AnnotationStorage(tenant_id=tenant_id)
    assert await _wait_for_span(storage, project, span_id)

    # 2. The identification cycle — real agent, real storage, real router.
    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    with patch.object(agents_router, "_annotation_queue", AnnotationQueue()):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            result = await run_annotation_cycle(
                tenant_id=tenant_id,
                runtime_url="http://runtime",
                agent_types=["routing"],
                http_client=client,
            )

            assert result["identified"] >= 1
            assert result["enqueued"] >= 1
            queue = agents_router.get_annotation_queue()
            request = queue.get(span_id)
            assert request is not None
            assert request.tenant_id == tenant_id
            assert request.query == "play something relaxing"
            assert request.routing_confidence == pytest.approx(0.2)

            # 3. A reviewer completes it over the real endpoint; the label
            # must persist durably (persisted: true) before completion.
            resp = await client.post(
                f"/agents/annotations/queue/{span_id}/complete",
                json={"label": "wrong", "reasoning": "should have gone to music"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["persisted"] is True

    # 4. The label is readable back from real Phoenix.
    rows = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        rows = await storage.query_annotated_spans(
            start_time=end - timedelta(hours=1),
            end_time=end,
            only_human_reviewed=True,
        )
        if rows:
            break
        await asyncio.sleep(2)
    assert len(rows) == 1
    assert rows[0]["span_id"] == span_id
    assert rows[0]["annotation_label"] == "wrong"
    assert rows[0]["annotation_reasoning"] == "should have gone to music"

    # 5. A fresh cycle (fresh queue — as after a runtime restart) must NOT
    # re-enqueue the reviewed span.
    with patch.object(agents_router, "_annotation_queue", AnnotationQueue()):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            rerun = await run_annotation_cycle(
                tenant_id=tenant_id,
                runtime_url="http://runtime",
                agent_types=["routing"],
                http_client=client,
            )
        assert rerun["already_annotated"] >= 1
        assert agents_router.get_annotation_queue().get(span_id) is None
