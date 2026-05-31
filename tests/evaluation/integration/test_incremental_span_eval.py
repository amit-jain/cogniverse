"""Real-Phoenix test for SpanEvaluator incremental evaluation.

Emits real ``search_service.search`` spans into the test Phoenix container,
then runs the evaluation pipeline twice and asserts the second (incremental)
run skips the spans already carrying the evaluator's annotation — the
behaviour that replaces the removed, broken
``TraceManager.get_unevaluated_traces``/``mark_trace_evaluated`` pair.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_evaluation.span_evaluator import SpanEvaluator


async def _wait_for_indexed_spans(
    provider, project_name: str, expected: int, timeout_s: float = 60.0
) -> int:
    """Poll the RAW span store until ``expected`` search spans are indexed.

    Polls the raw store (not ``get_recent_spans``, which falls back to mock
    data on an empty result and would mask indexing latency).
    """
    deadline = time.time() + timeout_s
    last = 0
    while time.time() < deadline:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)
        df = await provider.telemetry.traces.get_spans(
            project=project_name,
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        if df is not None and not df.empty and "name" in df.columns:
            last = int((df["name"] == "search_service.search").sum())
            if last >= expected:
                return last
        await asyncio.sleep(2.0)
    return last


async def _wait_for_annotated(
    evaluator: SpanEvaluator,
    evaluator_name: str,
    expected: int,
    timeout_s: float = 60.0,
) -> int:
    """Poll the annotation gate until ``expected`` spans are annotated."""
    deadline = time.time() + timeout_s
    last = 0
    while time.time() < deadline:
        skip = await evaluator._already_evaluated_span_ids(
            1, "search_service.search", [evaluator_name]
        )
        last = len(skip.get(evaluator_name, set()))
        if last >= expected:
            return last
        await asyncio.sleep(2.0)
    return last


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_incremental_run_skips_already_annotated_spans(
    search_evaluator_provider,
):
    """Two real spans → run #1 evaluates both, run #2 (incremental) skips
    both, ``incremental=False`` re-evaluates both."""
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    # Unique tenant/project per run so the shared session Phoenix container
    # can't leak spans/annotations between test runs.
    tenant_id = f"incr-eval-{uuid.uuid4().hex[:8]}"
    project_name = f"cogniverse-{tenant_id}"
    manager = get_telemetry_manager()

    queries = ["sunset over the ocean", "city traffic at night"]
    for i, query in enumerate(queries):
        with manager.span(
            name="search_service.search",
            tenant_id=tenant_id,
            attributes={
                "input.value": query,
                "output.value": json.dumps(
                    [{"video_id": f"v_{i}", "score": round(0.9 - 0.1 * i, 2)}]
                ),
            },
        ) as span:
            assert span is not None
    manager.force_flush(timeout_millis=10000)

    provider = search_evaluator_provider
    evaluator = SpanEvaluator(
        tenant_id=tenant_id, provider=provider, project_name=project_name
    )

    indexed = await _wait_for_indexed_spans(provider, project_name, expected=2)
    assert indexed == 2, f"expected 2 indexed search spans, got {indexed}"

    # Run #1 — both spans evaluated, annotations uploaded, nothing skipped.
    s1 = await evaluator.run_evaluation_pipeline(
        hours=1,
        evaluator_names=["relevance"],
        upload_evaluations=True,
        incremental=True,
    )
    assert s1["num_spans_retrieved"] == 2
    assert s1["num_skipped"] == 0
    assert s1["results"]["relevance"]["num_evaluated"] == 2
    assert s1["results"]["relevance"]["num_skipped"] == 0

    annotated = await _wait_for_annotated(evaluator, "relevance", expected=2)
    assert annotated == 2, f"expected 2 relevance annotations indexed, got {annotated}"

    # Run #2 — incremental: both already annotated for relevance → skipped.
    s2 = await evaluator.run_evaluation_pipeline(
        hours=1,
        evaluator_names=["relevance"],
        upload_evaluations=True,
        incremental=True,
    )
    assert s2["num_spans_retrieved"] == 2
    assert s2["num_skipped"] == 2
    assert s2["results"]["relevance"]["num_evaluated"] == 0
    assert s2["results"]["relevance"]["num_skipped"] == 2

    # incremental=False bypasses the gate and re-evaluates both.
    s3 = await evaluator.run_evaluation_pipeline(
        hours=1,
        evaluator_names=["relevance"],
        upload_evaluations=False,
        incremental=False,
    )
    assert s3["num_spans_retrieved"] == 2
    assert s3["num_skipped"] == 0
    assert s3["results"]["relevance"]["num_evaluated"] == 2


async def _wait_for_named_span(
    provider, project_name: str, span_name: str, timeout_s: float = 60.0
) -> bool:
    """Poll the raw span store until a span with ``span_name`` is indexed."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)
        df = await provider.telemetry.traces.get_spans(
            project=project_name,
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )
        if df is not None and not df.empty and "name" in df.columns:
            if int((df["name"] == span_name).sum()) >= 1:
                return True
        await asyncio.sleep(2.0)
    return False


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_get_recent_spans_keeps_non_search_agent_outputs(
    search_evaluator_provider,
):
    """A SummarizerAgent.process span (string output) must survive
    ``get_recent_spans(require_search_shape=False)`` with its raw output under
    ``outputs["value"]`` — on the old code every non-search span was dropped, so
    live quality eval scored 0 samples for SUMMARY/REPORT/GATEWAY. The default
    (search-shape) path must still drop it, proving search behaviour is intact.
    """
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    tenant_id = f"c4-eval-{uuid.uuid4().hex[:8]}"
    project_name = f"cogniverse-{tenant_id}"
    span_name = "SummarizerAgent.process"
    summary_text = "The video shows a sunrise timelapse over a coastal city."
    manager = get_telemetry_manager()

    with manager.span(
        name=span_name,
        tenant_id=tenant_id,
        attributes={
            "input.value": "summarize the coastal city video",
            "output.value": summary_text,
        },
    ) as span:
        assert span is not None
    manager.force_flush(timeout_millis=10000)

    provider = search_evaluator_provider
    evaluator = SpanEvaluator(
        tenant_id=tenant_id, provider=provider, project_name=project_name
    )

    assert await _wait_for_named_span(provider, project_name, span_name), (
        f"summary span {span_name!r} was not indexed in Phoenix"
    )

    # Live per-agent path: the summary span survives with its raw string output.
    kept = await evaluator.get_recent_spans(
        hours=1, operation_name=span_name, require_search_shape=False
    )
    assert len(kept) == 1
    row = kept.iloc[0]
    assert row["operation_name"] == span_name
    assert row["outputs"]["value"] == summary_text
    assert row["outputs"]["results"] == []
    assert row["attributes"]["query"] == "summarize the coastal city video"

    # Search-shape path (default) still drops the non-search span.
    dropped = await evaluator.get_recent_spans(
        hours=1, operation_name=span_name, require_search_shape=True
    )
    assert dropped.empty
