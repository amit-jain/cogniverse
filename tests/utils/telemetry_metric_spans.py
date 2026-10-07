"""Telemetry spans recorded the way their producers record them, with fixed
values so a reader's aggregates are exact."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from opentelemetry.trace import Status, StatusCode

from cogniverse_agents.inference.ab_harness import (
    ABArmResult,
    ABComparison,
    ABResult,
)
from cogniverse_foundation.telemetry.config import SPAN_NAME_PROFILE_SELECTION
from cogniverse_foundation.telemetry.span_contract import (
    OP_PROFILE_SELECTION,
    record_span_io,
)

MS = 1_000_000  # nanoseconds


def record_profile_selection(
    telemetry, tenant_id, modality, duration_ms, *, failed=False
):
    """A profile selection span of ``duration_ms`` with the output slot
    ``ProfileSelectionAgent`` writes."""
    tracer = telemetry._get_tracer_for_project(tenant_id, None)
    # Phoenix stores times to the microsecond; a whole-microsecond start
    # keeps the stored duration exact.
    start = time.time_ns() // 1000 * 1000 - 60_000 * MS
    span = tracer.start_span(SPAN_NAME_PROFILE_SELECTION, start_time=start)
    record_span_io(
        span,
        input_value="a query",
        output={
            "selected_profile": f"{modality}_profile",
            "modality": modality,
            "complexity": "simple",
            "intent": "search",
            "confidence": 0.9,
        },
        operation=OP_PROFILE_SELECTION,
    )
    if failed:
        span.set_status(Status(StatusCode.ERROR, "selection failed"))
    span.end(end_time=start + int(duration_ms * MS))


def ab_result(ab_id, query, latency_delta, tokens_delta, judge_delta, fallback):
    """An A/B result whose with-RLM arm differs from the without-RLM arm by
    the given deltas."""
    without = ABArmResult(
        arm="without_rlm",
        answer="plain",
        latency_ms=100.0,
        tokens_used=50,
        was_fallback=False,
        judge_score=0.5,
    )
    with_rlm = ABArmResult(
        arm="with_rlm",
        answer="recursive",
        latency_ms=100.0 + latency_delta,
        tokens_used=50 + tokens_delta,
        was_fallback=fallback,
        judge_score=0.5 + judge_delta,
    )
    return ABResult(
        ab_id=ab_id,
        query=query,
        context_size_chars=1000,
        without_rlm=without,
        with_rlm=with_rlm,
        comparison=ABComparison(
            latency_delta_ms=latency_delta,
            tokens_delta=tokens_delta,
            judge_delta=judge_delta,
            rlm_was_fallback=fallback,
        ),
    )


def record_trace(
    telemetry, tenant_id, name, duration_ms, *, minutes_ago, error=None, **attributes
):
    """A root span ``name`` of ``duration_ms`` started ``minutes_ago`` with
    one child span, failed with ``error`` when given. Returns the root's
    ``(trace_id, span_id, start_ns)``."""
    from opentelemetry import context, trace

    tracer = telemetry._get_tracer_for_project(tenant_id, None)
    start = time.time_ns() // 1000 * 1000 - int(minutes_ago * 60_000) * MS
    root = tracer.start_span(
        name, context=context.Context(), start_time=start, attributes=attributes
    )
    child = tracer.start_span(
        f"{name}.step",
        context=trace.set_span_in_context(root),
        start_time=start + MS,
    )
    child.end(end_time=start + 2 * MS)
    if error:
        root.set_status(Status(StatusCode.ERROR, error))
    root.end(end_time=start + int(duration_ms * MS))
    ids = root.get_span_context()
    return f"{ids.trace_id:032x}", f"{ids.span_id:016x}", start


SEARCH = "search_service.search"


def record_sample_traces(telemetry, tenant):
    """Six traces a minute apart, each with a child span: four
    ``video_colpali``/``hybrid`` searches of 100-400 ms, a failed
    ``video_colpali``/``bm25`` search of 1000 ms, and a 50 ms
    ``agent.dispatch`` naming its profile and strategy under other keys.
    Returns each trace's expected row, newest first."""
    recorded = [
        (SEARCH, 100, None, {"profile": "video_colpali", "strategy": "hybrid"}),
        (SEARCH, 200, None, {"profile": "video_colpali", "strategy": "hybrid"}),
        (SEARCH, 300, None, {"profile": "video_colpali", "strategy": "hybrid"}),
        (SEARCH, 400, None, {"profile": "video_colpali", "strategy": "hybrid"}),
        (
            SEARCH,
            1000,
            "backend down",
            {"profile": "video_colpali", "strategy": "bm25"},
        ),
        (
            "agent.dispatch",
            50,
            None,
            {"metadata.profile": "audio", "ranking_strategy": "semantic"},
        ),
    ]
    rows = []
    for age, (name, duration, error, attributes) in enumerate(recorded, start=1):
        trace_id, span_id, start = record_trace(
            telemetry,
            tenant,
            name,
            duration,
            minutes_ago=age,
            error=error,
            **attributes,
        )
        rows.append(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "start_time": pd.Timestamp(start, unit="ns", tz="UTC").isoformat(),
                "duration_ms": pytest.approx(float(duration), abs=1e-6),
                "operation": name,
                "succeeded": error is None,
                "profile": attributes.get(
                    "profile", attributes.get("metadata.profile")
                ),
                "strategy": attributes.get(
                    "strategy", attributes.get("ranking_strategy")
                ),
                "error": error,
            }
        )
    return rows
