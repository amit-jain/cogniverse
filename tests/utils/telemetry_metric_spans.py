"""Telemetry spans recorded the way their producers record them, with fixed
values so a reader's aggregates are exact."""

from __future__ import annotations

import time

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
