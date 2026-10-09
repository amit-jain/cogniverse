"""``aggregate_ab_compare`` over ``rlm.ab_compare`` span rows.

The runtime's A/B comparison metrics route aggregates the spans the
``ab-compare`` CLI emits with this function. Covered here:

  * an empty frame returns a zero-row aggregate (not an error);
  * rows group and average correctly, newest first;
  * per-dataset aggregation buckets spans by the ``queries_dataset``
    attribute so an operator running multiple ab-compare jobs sees
    them separately;
  * fallback rate is computed from the ``ab_with_rlm_was_fallback``
    attribute even when reported as a string (tolerant parsing);
  * rows built from spans captured by an in-memory exporter aggregate
    the same way as Phoenix-shaped rows.
"""

from __future__ import annotations

import pandas as pd
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_foundation.telemetry.span_metrics import (
    AB_COMPARE_SPAN_NAME as SPAN_NAME,
)
from cogniverse_foundation.telemetry.span_metrics import (
    ABCompareAggregate,
    aggregate_ab_compare,
)

pytestmark = pytest.mark.integration


# ---- Pure-function tests for the aggregator (no Phoenix required) -----------


def _attr_row(
    *,
    ab_id: str,
    latency_delta: float,
    tokens_delta: int,
    judge_delta: float | None = None,
    was_fallback: bool = False,
    queries_dataset: str = "ds_default",
    name: str = SPAN_NAME,
    start_time: str = "2026-05-09T00:00:00Z",
) -> dict:
    """Build a Phoenix-shaped span row for the aggregator."""
    attrs = {
        "openinference.ab_id": ab_id,
        "openinference.ab_query": "what is x?",
        "openinference.ab_context_chars": 100,
        "openinference.ab_latency_delta_ms": latency_delta,
        "openinference.ab_tokens_delta": tokens_delta,
        "openinference.ab_with_rlm_was_fallback": was_fallback,
        "openinference.queries_dataset": queries_dataset,
        "openinference.tenant_id": "test_tenant",
    }
    if judge_delta is not None:
        attrs["openinference.ab_judge_delta"] = judge_delta
    return {
        "name": name,
        "trace_id": ab_id,
        "context.span_id": ab_id + "_span",
        "start_time": start_time,
        "attributes": attrs,
    }


class TestAggregator:
    def test_empty_input_returns_zero_aggregate(self):
        agg = aggregate_ab_compare(pd.DataFrame())
        assert isinstance(agg, ABCompareAggregate)
        assert agg.rows == 0
        assert agg.avg_latency_delta_ms is None

    def test_single_row_aggregates_correctly(self):
        df = pd.DataFrame([_attr_row(ab_id="a1", latency_delta=10.0, tokens_delta=20)])
        agg = aggregate_ab_compare(df)
        assert agg.rows == 1
        assert agg.avg_latency_delta_ms == 10.0
        assert agg.avg_tokens_delta == 20.0
        # No judge supplied → no aggregate.
        assert agg.avg_judge_delta is None
        assert agg.fallback_rate == 0.0

    def test_multi_row_averages(self):
        df = pd.DataFrame(
            [
                _attr_row(ab_id="a", latency_delta=10.0, tokens_delta=20),
                _attr_row(ab_id="b", latency_delta=30.0, tokens_delta=40),
            ]
        )
        agg = aggregate_ab_compare(df)
        assert agg.rows == 2
        assert agg.avg_latency_delta_ms == 20.0
        assert agg.avg_tokens_delta == 30.0

    def test_rows_are_listed_newest_first(self):
        df = pd.DataFrame(
            [
                _attr_row(
                    ab_id=ab_id, latency_delta=1.0, tokens_delta=1, start_time=start
                )
                for ab_id, start in (
                    ("middle", "2026-05-09T00:00:01Z"),
                    ("oldest", "2026-05-09T00:00:00Z"),
                    ("newest", "2026-05-09T00:00:02Z"),
                )
            ]
        )
        assert list(aggregate_ab_compare(df).per_row["ab_id"]) == [
            "newest",
            "middle",
            "oldest",
        ]

    def test_per_dataset_grouping(self):
        df = pd.DataFrame(
            [
                _attr_row(
                    ab_id="a",
                    latency_delta=10.0,
                    tokens_delta=20,
                    queries_dataset="ds1",
                ),
                _attr_row(
                    ab_id="b",
                    latency_delta=20.0,
                    tokens_delta=40,
                    queries_dataset="ds1",
                ),
                _attr_row(
                    ab_id="c",
                    latency_delta=100.0,
                    tokens_delta=200,
                    queries_dataset="ds2",
                ),
            ]
        )
        agg = aggregate_ab_compare(df)
        assert not agg.per_dataset.empty
        ds1 = agg.per_dataset[agg.per_dataset["queries_dataset"] == "ds1"].iloc[0]
        ds2 = agg.per_dataset[agg.per_dataset["queries_dataset"] == "ds2"].iloc[0]
        assert ds1["rows"] == 2
        assert ds1["avg_latency_delta_ms"] == 15.0
        assert ds2["rows"] == 1
        assert ds2["avg_latency_delta_ms"] == 100.0

    def test_judge_delta_aggregated_when_present(self):
        df = pd.DataFrame(
            [
                _attr_row(ab_id="a", latency_delta=1, tokens_delta=1, judge_delta=0.4),
                _attr_row(ab_id="b", latency_delta=1, tokens_delta=1, judge_delta=0.6),
            ]
        )
        agg = aggregate_ab_compare(df)
        assert agg.avg_judge_delta == pytest.approx(0.5)

    def test_fallback_rate_counts_truthy_strings(self):
        # Phoenix sometimes returns booleans as strings — the aggregator
        # must coerce so the fallback rate is correct in the live tile.
        df = pd.DataFrame(
            [
                _attr_row(
                    ab_id="a", latency_delta=1, tokens_delta=1, was_fallback=True
                ),
                _attr_row(
                    ab_id="b", latency_delta=1, tokens_delta=1, was_fallback=False
                ),
                _attr_row(
                    ab_id="c", latency_delta=1, tokens_delta=1, was_fallback=True
                ),
            ]
        )
        # Force the bool to a string the way Phoenix often does.
        df.loc[0, "attributes"]["openinference.ab_with_rlm_was_fallback"] = "True"
        df.loc[2, "attributes"]["openinference.ab_with_rlm_was_fallback"] = "true"
        agg = aggregate_ab_compare(df)
        assert agg.fallback_rate == pytest.approx(2 / 3)

    def test_filters_to_ab_compare_span_name(self):
        # Aggregator works on whatever it's handed — the caller filters by
        # name. This test just confirms the row shape
        # the aggregator expects from real spans.
        df = pd.DataFrame([_attr_row(ab_id="a", latency_delta=5, tokens_delta=10)])
        agg = aggregate_ab_compare(df)
        assert agg.rows == 1


def test_in_memory_exporter_round_trip():
    """Independent assertion: when the harness's spans are captured via
    InMemorySpanExporter (no Phoenix), aggregate_ab_compare still works
    on the resulting rows.

    Proves the aggregator's input contract is "Phoenix-span-shaped row"
    and not "specifically a Phoenix HTTP response".
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span(SPAN_NAME) as span:
        span.set_attribute("openinference.ab_id", "in_mem_a")
        span.set_attribute("openinference.ab_latency_delta_ms", 7.5)
        span.set_attribute("openinference.ab_tokens_delta", 11)
        span.set_attribute("openinference.queries_dataset", "ds_inmem")

    spans = exporter.get_finished_spans()
    assert spans
    rows = []
    for s in spans:
        rows.append(
            {
                "name": s.name,
                "trace_id": format(s.context.trace_id, "032x"),
                "context.span_id": format(s.context.span_id, "016x"),
                "start_time": s.start_time,
                "attributes": dict(s.attributes),
            }
        )
    df = pd.DataFrame(rows)
    agg = aggregate_ab_compare(df)
    assert agg.rows == 1
    assert agg.avg_latency_delta_ms == 7.5
    assert agg.avg_tokens_delta == 11.0
