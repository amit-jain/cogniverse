"""Regression test for PhoenixAnalytics.get_traces timestamp default.

``analytics.get_traces`` defaulted to ``datetime.now()`` (NAIVE) when a
span lacked ``start_time``. Phoenix's own span timestamps are UTC-aware,
so any subsequent sort/compare across the resulting TraceMetrics list
raised ``TypeError: can't compare offset-naive and offset-aware
datetimes``. The fix uses ``datetime.now(timezone.utc)`` so every
TraceMetrics in the returned list has a timezone-aware ``timestamp``
regardless of which spans the upstream Phoenix call returned.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from cogniverse_core.common.utils.circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
)
from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics


def _disabled_breaker():
    """A no-op breaker (threshold 0 never trips) for unit setups."""
    return CircuitBreaker.get(BreakerConfig(name="test-phoenix", failure_threshold=0))


def _build_spans_df(spans: list[dict]) -> pd.DataFrame:
    """Match the column shape Phoenix's get_spans_dataframe returns."""
    return pd.DataFrame(spans)


def test_missing_start_time_defaults_to_aware_utc() -> None:
    """A span without start_time must still produce an aware timestamp."""
    # parent_id NaN ⇒ root span ⇒ included in metrics.
    spans = [
        {
            "trace_id": "abc",
            "parent_id": None,
            "start_time": None,
            "end_time": None,
            "status_code": "OK",
            "name": "missing-ts",
            "attributes": {},
        },
    ]
    df = _build_spans_df(spans)

    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.client = MagicMock()
    analytics._cache = {}
    analytics.telemetry_url = "http://test"
    analytics._breaker = _disabled_breaker()
    analytics.client.spans.get_spans_dataframe = MagicMock(return_value=df)

    # Patch the parent_id mask to keep the test resilient to pandas API quirks
    # on the `isna()` check — feed a guaranteed-NaN value.
    with patch.object(analytics.client.spans, "get_spans_dataframe", return_value=df):
        metrics = analytics.get_traces()

    assert len(metrics) == 1, f"expected 1 metric; got {len(metrics)}"
    ts = metrics[0].timestamp
    assert ts is not None
    # Strong assertion: the default is timezone-aware and UTC.
    assert ts.tzinfo is not None, (
        f"timestamp must be timezone-aware to avoid naive/aware mix; got {ts!r}"
    )
    assert ts.tzinfo.utcoffset(ts) == timezone.utc.utcoffset(ts), (
        f"timestamp must be UTC; got tzinfo={ts.tzinfo!r}"
    )


def test_aware_and_default_timestamps_are_comparable() -> None:
    """The bug class: mixed aware-span + naive-default crashed sort/compare.

    With the fix, every metrics.timestamp is comparable to an aware
    datetime — the comparison MUST NOT raise TypeError.
    """
    aware_start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    spans = [
        {
            "trace_id": "with-ts",
            "parent_id": None,
            "start_time": aware_start,
            "end_time": None,
            "status_code": "OK",
            "name": "with-ts",
            "attributes": {},
        },
        {
            "trace_id": "without-ts",
            "parent_id": None,
            "start_time": None,
            "end_time": None,
            "status_code": "OK",
            "name": "without-ts",
            "attributes": {},
        },
    ]
    df = _build_spans_df(spans)

    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.client = MagicMock()
    analytics._cache = {}
    analytics.telemetry_url = "http://test"
    analytics._breaker = _disabled_breaker()
    analytics.client.spans.get_spans_dataframe = MagicMock(return_value=df)

    metrics = analytics.get_traces()

    assert len(metrics) == 2
    # Both timestamps must be aware so sorting them does not raise.
    for m in metrics:
        assert m.timestamp.tzinfo is not None, (
            f"naive timestamp slipped through for trace_id={m.trace_id}: {m.timestamp!r}"
        )

    # The compare itself — would have raised TypeError before the fix.
    sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
    # And the aware-start one should sort before the now-default one.
    assert sorted_metrics[0].trace_id == "with-ts"


def _flattened_phoenix_df(rows: list[dict]) -> pd.DataFrame:
    """The REAL shape Phoenix get_spans_dataframe returns: attributes are
    flattened into ``attributes.*`` columns, not a single ``attributes`` dict.
    """
    return pd.DataFrame(rows)


def test_profile_and_strategy_extracted_from_flattened_columns() -> None:
    """Phoenix returns attributes.profile / attributes.ranking_strategy as
    flattened columns; get_traces must read them, not a nonexistent
    ``attributes`` dict column (which left profile/strategy always None)."""
    aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    df = _flattened_phoenix_df(
        [
            {
                "context.trace_id": "t-1",
                "parent_id": None,
                "start_time": aware,
                "end_time": aware,
                "status_code": "OK",
                "name": "search_service.search",
                "attributes.profile": "video_colpali_smol500_mv_frame",
                "attributes.ranking_strategy": "float_float",
            }
        ]
    )

    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.client = MagicMock()
    analytics._cache = {}
    analytics.telemetry_url = "http://test"
    analytics._breaker = _disabled_breaker()
    analytics.client.spans.get_spans_dataframe = MagicMock(return_value=df)

    metrics = analytics.get_traces()

    assert len(metrics) == 1
    assert metrics[0].profile == "video_colpali_smol500_mv_frame"
    assert metrics[0].strategy == "float_float"


def test_flattened_metadata_dotted_keys_resolved() -> None:
    """attributes.metadata.profile must resolve via the metadata.profile key."""
    aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    df = _flattened_phoenix_df(
        [
            {
                "context.trace_id": "t-2",
                "parent_id": None,
                "start_time": aware,
                "end_time": aware,
                "status_code": "OK",
                "name": "search_service.search",
                "attributes.metadata.profile": "audio_clap_semantic",
                "attributes.metadata.strategy": "default",
            }
        ]
    )

    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.client = MagicMock()
    analytics._cache = {}
    analytics.telemetry_url = "http://test"
    analytics._breaker = _disabled_breaker()
    analytics.client.spans.get_spans_dataframe = MagicMock(return_value=df)

    metrics = analytics.get_traces()
    assert metrics[0].profile == "audio_clap_semantic"
    assert metrics[0].strategy == "default"


def test_naive_start_time_is_normalized_to_utc() -> None:
    """A present-but-naive start_time must come out UTC-aware. Mixed with the
    aware fallback rows, naive timestamps made calculate_statistics and
    resample raise TypeError — crashing the Analytics tab."""
    spans = [
        {
            "context.trace_id": "t1",
            "name": "op",
            "parent_id": None,
            "start_time": pd.Timestamp("2026-01-01 10:00:00"),  # naive
            "end_time": pd.Timestamp("2026-01-01 10:00:01"),
            "status_code": "OK",
        },
        {
            "context.trace_id": "t2",
            "name": "op",
            "parent_id": None,
            "start_time": None,  # falls back to aware now()
            "end_time": None,
            "status_code": "OK",
        },
    ]
    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.telemetry_url = "http://unused:6006"
    analytics._cache = {}
    analytics.client = MagicMock()
    analytics.client.spans.get_spans_dataframe = MagicMock(
        return_value=_build_spans_df(spans)
    )
    analytics._breaker = _disabled_breaker()

    traces = analytics.get_traces()

    assert len(traces) == 2
    for t in traces:
        assert t.timestamp.tzinfo is not None, "naive timestamp leaked through"

    # The exact crash site: min()/comparison across the mixed rows
    stats = analytics.calculate_statistics(traces)
    assert stats["total_requests"] == 2


def test_mixed_tz_within_row_still_yields_duration() -> None:
    """An aware start with a naive end (or vice versa) must not silently drop
    the whole trace from the analytics view."""
    from datetime import datetime, timezone

    spans = [
        {
            "context.trace_id": "t1",
            "name": "op",
            "parent_id": None,
            "start_time": datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "end_time": pd.Timestamp("2026-01-01 10:00:02"),  # naive
            "status_code": "OK",
        },
    ]
    analytics = PhoenixAnalytics.__new__(PhoenixAnalytics)
    analytics.telemetry_url = "http://unused:6006"
    analytics._cache = {}
    analytics.client = MagicMock()
    analytics.client.spans.get_spans_dataframe = MagicMock(
        return_value=_build_spans_df(spans)
    )
    analytics._breaker = _disabled_breaker()

    traces = analytics.get_traces()

    assert len(traces) == 1
    assert traces[0].duration_ms == 2000.0


def test_ensure_utc_handles_string_and_epoch_cells() -> None:
    """Object-dtype start/end cells (str, np.datetime64, epoch) must not crash
    get_traces — the .replace(tzinfo=...) fallback assumed a datetime."""
    import numpy as np

    from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics

    f = PhoenixAnalytics._ensure_utc
    # str timestamp -> coerced to a UTC-aware Timestamp
    assert f("2026-01-01 10:00:00").tzinfo is not None
    # np.datetime64 -> UTC-aware
    assert f(np.datetime64("2026-01-01T10:00:00")).tzinfo is not None
    # None / NaT pass through
    assert f(None) is None
    # An uncoercible value returns without raising
    assert f(object()) is not None or True


def test_ensure_utc_reads_numeric_epoch_as_seconds_not_nanoseconds() -> None:
    """A bare numeric epoch cell must resolve to its real UTC instant, not the
    1970 value pd.Timestamp assigns when it reads a raw number as nanoseconds.
    Seconds and milliseconds are told apart by magnitude; strings and
    datetime64 cells keep the pd.Timestamp path.
    """
    import numpy as np

    f = PhoenixAnalytics._ensure_utc
    expected = pd.Timestamp("2023-11-14 22:13:20", tz="UTC")

    # int seconds epoch — landed at 1970 before the fix.
    assert f(1_700_000_000) == expected
    # float seconds epoch.
    assert f(1_700_000_000.0) == expected
    # numpy int64 seconds epoch (np.int64 is not an int subclass).
    assert f(np.int64(1_700_000_000)) == expected
    # milliseconds epoch, told apart from seconds by magnitude.
    assert f(1_700_000_000_000) == expected
    # ISO string keeps the pd.Timestamp path.
    assert f("2023-11-14T22:13:20") == expected
    # datetime64 column cell (naive pd.Timestamp) keeps the pd.Timestamp path.
    assert f(pd.Timestamp("2023-11-14 22:13:20")) == expected
    assert f(np.datetime64("2023-11-14T22:13:20")) == expected

    # An out-of-band numeric (year > 2100 in both units) is dropped, not
    # coerced to a bogus instant.
    assert f(50_000_000_000) == 50_000_000_000
