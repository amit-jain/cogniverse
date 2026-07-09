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

from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics


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
    analytics.client.spans.get_spans_dataframe = MagicMock(return_value=df)

    metrics = analytics.get_traces()
    assert metrics[0].profile == "audio_clap_semantic"
    assert metrics[0].strategy == "default"
