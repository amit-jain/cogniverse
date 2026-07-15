"""PhoenixAnalytics statistics, outlier, plot, and report assembly.

Seven of its eight public methods render straight into the dashboard's
Analytics tab yet none executed in any test — a wrong percentile, a broken
group-by, or a crashing plot shipped green. These pin the math with exact
values on a constructed trace set and prove every figure assembles.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cogniverse_evaluation.providers.base import TraceMetrics
from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _traces() -> list[TraceMetrics]:
    """10 traces: colpali 100..500ms (2 errors), videoprism 600..1000ms clean."""
    base = datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
    out = []
    for i in range(10):
        duration = (i + 1) * 100.0
        profile = "colpali" if i < 5 else "videoprism"
        status = "error" if i in (1, 3) else "success"
        out.append(
            TraceMetrics(
                trace_id=f"t{i}",
                timestamp=base.replace(minute=i),
                duration_ms=duration,
                operation="search" if i % 2 == 0 else "summarize",
                status=status,
                profile=profile,
                strategy="default",
            )
        )
    return out


@pytest.fixture()
def analytics():
    return PhoenixAnalytics(telemetry_url="http://localhost:1")  # never contacted


def test_response_time_percentiles_are_exact(analytics):
    stats = analytics.calculate_statistics(_traces())

    rt = stats["response_time"]
    assert rt["mean"] == 550.0
    assert rt["median"] == 550.0
    assert rt["min"] == 100.0
    assert rt["max"] == 1000.0
    assert rt["p50"] == 550.0
    assert rt["p75"] == pytest.approx(775.0)
    assert rt["p90"] == pytest.approx(910.0)
    assert rt["p95"] == pytest.approx(955.0)
    assert rt["p99"] == pytest.approx(991.0)
    assert round(rt["std"], 3) == 302.765
    assert stats["total_requests"] == 10


def test_status_and_outlier_sections_are_exact(analytics):
    stats = analytics.calculate_statistics(_traces())

    assert stats["status"]["success_rate"] == 0.8
    assert stats["status"]["error_rate"] == 0.2
    assert stats["status"]["counts"] == {"success": 8, "error": 2}

    # 100..1000 has no IQR outliers (bounds -350..1450).
    assert stats["outliers"]["count"] == 0
    assert stats["outliers"]["percentage"] == 0.0

    assert stats["temporal"]["requests_by_hour"] == {10: 10}


def test_extreme_duration_is_detected_as_outlier(analytics):
    traces = _traces()
    traces.append(
        TraceMetrics(
            trace_id="spike",
            timestamp=datetime(2026, 7, 15, 10, 30, tzinfo=timezone.utc),
            duration_ms=10000.0,
            operation="search",
            status="success",
            profile="colpali",
        )
    )
    stats = analytics.calculate_statistics(traces)
    assert stats["outliers"]["count"] == 1
    assert stats["outliers"]["values"] == [10000.0]


def test_group_by_profile_math_is_exact(analytics):
    stats = analytics.calculate_statistics(_traces(), group_by="profile")

    by_profile = stats["by_profile"]
    assert by_profile["colpali"]["count"] == 5
    assert by_profile["colpali"]["mean_duration"] == 300.0
    assert by_profile["colpali"]["error_rate"] == 0.4
    assert by_profile["videoprism"]["count"] == 5
    assert by_profile["videoprism"]["mean_duration"] == 800.0
    assert by_profile["videoprism"]["error_rate"] == 0.0


def test_empty_traces_return_error_marker(analytics):
    assert analytics.calculate_statistics([]) == {"error": "No traces provided"}


def test_every_plot_assembles_a_real_figure(analytics):
    import plotly.graph_objects as go

    traces = _traces()
    figures = [
        analytics.create_time_series_plot(traces),
        analytics.create_distribution_plot(traces, group_by="profile"),
        analytics.create_heatmap(traces),
        analytics.create_outlier_plot(traces),
        analytics.create_comparison_plot(traces),
    ]
    for fig in figures:
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1, "figure assembled with no data traces"


def test_generate_report_assembles_stats_and_visualizations(analytics, monkeypatch):
    import json

    monkeypatch.setattr(analytics, "get_traces", lambda *a, **k: _traces())

    report = analytics.generate_report()

    assert report["summary"]["total_requests"] == 10
    assert report["summary"]["mean_response_time"] == 550.0
    assert report["statistics"]["status"]["error_rate"] == 0.2
    assert "colpali" in report["statistics_by_profile"]["by_profile"]
    assert "search" in report["statistics_by_operation"]["by_operation"]
    for name, payload in report["visualizations"].items():
        parsed = json.loads(payload)
        assert parsed.get("data"), f"visualization {name} serialized empty"


def test_provider_client_property_contract():
    """PhoenixProvider.client returns a sync phoenix Client bound to the
    initialized endpoint, and refuses before initialization."""
    from phoenix.client import Client

    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    provider = object.__new__(PhoenixProvider)
    provider._http_endpoint = None
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = provider.client

    provider._http_endpoint = "http://localhost:6006"
    client = provider.client
    assert isinstance(client, Client)


def test_single_trace_stats_are_json_safe(analytics):
    """One trace → pandas std is NaN; the stats dict must stay strict-JSON
    encodable (NaN would 500 through a JSONResponse)."""
    import json

    stats = analytics.calculate_statistics(_traces()[:1])
    assert stats["response_time"]["std"] is None
    assert stats["response_time"]["mean"] == 100.0
    json.dumps(stats, allow_nan=False)  # must not raise
