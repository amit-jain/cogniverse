"""Unit tests for the dashboard RCA tab gate."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import cogniverse_dashboard


def _trace(
    trace_id: str,
    *,
    status: str,
    error: str | None,
    operation: str,
    profile: str,
    strategy: str,
    duration_ms: float,
    timestamp: datetime,
):
    return SimpleNamespace(
        trace_id=trace_id,
        status=status,
        error=error,
        operation=operation,
        profile=profile,
        strategy=strategy,
        duration_ms=duration_ms,
        timestamp=timestamp,
    )


@pytest.mark.unit
def test_rca_waits_for_explicit_run(
    monkeypatch,
):
    import cogniverse_dashboard.agent_status as agent_status
    import cogniverse_dashboard.utils.traces as traces_utils
    import cogniverse_evaluation.analysis.root_cause_analysis as rca_module

    trace_rows = [
        _trace(
            "t1",
            status="success",
            error=None,
            operation="search",
            profile="p1",
            strategy="s1",
            duration_ms=100.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        ),
        _trace(
            "t2",
            status="error",
            error="timeout",
            operation="search",
            profile="p1",
            strategy="s1",
            duration_ms=110.0,
            timestamp=datetime(2024, 1, 1, 12, 1, 0),
        ),
    ]

    monkeypatch.setattr(
        traces_utils,
        "fetch_tenant_traces_safely",
        lambda *args, **kwargs: (trace_rows, None),
    )
    monkeypatch.setattr(
        agent_status,
        "probe_agents",
        lambda runtime_url, agents: {
            agent: {"status": "online", "url": f"{runtime_url}/{agent}"}
            for agent in agents
        },
    )

    from types import SimpleNamespace

    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200, json=lambda: {}, text=""
        ),
    )

    calls = {"count": 0}

    def _recording_analyze(self, *args, **kwargs):
        calls["count"] += 1
        return {
            "summary": {
                "total_traces": len(trace_rows),
                "failed_traces": 1,
                "performance_degraded": 0,
                "failure_rate": 0.5,
                "analysis_time": "2024-01-01T12:00:00+00:00",
            },
            "failure_analysis": {},
            "performance_analysis": {},
            "root_causes": [],
            "recommendations": [],
            "statistical_analysis": {},
        }

    monkeypatch.setattr(
        rca_module.RootCauseAnalyzer, "analyze_failures", _recording_analyze
    )

    app_path = Path(cogniverse_dashboard.__file__).parent / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60)
    app.session_state["active_tenant"] = "gate_org:t1"
    app.run()

    assert app.exception == []
    assert [b.label for b in app.button if b.label == "Run analysis"] == [
        "Run analysis"
    ]
    assert calls["count"] == 0


def test_rca_runs_exactly_once_on_explicit_submit(
    monkeypatch,
):
    import cogniverse_dashboard.agent_status as agent_status
    import cogniverse_dashboard.utils.traces as traces_utils
    import cogniverse_evaluation.analysis.root_cause_analysis as rca_module

    trace_rows = [
        _trace(
            "t1",
            status="success",
            error=None,
            operation="search",
            profile="p1",
            strategy="s1",
            duration_ms=100.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        ),
        _trace(
            "t2",
            status="error",
            error="timeout",
            operation="search",
            profile="p1",
            strategy="s1",
            duration_ms=110.0,
            timestamp=datetime(2024, 1, 1, 12, 1, 0),
        ),
    ]

    monkeypatch.setattr(
        traces_utils,
        "fetch_tenant_traces_safely",
        lambda *args, **kwargs: (trace_rows, None),
    )
    monkeypatch.setattr(
        agent_status,
        "probe_agents",
        lambda runtime_url, agents: {
            agent: {"status": "online", "url": f"{runtime_url}/{agent}"}
            for agent in agents
        },
    )

    from types import SimpleNamespace

    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200, json=lambda: {}, text=""
        ),
    )

    calls = {"count": 0}

    def _recording_analyze(self, *args, **kwargs):
        calls["count"] += 1
        return {
            "summary": {
                "total_traces": len(trace_rows),
                "failed_traces": 1,
                "performance_degraded": 0,
                "failure_rate": 0.5,
                "analysis_time": "2024-01-01T12:00:00+00:00",
            },
            "failure_analysis": {},
            "performance_analysis": {},
            "root_causes": [],
            "recommendations": [],
            "statistical_analysis": {},
        }

    monkeypatch.setattr(
        rca_module.RootCauseAnalyzer, "analyze_failures", _recording_analyze
    )

    app_path = Path(cogniverse_dashboard.__file__).parent / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60)
    app.session_state["active_tenant"] = "gate_org:t1"
    app.run()
    assert app.exception == []
    assert calls["count"] == 0

    submit_buttons = [b for b in app.button if b.label == "Run analysis"]
    assert len(submit_buttons) == 1, [b.label for b in app.button]
    submit_buttons[0].click()
    app.run()

    assert app.exception == []
    assert calls["count"] == 1
