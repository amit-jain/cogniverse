"""Smoke coverage for every ``render_*_tab`` entry point.

Each dashboard tab declares a top-level ``render_*_tab()`` function that
``app.py`` calls inside ``st.tabs(...)``. A widget-construction regression
(e.g. an st.dataframe kwarg drop) only surfaces at user-click time. These
smoke tests use ``streamlit.testing.v1.AppTest`` to drive each render in
isolation and assert no exception is raised. They run against minimal
session_state — environment-bound widgets that hard-require a real
runtime / Phoenix are exercised through their fallback branches.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Tabs that render purely from session_state + helpers, no backend calls.
# Each entry is the bare-minimum session_state needed to reach the render
# without exercising a live backend connection.
PURE_SMOKE_TABS: list[tuple[str, str, dict]] = [
    (
        "approval_queue",
        "render_approval_queue_tab",
        {"current_tenant": "acme"},
    ),
    (
        "memory_management",
        "render_memory_management_tab",
        {"current_tenant": "acme"},
    ),
    (
        "config_management",
        "render_config_management_tab",
        {"current_tenant": "acme"},
    ),
    (
        "tenant_management",
        "render_tenant_management_tab",
        {"current_tenant": "acme"},
    ),
    (
        "routing_evaluation",
        "render_routing_evaluation_tab",
        {"current_tenant": "acme"},
    ),
    (
        "profile_metrics",
        "render_profile_metrics_tab",
        {"current_tenant": "acme"},
    ),
    (
        "orchestration_annotation",
        "render_orchestration_annotation_tab",
        {"current_tenant": "acme"},
    ),
    (
        "embedding_atlas",
        "render_embedding_atlas_tab",
        {"current_tenant": "acme"},
    ),
    (
        "rlm_ab_compare",
        "render_rlm_ab_compare_tab",
        {"current_tenant": "acme"},
    ),
    (
        "backend_profile",
        "render_backend_profile_tab",
        {"current_tenant": "acme"},
    ),
    (
        "evaluation",
        "render_evaluation_tab",
        {"current_tenant": "acme"},
    ),
    (
        "optimization",
        "render_enhanced_optimization_tab",
        {"current_tenant": "acme"},
    ),
]


def _build_app_script(
    tmp_path: Path, module: str, fn_name: str, session_state: dict
) -> str:
    """Write a one-shot Streamlit script that imports the target tab and
    invokes its render function. Returns the script path as a string."""
    script = textwrap.dedent(
        f"""
        import streamlit as st
        for k, v in {session_state!r}.items():
            st.session_state[k] = v
        from cogniverse_dashboard.tabs.{module} import {fn_name}
        {fn_name}()
        """
    ).strip()
    script_path = tmp_path / f"app_{module}.py"
    script_path.write_text(script)
    return str(script_path)


@pytest.mark.parametrize("module,fn_name,session_state", PURE_SMOKE_TABS)
def test_render_tab_runs_without_uncaught_exception(
    tmp_path: Path, module: str, fn_name: str, session_state: dict
) -> None:
    """Each tab's render path must execute without raising.

    Tabs handle their own backend-unavailable paths internally (st.warning
    / st.info / early return). A bare exception escaping the render is a
    bug — the user would see a Streamlit error banner instead of a tab.
    """
    script_path = _build_app_script(tmp_path, module, fn_name, session_state)
    # The embedding_atlas tab lazy-imports umap + sklearn + embedding_atlas at
    # render time, which alone can exceed 20s on a cold/slow CI runner; a
    # generous ceiling keeps the fast tabs quick (a clean render returns
    # immediately) while giving the heavy one room.
    at = AppTest.from_file(script_path, default_timeout=60)
    at.run()
    # AppTest collects exceptions in .exception. An empty list means every
    # widget rendered cleanly.
    assert at.exception == [], (
        f"{fn_name} raised on render: {[str(e.value) for e in at.exception]}"
    )


def test_routing_evaluation_tab_fetches_spans_once(tmp_path, monkeypatch):
    """The routing-evaluation tab pulls the span window once and reuses it for
    summary/confidence/temporal — it must not re-query the provider per panel.

    Also exercises the annotation-stats panel, whose async
    ``get_annotation_statistics`` must be awaited (a bare call leaves a
    coroutine that ``stats.get(...)`` cannot index).
    """
    from unittest.mock import AsyncMock

    import pandas as pd

    from cogniverse_agents.routing.annotation_storage import RoutingAnnotationStorage

    monkeypatch.setattr(
        RoutingAnnotationStorage,
        "get_annotation_statistics",
        AsyncMock(return_value={"total": 3, "pending_review": 1}),
    )

    get_spans_calls = {"n": 0}
    spans_df = pd.DataFrame(
        [
            {
                "name": "cogniverse.routing",
                "attributes.routing": {
                    "chosen_agent": "video_search",
                    "confidence": 0.9,
                    "processing_time": 12.0,
                },
                "attributes": {},
                "status": "OK",
                "status_code": "OK",
                "parent_id": "p1",
                "start_time": "2026-06-05T00:00:00+00:00",
            },
            {
                "name": "cogniverse.routing",
                "attributes.routing": {
                    "chosen_agent": "text_search",
                    "confidence": 0.4,
                    "processing_time": 30.0,
                },
                "attributes": {},
                "status": "OK",
                "status_code": "OK",
                "parent_id": "p2",
                "start_time": "2026-06-05T00:01:00+00:00",
            },
        ]
    )

    class _Traces:
        async def get_spans(self, **kwargs):
            get_spans_calls["n"] += 1
            return spans_df

    class _Provider:
        traces = _Traces()

    class _Manager:
        def get_provider(self, tenant_id=None):
            return _Provider()

    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: _Manager(),
    )

    script_path = _build_app_script(
        tmp_path,
        "routing_evaluation",
        "render_routing_evaluation_tab",
        {"current_tenant": "acme"},
    )
    at = AppTest.from_file(script_path, default_timeout=30)
    at.run()

    assert at.exception == [], f"render raised: {[str(e.value) for e in at.exception]}"
    # Summary, confidence and temporal panels all reuse the single fetch.
    assert get_spans_calls["n"] == 1, (
        f"expected exactly one span fetch, got {get_spans_calls['n']}"
    )
