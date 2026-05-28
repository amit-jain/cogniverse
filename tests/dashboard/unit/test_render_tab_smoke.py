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
    at = AppTest.from_file(script_path, default_timeout=20)
    at.run()
    # AppTest collects exceptions in .exception. An empty list means every
    # widget rendered cleanly.
    assert at.exception == [], (
        f"{fn_name} raised on render: "
        f"{[str(e.value) for e in at.exception]}"
    )
