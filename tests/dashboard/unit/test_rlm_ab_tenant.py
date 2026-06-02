"""render_rlm_ab_compare_tab reads the gate-selected tenant.

The app shell stores it under "current_tenant"; the tab read "tenant_id"
(never set), so it always fell back to a manual text input.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _app_test(tmp_path: Path, session_state: dict) -> AppTest:
    script = textwrap.dedent(
        f"""
        import streamlit as st
        for k, v in {session_state!r}.items():
            st.session_state[k] = v
        from cogniverse_dashboard.tabs.rlm_ab_compare import render_rlm_ab_compare_tab
        render_rlm_ab_compare_tab()
        """
    ).strip()
    path = tmp_path / "app_rlm_ab.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=20)


def test_current_tenant_skips_manual_tenant_input(tmp_path: Path) -> None:
    at = _app_test(tmp_path, {"current_tenant": "acme:acme"})
    at.run()

    labels = [ti.label for ti in at.text_input]
    # current_tenant was used → no manual "Tenant id" fallback input.
    assert "Tenant id" not in labels


def test_without_current_tenant_shows_manual_input(tmp_path: Path) -> None:
    at = _app_test(tmp_path, {})
    at.run()

    labels = [ti.label for ti in at.text_input]
    assert "Tenant id" in labels
