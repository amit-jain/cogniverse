"""Interaction test for the system-config form wiring.

`save_system_config_edits` is unit-tested directly; this drives the form
submit → save wiring the render-tab smoke test never exercises.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _system_config_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        import streamlit as st
        import cogniverse_dashboard.tabs.config_management as cm
        from cogniverse_foundation.config.unified_config import SystemConfig

        class _Mgr:
            def get_system_config(self):
                return SystemConfig()

        calls = st.session_state.setdefault("_save_calls", [])

        def _fake_save(manager, current, **edits):
            calls.append(edits)
            return current

        cm.save_system_config_edits = _fake_save
        cm.render_system_config_ui(_Mgr(), "acme:prod")
        """
    ).strip()
    path = tmp_path / "app_system_config.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_system_config_submit_calls_save_with_edit_kwargs(tmp_path: Path) -> None:
    at = _system_config_app(tmp_path)
    at.run()
    at.button[0].click().run()

    calls = at.session_state["_save_calls"]
    assert len(calls) == 1, calls
    edits = calls[0]
    # The form threads its fields through to save_system_config_edits.
    for key in ("video_agent_url", "search_backend", "llm_model", "environment"):
        assert key in edits
