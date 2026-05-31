"""render_search_summary must guard on current_search_results so the
Summarize button is absent until a search has populated it."""

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
        from cogniverse_dashboard.search_summary import render_search_summary
        render_search_summary()
        """
    ).strip()
    path = tmp_path / "app_search_summary.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=20)


def test_no_search_results_renders_no_button_and_no_exception(tmp_path: Path) -> None:
    at = _app_test(tmp_path, {})
    at.run()

    assert at.exception == [], [str(e.value) for e in at.exception]
    # Guarded: the Summarize button must not exist before any search has run,
    # so its handler can never read the unset current_search_results.
    assert len(at.button) == 0


def test_results_present_renders_button_without_exception(tmp_path: Path) -> None:
    at = _app_test(
        tmp_path,
        {"current_search_results": {"query": "robots", "results": {}}},
    )
    at.run()

    assert at.exception == [], [str(e.value) for e in at.exception]
    assert len(at.button) == 1
    assert "Summarize" in at.button[0].label
