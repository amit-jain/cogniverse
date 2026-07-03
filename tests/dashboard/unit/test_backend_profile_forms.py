"""Interaction tests for backend-profile sub-form wiring.

`delete_profile_via_api` is unit-tested directly elsewhere; this drives the
delete section's confirmation gate → API wiring, which the render-tab smoke
test never exercises.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _restore_delete_profile_api():
    """The AppTest scripts below monkey-patch the real module attribute
    ``backend_profile.delete_profile_via_api`` in-process; without restoring
    it, every later test file that imports the function gets the fake."""
    import cogniverse_dashboard.tabs.backend_profile as bp

    original = bp.delete_profile_via_api
    yield
    bp.delete_profile_via_api = original


def _delete_profile_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        import streamlit as st
        import cogniverse_dashboard.tabs.backend_profile as bp

        calls = st.session_state.setdefault("_del_calls", [])

        def _fake_delete(profile_name, tenant_id, delete_schema=False):
            calls.append((profile_name, tenant_id, delete_schema))
            return {"success": True}

        bp.delete_profile_via_api = _fake_delete
        bp.render_delete_profile_section(object(), "acme:prod", "video_colpali")
        """
    ).strip()
    path = tmp_path / "app_delete_profile.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_delete_blocked_when_confirmation_does_not_match(tmp_path: Path) -> None:
    at = _delete_profile_app(tmp_path)
    at.run()
    at.text_input(key="delete_confirmation").set_value("wrong")
    at.button[0].click().run()

    assert at.session_state["_del_calls"] == []
    assert any("does not match" in e.value for e in at.error)


def test_delete_calls_api_on_matching_confirmation(tmp_path: Path) -> None:
    at = _delete_profile_app(tmp_path)
    at.run()
    at.text_input(key="delete_confirmation").set_value("video_colpali")
    at.button[0].click().run()

    assert at.session_state["_del_calls"] == [("video_colpali", "acme:prod", False)]
