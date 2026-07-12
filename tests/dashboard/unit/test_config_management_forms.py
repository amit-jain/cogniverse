"""Interaction test for the system-config form wiring.

`save_system_config_edits` is unit-tested directly; this drives the form
submit → save wiring the render-tab smoke test never exercises.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _restore_patched_module_attr():
    """The AppTest scripts below monkey-patch a real module attribute
    in-process; restore it so the fake doesn't leak into later test files."""
    import cogniverse_dashboard.tabs.config_management as _m

    original = _m.save_system_config_edits
    yield
    _m.save_system_config_edits = original


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


def _routing_config_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        import streamlit as st
        import cogniverse_dashboard.tabs.config_management as cm
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        mgr = st.session_state.setdefault(
            "_mgr", ConfigManager(store=InMemoryConfigStore())
        )
        cm.render_routing_config_ui(mgr, "acme:prod")
        """
    ).strip()
    path = tmp_path / "app_routing_config.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_routing_config_save_persists_with_tenant(tmp_path: Path) -> None:
    at = _routing_config_app(tmp_path)
    at.run()
    at.button[0].click().run()

    # Pre-fix: RoutingConfigUnified(...) is built without tenant_id (before the
    # save try/except), so __post_init__ raises ValueError as an uncaught
    # script exception.
    assert not at.exception, [str(e) for e in at.exception]
    saved = at.session_state["_mgr"].get_routing_config("acme:prod")
    assert saved is not None and saved.tenant_id == "acme:prod"
