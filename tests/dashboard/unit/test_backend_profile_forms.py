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


@pytest.mark.unit
def test_get_runtime_api_url_reads_the_key_the_app_sets():
    """The dashboard app populates ``runtime_url`` (its authoritative in-cluster
    URL); backend_profile must read that, not the never-set ``runtime_api_url``
    that made every profile/schema call fall back to localhost:8000 in a pod."""
    import streamlit as st

    import cogniverse_dashboard.tabs.backend_profile as bp

    st.session_state.clear()
    st.session_state["runtime_url"] = "http://cogniverse-runtime:28000"
    try:
        assert bp.get_runtime_api_url() == "http://cogniverse-runtime:28000"
    finally:
        st.session_state.clear()


def test_backend_profile_and_tenant_management_share_one_helper():
    """The two tabs must not carry divergent copies of the URL resolver."""
    import cogniverse_dashboard.tabs.backend_profile as bp
    import cogniverse_dashboard.tabs.tenant_management as tm

    assert bp.get_runtime_api_url is tm.get_runtime_api_url


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


def _profile_manager_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        import streamlit as st

        from cogniverse_dashboard.tabs import backend_profile as bp
        from cogniverse_foundation.config.unified_config import BackendProfileConfig

        st.session_state["current_tenant"] = "acme:prod"

        bp.get_profile_schema_status = lambda name, tenant: {
            "schema_deployed": True,
            "tenant_schema_name": f"{tenant}:{name}",
            "error": None,
        }

        class _Manager:
            def get_backend_profile(self, profile_name, tenant_id=None, service="backend"):
                return BackendProfileConfig(
                    profile_name=profile_name,
                    type="video",
                    description="High quality backend profile",
                    schema_name="video_colpali_smol500_mv_frame",
                    embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
                    pipeline_config={"fps": 60},
                    strategies={"search": {"enabled": True}},
                    embedding_type="multi_vector",
                    model_specific={"quantized": False},
                )

        bp.render_profile_manager(
            _Manager(),
            "acme:prod",
            "video_colpali_smol500_mv_frame",
        )
        """
    ).strip()
    path = tmp_path / "app_profile_manager.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_profile_manager_renders_dataclass_profile(tmp_path: Path) -> None:
    at = _profile_manager_app(tmp_path)
    at.run()

    assert at.exception == []
    assert [tab.label for tab in at.tabs] == [
        "📝 Edit",
        "🚀 Deploy Schema",
        "🗑️ Delete",
    ]

    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics == {
        "Type": "video",
        "Embedding Type": "multi_vector",
        "Schema": "video_colpali_smol500_mv_frame",
        "Schema Status": "✅ Deployed",
        "Base Schema": "video_colpali_smol500_mv_frame",
        "Embedding Model": "TomoroAI/tomoro-colqwen3-embed-4b",
    }

    markdown = " ".join(block.value for block in at.markdown)
    assert "Profile: `video_colpali_smol500_mv_frame`" in markdown
    assert [info.value for info in at.info][0] == (
        "**Description:** High quality backend profile"
    )
