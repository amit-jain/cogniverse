"""Real-Phoenix + Streamlit AppTest for the Profile Selection sub-tab.

Two bugs kept this tab from ever rendering real data:
  1. a local ``from datetime import timedelta`` made ``timedelta`` a local of
     the render function, so the nested connectivity-probe closure raised
     NameError and the tab always reported "Telemetry provider is not available";
  2. the main span fetch called ``get_spans()`` without the required ``project``
     arg, raising TypeError once the probe was fixed.
This emits a real search span, renders the tab, and asserts it reaches the
profile-usage section.
"""

from __future__ import annotations

import textwrap
import time
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration


def test_profile_selection_tab_shows_search_spans(
    phoenix_container, telemetry_manager_with_phoenix, tmp_path
):
    manager = telemetry_manager_with_phoenix
    tenant_id = f"profsel{uuid4().hex[:8]}"
    project = f"cogniverse-{tenant_id}"

    with manager.span(
        name="video_search",
        tenant_id=tenant_id,
        attributes={"input.query": "sunset drone"},
    ):
        pass
    manager.force_flush(timeout_millis=10000)

    # Poll via the raw Phoenix client so no TelemetryProvider is cached here.
    from phoenix.client import Client

    client = Client(base_url=phoenix_container["http_endpoint"])

    def _indexed() -> bool:
        try:
            df = client.spans.get_spans_dataframe(project_identifier=project, limit=200)
        except Exception:
            return False
        return df is not None and not df.empty and (df["name"] == "video_search").any()

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not _indexed():
        time.sleep(2)
    assert _indexed(), f"video_search span not indexed in {project}"

    script = textwrap.dedent(
        f"""
        import streamlit as st
        st.session_state["current_tenant"] = "{tenant_id}"
        from cogniverse_dashboard.tabs.optimization import (
            _render_profile_selection_tab,
        )
        _render_profile_selection_tab()
        """
    ).strip()
    script_path = tmp_path / "app_profsel.py"
    script_path.write_text(script)

    at = AppTest.from_file(str(script_path), default_timeout=60)
    at.run()

    assert at.exception == [], [str(e) for e in at.exception]
    assert not any("Telemetry provider is not available" in w.value for w in at.warning)
    assert not any("Error fetching profile data" in e.value for e in at.error)
    assert any("Found 1 search spans" in s.value for s in at.success)
    assert any("Profile Usage Statistics" in m.value for m in at.subheader)
