"""The dashboard shares one pooled runtime HTTP client across actions."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _clear_cached_client():
    import streamlit as st

    yield
    st.cache_data.clear()


def test_runtime_client_is_shared_and_pooled(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import httpx
        import streamlit as st

        from cogniverse_dashboard.utils.runtime_client import get_runtime_client

        first = get_runtime_client()
        second = get_runtime_client()
        st.session_state["_shared"] = first is second
        st.session_state["_is_client"] = isinstance(first, httpx.Client)
        st.session_state["_timeout"] = first.timeout.read
        st.session_state["_connect"] = first.timeout.connect
        """
    ).strip()
    path = tmp_path / "app_runtime_client.py"
    path.write_text(script)
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()

    assert at.session_state["_shared"] is True
    assert at.session_state["_is_client"] is True
    assert at.session_state["_timeout"] == 120.0
    assert at.session_state["_connect"] == 10.0
