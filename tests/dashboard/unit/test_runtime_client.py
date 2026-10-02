"""The dashboard shares one pooled runtime HTTP client across actions."""

from __future__ import annotations

import textwrap
from pathlib import Path

import httpx
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


def _httpx_response(status: int, content: bytes) -> httpx.Response:
    return httpx.Response(
        status, content=content, headers={"Content-Type": "application/json"}
    )


TYPED_BODY = (
    b'{"detail": {"error": "store_unavailable", "message": "The pin-quota store '
    b'did not answer; retry.", "failure": "ConfigStoreUnavailableError", '
    b'"store": "pin-quota", "tenant_id": "acme:acme"}}'
)


@pytest.mark.parametrize(
    "status, content, shown",
    [
        (503, TYPED_BODY, "The pin-quota store did not answer; retry."),
        (404, b'{"detail": "Agent \'search\' not found"}', "Agent 'search' not found"),
        (
            422,
            b'{"detail": [{"loc": ["body", "query"], "msg": "Field required"}]}',
            '{"detail": [{"loc": ["body", "query"], "msg": "Field required"}]}',
        ),
        (502, b"<html>Bad Gateway</html>", "<html>Bad Gateway</html>"),
        (500, b"", ""),
    ],
    ids=["typed", "string_detail", "validation_list", "not_json", "empty"],
)
def test_a_runtime_error_shows_the_typed_message(status, content, shown) -> None:
    from cogniverse_dashboard.utils.runtime_client import runtime_error_message

    assert runtime_error_message(_httpx_response(status, content)) == shown


def test_a_requests_response_shows_the_typed_message() -> None:
    import requests

    from cogniverse_dashboard.utils.runtime_client import runtime_error_message

    response = requests.Response()
    response.status_code = 503
    response._content = TYPED_BODY
    response.headers["Content-Type"] = "application/json"

    assert runtime_error_message(response) == (
        "The pin-quota store did not answer; retry."
    )
