"""Unit tests for the dashboard Evaluation tab's Phoenix URL resolution.

Regression guard: the tab hardcoded ``http://localhost:6006`` in 8 places,
so it broke against any non-localhost Phoenix. It must read the configured
``phoenix_url`` the app shell stores in session state.
"""

import pytest

from cogniverse_dashboard.tabs import evaluation


def test_phoenix_base_url_uses_configured_url(monkeypatch):
    monkeypatch.setattr(
        evaluation.st, "session_state", {"phoenix_url": "http://phoenix.acme:6006"}
    )
    assert evaluation._phoenix_base_url() == "http://phoenix.acme:6006"


@pytest.mark.parametrize("session_state", [{}, {"phoenix_url": ""}])
def test_phoenix_base_url_raises_when_unconfigured(monkeypatch, session_state):
    monkeypatch.setattr(evaluation.st, "session_state", session_state)
    with pytest.raises(evaluation.PhoenixUnavailableError) as excinfo:
        evaluation._phoenix_base_url()
    assert str(excinfo.value) == (
        "Phoenix is not configured for this dashboard: the session has no "
        "phoenix_url (SystemConfig.telemetry_url)"
    )


def test_an_unconfigured_session_renders_the_datasets_error(monkeypatch):
    monkeypatch.setattr(evaluation.st, "session_state", {})
    evaluation.get_phoenix_datasets.clear()
    with pytest.raises(evaluation.PhoenixUnavailableError) as excinfo:
        evaluation.get_phoenix_datasets()
    assert str(excinfo.value) == (
        "Phoenix is not configured for this dashboard: the session has no "
        "phoenix_url (SystemConfig.telemetry_url)"
    )


class TestPhoenixFaultContract:
    """A Phoenix outage must raise (and render as an error), never return the
    same empty shape a fresh project produces; a hung Phoenix is bounded by
    the request timeout."""

    def test_graphql_raises_on_dead_endpoint(self, monkeypatch):
        from cogniverse_dashboard.tabs import evaluation as tab

        monkeypatch.setitem(
            __import__("streamlit").session_state,
            "phoenix_url",
            "http://127.0.0.1:29071",
        )
        with pytest.raises(tab.PhoenixUnavailableError, match="unreachable"):
            tab.query_phoenix_graphql("query { datasets { edges { node { id } } } }")

    def test_experiment_runs_raise_on_dead_endpoint(self, monkeypatch):
        from cogniverse_dashboard.tabs import evaluation as tab

        monkeypatch.setitem(
            __import__("streamlit").session_state,
            "phoenix_url",
            "http://127.0.0.1:29071",
        )
        with pytest.raises(tab.PhoenixUnavailableError, match="unreachable"):
            tab.get_experiment_runs("exp-1")

    def test_graphql_raises_on_error_status(self, monkeypatch):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from cogniverse_dashboard.tabs import evaluation as tab

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"boom")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setitem(
                __import__("streamlit").session_state,
                "phoenix_url",
                f"http://127.0.0.1:{port}",
            )
            with pytest.raises(tab.PhoenixUnavailableError, match="HTTP 500"):
                tab.query_phoenix_graphql("query { x }")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_experiment_data_for_dataset_raises_on_dead_endpoint(self, monkeypatch):
        """Regression: the experiment-listing fetch swallowed every
        exception into st.error and fell through to an empty result, which
        the 60s cache then served silently as "no experiments" for the rest
        of the TTL window with the error never shown again."""
        from cogniverse_dashboard.tabs import evaluation as tab

        monkeypatch.setitem(
            __import__("streamlit").session_state,
            "phoenix_url",
            "http://127.0.0.1:29071",
        )
        tab.st.cache_data.clear()
        try:
            with pytest.raises(tab.PhoenixUnavailableError, match="unreachable"):
                tab.get_all_experiment_data_for_dataset("dataset-dead")
        finally:
            tab.st.cache_data.clear()

    def test_experiment_data_for_dataset_raises_on_error_status(self, monkeypatch):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from cogniverse_dashboard.tabs import evaluation as tab

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"unavailable")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setitem(
                __import__("streamlit").session_state,
                "phoenix_url",
                f"http://127.0.0.1:{port}",
            )
            tab.st.cache_data.clear()
            with pytest.raises(tab.PhoenixUnavailableError, match="HTTP 503"):
                tab.get_all_experiment_data_for_dataset("dataset-503")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            tab.st.cache_data.clear()

    def test_requests_carry_timeouts(self):
        """Every Phoenix call in the tab is bounded — a hung Phoenix must not
        freeze the dashboard indefinitely."""
        import inspect

        from cogniverse_dashboard.tabs import evaluation as tab

        source = inspect.getsource(tab)
        calls = [
            line
            for line in source.splitlines()
            if "requests.get(" in line or "requests.post(" in line
        ]
        assert calls, "expected requests calls in the tab"
        # Multi-line calls: check the call sites via the compiled source —
        # each requests.(get|post) block must contain a timeout kwarg.
        import re as _re

        blocks = _re.findall(r"requests\.(?:get|post)\((?:[^()]|\([^()]*\))*\)", source)
        assert blocks, "expected requests call blocks"
        for block in blocks:
            assert "timeout" in block, f"unbounded Phoenix call: {block[:80]}"


def test_the_tab_renders_an_unconfigured_phoenix_as_an_error():
    from streamlit.testing.v1 import AppTest

    def _script():
        from cogniverse_dashboard.tabs.evaluation import (
            get_phoenix_datasets,
            render_evaluation_tab,
        )

        get_phoenix_datasets.clear()
        render_evaluation_tab()

    app = AppTest.from_function(_script, default_timeout=60).run()

    assert [e.message for e in app.exception] == []
    assert [e.value for e in app.error] == [
        "Cannot load datasets: Phoenix is not configured for this dashboard: "
        "the session has no phoenix_url (SystemConfig.telemetry_url)"
    ]
