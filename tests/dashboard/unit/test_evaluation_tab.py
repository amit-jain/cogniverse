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


def test_phoenix_base_url_falls_back_to_localhost_when_unset(monkeypatch):
    monkeypatch.setattr(evaluation.st, "session_state", {})
    assert evaluation._phoenix_base_url() == "http://localhost:6006"


def test_phoenix_base_url_falls_back_when_value_empty(monkeypatch):
    monkeypatch.setattr(evaluation.st, "session_state", {"phoenix_url": ""})
    assert evaluation._phoenix_base_url() == "http://localhost:6006"


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
