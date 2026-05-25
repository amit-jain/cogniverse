"""Unit tests for the dashboard Evaluation tab's Phoenix URL resolution.

Regression guard: the tab hardcoded ``http://localhost:6006`` in 8 places,
so it broke against any non-localhost Phoenix. It must read the configured
``phoenix_url`` the app shell stores in session state.
"""

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
