"""The e2e backend-env bridge, verified without loading the session fixtures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "e2e" / "backend_env.py"
_spec = importlib.util.spec_from_file_location("e2e_backend_env", _MODULE_PATH)
backend_env = importlib.util.module_from_spec(_spec)
sys.modules["e2e_backend_env"] = backend_env
_spec.loader.exec_module(backend_env)

DEAD_SENTINEL = "29071"


def test_splits_an_explicit_port():
    assert backend_env.backend_env_from_vespa_url("http://localhost:33080") == (
        "http://localhost",
        "33080",
    )


def test_defaults_the_port_by_scheme_rather_than_leaving_it_empty():
    assert backend_env.backend_env_from_vespa_url("https://vespa.example") == (
        "https://vespa.example",
        "443",
    )
    assert backend_env.backend_env_from_vespa_url("http://vespa.example") == (
        "http://vespa.example",
        "80",
    )


def test_refuses_a_url_it_cannot_split():
    with pytest.raises(ValueError) as excinfo:
        backend_env.backend_env_from_vespa_url("localhost:33080")

    assert "VESPA_URL" in str(excinfo.value)


def test_export_publishes_the_live_endpoint_not_the_dead_sentinel(monkeypatch):
    monkeypatch.delenv("TEST_BACKEND_URL", raising=False)
    monkeypatch.delenv("TEST_BACKEND_PORT", raising=False)
    monkeypatch.setenv("VESPA_URL", "http://localhost:33080")

    assert backend_env.export_backend_env() == ("http://localhost", "33080")
    assert backend_env.os.environ["TEST_BACKEND_PORT"] == "33080"
    assert backend_env.os.environ["TEST_BACKEND_PORT"] != DEAD_SENTINEL


def test_export_does_not_override_an_explicit_value(monkeypatch):
    monkeypatch.setenv("TEST_BACKEND_URL", "http://explicit")
    monkeypatch.setenv("TEST_BACKEND_PORT", "44444")
    monkeypatch.setenv("VESPA_URL", "http://localhost:33080")

    backend_env.export_backend_env()

    assert backend_env.os.environ["TEST_BACKEND_URL"] == "http://explicit"
    assert backend_env.os.environ["TEST_BACKEND_PORT"] == "44444"
