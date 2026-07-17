"""Module-level int env parsing degrades to a default instead of crash-looping.

The queue constants (active-counter TTL, status-stream bounds) parse int env
vars at import. A malformed value used to raise ValueError at import and
crash-loop the worker; now it warns and falls back to the default.
"""

from __future__ import annotations

import pytest

from cogniverse_runtime.ingestion_worker import queue

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_parses_a_valid_value(monkeypatch):
    monkeypatch.setenv("X_TEST_INT_ENV", "42")
    assert queue._int_env("X_TEST_INT_ENV", 5) == 42


def test_defaults_when_unset():
    assert queue._int_env("X_TEST_UNSET_ENV_NAME", 7) == 7


def test_defaults_when_empty(monkeypatch):
    monkeypatch.setenv("X_TEST_EMPTY_ENV", "")
    assert queue._int_env("X_TEST_EMPTY_ENV", 3) == 3


def test_defaults_on_malformed_instead_of_raising(monkeypatch, caplog):
    monkeypatch.setenv("X_TEST_BAD_ENV", "not-a-number")
    import logging

    with caplog.at_level(logging.WARNING):
        assert queue._int_env("X_TEST_BAD_ENV", 9) == 9
    assert any("X_TEST_BAD_ENV" in r.getMessage() for r in caplog.records)
