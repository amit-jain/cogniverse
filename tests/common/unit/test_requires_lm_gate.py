"""Tests for the runtime gate on LM-backed integration cases."""

import pytest

from tests import conftest as root_conftest
from tests.fixtures import llm as llm_fixtures


class _MarkedItem:
    @staticmethod
    def get_closest_marker(name):
        return object() if name == "requires_lm" else None


def test_gate_runs_after_session_fixture_setup(monkeypatch):
    monkeypatch.setattr(llm_fixtures, "is_test_lm_available", lambda: True)

    root_conftest.pytest_runtest_setup(_MarkedItem())

    assert root_conftest.pytest_runtest_setup.pytest_impl["trylast"] is True


def test_gate_fails_with_exact_endpoint_after_unsuccessful_provision(monkeypatch):
    """Unreachable endpoint on a ``requires_lm`` test FAILS — never skips."""
    monkeypatch.setattr(llm_fixtures, "is_test_lm_available", lambda: False)
    monkeypatch.setattr(
        llm_fixtures,
        "resolve_base_url",
        lambda: "http://127.0.0.1:29999/v1",
    )

    with pytest.raises(
        pytest.fail.Exception,
        match=(
            r"Exact configured LLM endpoint not reachable "
            r"\(http://127\.0\.0\.1:29999/v1\)"
        ),
    ):
        root_conftest.pytest_runtest_setup(_MarkedItem())
