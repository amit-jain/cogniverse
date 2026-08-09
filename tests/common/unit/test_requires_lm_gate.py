"""Tests for the runtime gate on LM-backed integration cases."""

import json

import pytest

from tests import conftest as root_conftest
from tests.fixtures import llm as llm_fixtures
from tests.utils import hermetic_llm


class _MarkedItem:
    @staticmethod
    def get_closest_marker(name):
        return object() if name == "requires_lm" else None


def test_lm_fixture_rejects_missing_config_without_legacy_default(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(missing))
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_base_url()

    assert str(error.value) == f"Test LM config file does not exist: {missing}"


def test_lm_fixture_rejects_malformed_config(monkeypatch, tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json")
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(malformed))
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_bare_model()

    assert str(error.value) == f"Test LM config file is not valid JSON: {malformed}"


def test_lm_fixture_rejects_incomplete_primary_config(monkeypatch, tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps({"llm_config": {"primary": {"model": hermetic_llm.MODEL}}})
    )
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(incomplete))
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_base_url()

    assert str(error.value) == (
        f"Test LM config requires non-empty llm_config.primary.api_base and model: "
        f"{incomplete}"
    )


def test_lm_fixture_requires_complete_explicit_environment(monkeypatch):
    monkeypatch.setenv("TEST_LLM_API_BASE", "http://127.0.0.1:29110/v1")
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)

    with pytest.raises(ValueError) as error:
        llm_fixtures.resolve_base_url()

    assert str(error.value) == (
        "Test LM environment requires both TEST_LLM_API_BASE and TEST_LLM_MODEL"
    )


def test_lm_fixture_explicit_environment_overrides_missing_config(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("TEST_LLM_API_BASE", "http://127.0.0.1:29110/v1")
    monkeypatch.setenv("TEST_LLM_MODEL", hermetic_llm.MODEL)

    assert llm_fixtures.resolve_base_url() == "http://127.0.0.1:29110/v1"
    assert llm_fixtures.resolve_bare_model() == hermetic_llm.MODEL


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
