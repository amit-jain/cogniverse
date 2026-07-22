"""``resolve_llm_config`` — fallback path from ``config_manager`` to
the system primary LM endpoint.

Knowledge-family agents (multi-doc synth, federated query, KG traversal)
previously accepted a ``config_manager`` constructor param but never
read it. The helper gives that param a real consumer: when no explicit
``llm_config`` is passed, it derives the LM endpoint from the system
config via ``config_manager.get_system_config()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents._llm_resolution import resolve_llm_config

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_explicit_llm_config_returned_verbatim() -> None:
    explicit = MagicMock(model="gpt-4")
    assert resolve_llm_config(explicit, MagicMock()) is explicit


def test_no_config_manager_and_no_explicit_returns_none() -> None:
    assert resolve_llm_config(None, None) is None


def test_falls_back_to_config_manager_primary_endpoint() -> None:
    cm = MagicMock()
    fake_endpoint = {
        "model": "gpt-4o-mini",
        "api_base": "http://vllm.test",
        "api_key": "x",
    }
    with patch(
        "cogniverse_foundation.config.utils.get_config",
        return_value={"llm_config": {"primary": fake_endpoint}},
    ):
        resolved = resolve_llm_config(None, cm)
    assert resolved is not None
    assert resolved.model == "gpt-4o-mini"
    assert resolved.api_base == "http://vllm.test"


def test_config_manager_without_primary_endpoint_returns_none() -> None:
    cm = MagicMock()
    with patch(
        "cogniverse_foundation.config.utils.get_config",
        return_value={},
    ):
        assert resolve_llm_config(None, cm) is None


def test_explicit_overrides_config_manager_fallback() -> None:
    """Caller-supplied config wins over the system primary."""
    explicit = MagicMock(model="caller-custom-model")
    cm = MagicMock()
    with patch(
        "cogniverse_foundation.config.utils.get_config",
        return_value={"llm_config": {"primary": {"model": "system-primary"}}},
    ):
        out = resolve_llm_config(explicit, cm)
    assert out is explicit
    # The fallback path must NOT run when explicit is set.
    assert out.model == "caller-custom-model"
