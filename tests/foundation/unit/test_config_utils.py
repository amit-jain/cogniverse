"""Unit tests for ``ConfigUtils`` LM-key resolution.

Agents build their per-tenant DSPy LM from ``llm_model`` / ``llm_base_url`` /
``llm_api_key``. These resolve from config.json's complete ``llm_config.primary``
(model + in-cluster api_base + no-auth key) so litellm targets the in-cluster
endpoint; they fall back to the piecemeal system config only when primary is
unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cogniverse_foundation.config.unified_config import (
    LLMConfig,
    LLMEndpointConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import ConfigUtils


def _utils(system_config: SystemConfig) -> ConfigUtils:
    cu = ConfigUtils("acme:acme", config_manager=MagicMock())
    cu._system_config = system_config  # pin lazy system config (no backend)
    return cu


def test_llm_fields_resolve_from_primary():
    primary = LLMEndpointConfig(
        model="openai/gemma3:4b",
        api_base="http://cogniverse-llm:11434/v1",
        api_key="placeholder-no-auth-needed",
    )
    cu = _utils(SystemConfig(llm_model="bare:model", base_url=""))
    with patch.object(
        cu, "get_llm_config", return_value=LLMConfig(primary=primary, teacher=primary)
    ):
        assert cu.get("llm_model") == "openai/gemma3:4b"
        assert cu.get("llm_base_url") == "http://cogniverse-llm:11434/v1"
        assert cu.get("llm_api_key") == "placeholder-no-auth-needed"


def test_llm_base_url_falls_back_to_system_config_when_primary_missing():
    cu = _utils(SystemConfig(base_url="http://fallback:8000/v1"))
    # No config.json llm_config → get_llm_config raises → fall back.
    with patch.object(cu, "get_llm_config", side_effect=ValueError("missing")):
        assert cu.get("llm_base_url") == "http://fallback:8000/v1"


def test_empty_primary_api_base_falls_back():
    primary = LLMEndpointConfig(model="openai/gemma3:4b", api_base="")
    cu = _utils(SystemConfig(base_url="http://fallback:8000/v1"))
    with patch.object(
        cu, "get_llm_config", return_value=LLMConfig(primary=primary, teacher=primary)
    ):
        # Empty primary api_base must not shadow a usable system base_url.
        assert cu.get("llm_base_url") == "http://fallback:8000/v1"
