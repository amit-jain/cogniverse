"""Unit tests for ``ConfigUtils`` key resolution.

Agents build their per-tenant DSPy LM off ``llm_base_url``; the system
config stores the LM endpoint under ``base_url``. The alias keeps the two
in sync so litellm targets the in-cluster endpoint instead of the public
OpenAI host.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_foundation.config.utils import ConfigUtils


def _utils_with_system_config(system_config: SystemConfig) -> ConfigUtils:
    cu = ConfigUtils("acme:acme", config_manager=MagicMock())
    # Pin the lazily-loaded system config so .get() resolves against it
    # without touching the backend.
    cu._system_config = system_config
    return cu


def test_llm_base_url_aliases_base_url():
    cu = _utils_with_system_config(
        SystemConfig(base_url="http://lm.cogniverse.svc:8000/v1")
    )
    assert cu.get("llm_base_url") == "http://lm.cogniverse.svc:8000/v1"
    assert cu.get("llm_base_url") == cu.get("base_url")


def test_llm_model_and_base_url_both_resolve():
    cu = _utils_with_system_config(
        SystemConfig(
            llm_model="openai/gemma3:4b",
            base_url="http://lm.cogniverse.svc:8000/v1",
        )
    )
    assert cu.get("llm_model") == "openai/gemma3:4b"
    assert cu.get("llm_base_url") == "http://lm.cogniverse.svc:8000/v1"
