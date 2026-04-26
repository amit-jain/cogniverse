"""Tests for create_dspy_lm engine resolution via SystemConfig.

Verifies the full chart→runtime→factory wiring path:
  chart sets LLM_ENGINE / LLM_MODEL  →
    main.py reads env into SystemConfig.llm_engine / llm_model  →
      create_dspy_lm reads SystemConfig and applies the matching
      DSPy/litellm prefix to the LLMEndpointConfig's bare model id.

The factory stays a single chokepoint: every dspy.LM in the codebase
goes through it, and every one of them gets the correct prefix without
the caller knowing which engine the deployment chose.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


@pytest.fixture(autouse=True)
def _reset_default_config_manager(monkeypatch):
    """Each test gets a fresh default ConfigManager.

    create_default_config_manager memoises a singleton; tests that
    mutate system_config.llm_engine bleed across test boundaries unless
    the cache is cleared. Patching the function directly avoids that.
    """
    from cogniverse_foundation.config import utils

    real_factory = utils.create_default_config_manager

    # Force a brand-new ConfigManager per test by clearing the singleton
    # holder (utils caches via a module-level reference inside
    # create_default_config_manager). The lazy import inside
    # create_dspy_lm picks up whichever instance the fixture installs.
    holder = {"cm": None}

    def _stub_factory():
        if holder["cm"] is None:
            holder["cm"] = real_factory()
        return holder["cm"]

    monkeypatch.setattr(utils, "create_default_config_manager", _stub_factory)
    yield holder


def _set_engine(holder, engine: str, model: str = "qwen3:4b"):
    cm = holder["cm"]
    if cm is None:
        from cogniverse_foundation.config import utils

        cm = utils.create_default_config_manager()
        holder["cm"] = cm
    sc = cm.get_system_config()
    sc.llm_engine = engine
    sc.llm_model = model
    cm.set_system_config(sc)


class TestFactoryEngineResolution:
    """create_dspy_lm picks the prefix from system_config.llm_engine."""

    def test_ollama_engine_applies_ollama_chat_prefix(
        self, _reset_default_config_manager
    ):
        _set_engine(_reset_default_config_manager, "ollama")
        endpoint = LLMEndpointConfig(model="qwen3:4b", api_base="http://ollama:11434")
        lm = create_dspy_lm(endpoint)
        assert lm.model == "ollama_chat/qwen3:4b", (
            "engine=ollama must produce the chat-completions litellm prefix"
        )
        assert lm.kwargs["api_base"] == "http://ollama:11434"

    def test_vllm_engine_applies_hosted_vllm_prefix_with_hf_org(
        self, _reset_default_config_manager
    ):
        _set_engine(_reset_default_config_manager, "vllm")
        endpoint = LLMEndpointConfig(
            model="Qwen/Qwen2.5-7B-Instruct",
            api_base="http://vllm:8000",
        )
        lm = create_dspy_lm(endpoint)
        assert lm.model == "hosted_vllm/Qwen/Qwen2.5-7B-Instruct", (
            "engine=vllm must keep HF Org/Name and add hosted_vllm prefix"
        )

    def test_external_engine_applies_openai_prefix(self, _reset_default_config_manager):
        _set_engine(_reset_default_config_manager, "external")
        endpoint = LLMEndpointConfig(model="gpt-4", api_base="https://api.openai.com")
        lm = create_dspy_lm(endpoint)
        assert lm.model == "openai/gpt-4"

    def test_pre_prefixed_config_is_not_double_prefixed(
        self, _reset_default_config_manager
    ):
        """A config that still ships ``ollama/qwen3:4b`` (legacy shape)
        must round-trip cleanly under engine=ollama — not produce
        ``ollama_chat/ollama/qwen3:4b``."""
        _set_engine(_reset_default_config_manager, "ollama")
        endpoint = LLMEndpointConfig(
            model="ollama/qwen3:4b", api_base="http://ollama:11434"
        )
        lm = create_dspy_lm(endpoint)
        assert lm.model == "ollama_chat/qwen3:4b"

    def test_engine_change_at_runtime_picks_up_new_prefix(
        self, _reset_default_config_manager
    ):
        """The factory reads engine on every call — flipping the config
        between calls (as a deployment does on helm upgrade) must not
        leave a stale prefix cached."""
        endpoint = LLMEndpointConfig(model="qwen3:4b", api_base="http://x")

        _set_engine(_reset_default_config_manager, "ollama")
        lm1 = create_dspy_lm(endpoint)
        assert lm1.model == "ollama_chat/qwen3:4b"

        _set_engine(_reset_default_config_manager, "vllm")
        lm2 = create_dspy_lm(endpoint)
        assert lm2.model == "hosted_vllm/qwen3:4b"

    def test_empty_model_raises(self, _reset_default_config_manager):
        _set_engine(_reset_default_config_manager, "ollama")
        endpoint = LLMEndpointConfig(model="", api_base="http://x")
        with pytest.raises(ValueError, match="model is required"):
            create_dspy_lm(endpoint)
