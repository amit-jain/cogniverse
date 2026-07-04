"""Worker default-LM construction must go through create_dspy_lm.

The worker previously built ``dspy.LM(...)`` raw from LLM_ENDPOINT/LLM_MODEL
env, so the fallback LM (sufficient-context gate + ClaimExtractor when no
per-tenant config resolves) never got retries/timeout/seed/extra_headers and
ignored the config store's ``llm_config.primary`` entirely. These tests pin
the resolution order: config-store primary first, env fallback second, and
both paths through the ``create_dspy_lm`` factory.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_runtime.ingestion_worker.worker import _configure_dspy_lm

PRIMARY = {
    "model": "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
    "api_base": "http://vllm:8000/v1",
    "temperature": 0.2,
    "max_tokens": 512,
    "num_retries": 4,
    "request_timeout": 33.0,
    "seed": 7,
    "extra_headers": {"x-vsr-task": "kg"},
}


@pytest.fixture
def factory_capture(monkeypatch):
    """Capture the LLMEndpointConfig handed to create_dspy_lm and the LM
    handed to dspy.configure, without constructing a real dspy.LM or
    mutating global dspy settings."""
    captured = {}
    fake_lm = MagicMock(name="fake_lm")

    def _create(config):
        captured["config"] = config
        return fake_lm

    monkeypatch.setattr(
        "cogniverse_foundation.config.llm_factory.create_dspy_lm", _create
    )
    monkeypatch.setattr(
        "dspy.configure", lambda lm: captured.__setitem__("configured_lm", lm)
    )
    captured["fake_lm"] = fake_lm
    return captured


def _config_json(tmp_path, monkeypatch, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(path))
    return path


@pytest.mark.unit
class TestConfigureDspyLm:
    def test_config_store_primary_reaches_factory(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        _configure_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(**PRIMARY)
        assert factory_capture["configured_lm"] is factory_capture["fake_lm"]

    def test_config_store_primary_wins_over_env(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})
        monkeypatch.setenv("LLM_ENDPOINT", "http://other:9999/v1")
        monkeypatch.setenv("LLM_MODEL", "gemma3:4b")

        _configure_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(**PRIMARY)

    def test_env_fallback_constructs_through_factory(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {})
        monkeypatch.setenv("LLM_ENDPOINT", "http://vllm:8000/v1/")
        monkeypatch.setenv("LLM_MODEL", "gemma3:4b")

        _configure_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(
            model="openai/gemma3:4b",
            api_base="http://vllm:8000/v1",
            temperature=0.0,
        )
        assert factory_capture["configured_lm"] is factory_capture["fake_lm"]

    def test_env_fallback_keeps_existing_provider_prefix(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {})
        monkeypatch.setenv("LLM_ENDPOINT", "http://ollama:11434")
        monkeypatch.setenv("LLM_MODEL", "ollama_chat/llama3")

        _configure_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(
            model="ollama_chat/llama3",
            api_base="http://ollama:11434",
            temperature=0.0,
        )

    def test_unreachable_config_store_falls_back_to_env(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})
        monkeypatch.setenv("LLM_ENDPOINT", "http://vllm:8000/v1")
        monkeypatch.setenv("LLM_MODEL", "gemma3:4b")

        broken = MagicMock(name="config_manager")
        broken.get_system_config.side_effect = RuntimeError("config store down")

        _configure_dspy_lm(broken)

        assert factory_capture["config"] == LLMEndpointConfig(
            model="openai/gemma3:4b",
            api_base="http://vllm:8000/v1",
            temperature=0.0,
        )

    def test_nothing_available_leaves_dspy_unconfigured(
        self, tmp_path, monkeypatch, factory_capture, caplog
    ):
        _config_json(tmp_path, monkeypatch, {})
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        with caplog.at_level("WARNING"):
            _configure_dspy_lm(MagicMock(name="config_manager"))

        assert "config" not in factory_capture
        assert "configured_lm" not in factory_capture
        assert "No LM is loaded" in caplog.text
