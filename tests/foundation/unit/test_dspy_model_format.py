"""Unit tests for the DSPy/litellm model-format helper.

Covers the three deployment engines the chart supports
(``llm.engine: ollama | vllm | external``), the idempotent strip-then-
prefix behaviour that lets pre-prefixed configs keep working, and the
HuggingFace-org passthrough that vLLM model ids depend on.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.dspy import bare_model_name, format_dspy_model


class TestFormatDspyModel:
    """Engine-prefix mapping + idempotent strip-then-re-prefix."""

    @pytest.mark.parametrize(
        "model,engine,expected",
        [
            # Bare ids — the canonical new-config shape.
            ("qwen3:4b", "ollama", "ollama_chat/qwen3:4b"),
            ("gemma3:4b", "ollama", "ollama_chat/gemma3:4b"),
            ("gpt-4", "external", "openai/gpt-4"),
            # HuggingFace ``Org/Name`` ids must NOT have ``Org/`` stripped —
            # ``Qwen`` is not a litellm provider, so the slash is data.
            (
                "Qwen/Qwen2.5-7B-Instruct",
                "vllm",
                "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
            ),
            (
                "meta-llama/Llama-3.1-8B-Instruct",
                "vllm",
                "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
            ),
            # Idempotent: a model that already carries a known DSPy prefix
            # (because the config still ships it) is stripped, then the
            # engine prefix is re-applied.
            ("ollama/qwen3:4b", "ollama", "ollama_chat/qwen3:4b"),
            ("ollama_chat/qwen3:4b", "ollama", "ollama_chat/qwen3:4b"),
            (
                "hosted_vllm/Qwen/Qwen2.5-7B",
                "vllm",
                "hosted_vllm/Qwen/Qwen2.5-7B",
            ),
            ("openai/gpt-4", "external", "openai/gpt-4"),
            # Cross-engine swap: a config carrying ollama/<m> must yield the
            # vllm prefix when the deployment runs vLLM.
            ("ollama/qwen3:4b", "vllm", "hosted_vllm/qwen3:4b"),
        ],
    )
    def test_engine_prefix_mapping(self, model: str, engine: str, expected: str):
        assert format_dspy_model(model, engine) == expected

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            format_dspy_model("", "ollama")

    def test_unknown_engine_raises_with_actionable_message(self):
        with pytest.raises(ValueError) as exc:
            format_dspy_model("qwen3:4b", "huggingface")
        # The error must list valid engines so the operator knows what to fix.
        assert "ollama" in str(exc.value)
        assert "vllm" in str(exc.value)
        assert "external" in str(exc.value)
        assert "huggingface" in str(exc.value)


class TestBareModelName:
    """Strip-known-prefix-only behaviour used by the OAI-API code path."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # No slash — pass through.
            ("qwen3:4b", "qwen3:4b"),
            # Known DSPy prefix — stripped.
            ("ollama/qwen3:4b", "qwen3:4b"),
            ("ollama_chat/qwen3:4b", "qwen3:4b"),
            ("hosted_vllm/Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
            ("openai/gpt-4", "gpt-4"),
            # HuggingFace org/name — leading segment is NOT a known DSPy
            # prefix, so the slash is data and must survive.
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
            ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
            # Multi-slash with known prefix only strips the first segment —
            # the HF Org/Name embedded after must survive.
            ("hosted_vllm/meta-llama/Llama-3.1", "meta-llama/Llama-3.1"),
        ],
    )
    def test_strip_only_known_prefixes(self, model: str, expected: str):
        assert bare_model_name(model) == expected
