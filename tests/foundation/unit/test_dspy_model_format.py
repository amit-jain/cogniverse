"""Unit tests for ``bare_model_name``.

The chart picks the litellm provider prefix the runtime emits via
``cogniverse.llmProviderPrefix`` in templates/_helpers.tpl (currently
``openai/`` for every backend). The runtime occasionally needs to
talk to the OpenAI-compatible HTTP API directly with a bare model
name in the request body — that's what ``bare_model_name`` is for.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.dspy import bare_model_name, ensure_provider_prefix


class TestBareModelName:
    """Strip-known-prefix-only behaviour used by the OAI-API code path."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # No slash — pass through.
            ("qwen3:4b", "qwen3:4b"),
            # Known litellm prefix — stripped.
            ("ollama/qwen3:4b", "qwen3:4b"),
            ("ollama_chat/qwen3:4b", "qwen3:4b"),
            ("hosted_vllm/Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
            ("openai/gpt-4", "gpt-4"),
            # HuggingFace org/name — leading segment is NOT a known
            # litellm provider prefix, so the slash is data and must
            # survive.
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
            ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
            # Multi-slash with known prefix only strips the first
            # segment — the HF Org/Name embedded after must survive.
            ("hosted_vllm/meta-llama/Llama-3.1", "meta-llama/Llama-3.1"),
        ],
    )
    def test_strip_only_known_prefixes(self, model: str, expected: str):
        assert bare_model_name(model) == expected


class TestEnsureProviderPrefix:
    """Bare ids must gain a provider prefix so litellm can route them."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Bare ollama tag (the engine=ollama default) — gains openai/.
            ("gemma3:4b", "openai/gemma3:4b"),
            ("qwen3:4b", "openai/qwen3:4b"),
            # HF org/name — leading segment is NOT a known prefix, so litellm
            # still needs a provider; the org slash is preserved after it.
            ("Qwen/Qwen2.5-7B", "openai/Qwen/Qwen2.5-7B"),
            ("google/gemma-4-e4b-it", "openai/google/gemma-4-e4b-it"),
            # Already prefixed with a known provider — untouched.
            ("openai/gpt-4o", "openai/gpt-4o"),
            ("ollama/llama3", "ollama/llama3"),
            ("hosted_vllm/Qwen/Qwen2.5-7B", "hosted_vllm/Qwen/Qwen2.5-7B"),
            # Empty passes through unchanged.
            ("", ""),
        ],
    )
    def test_prefix_only_unprefixed(self, model: str, expected: str):
        assert ensure_provider_prefix(model) == expected

    def test_ensure_then_bare_round_trip(self):
        # ensure_provider_prefix and bare_model_name are inverses for the
        # bare→prefixed→bare cycle the runtime relies on.
        assert bare_model_name(ensure_provider_prefix("gemma3:4b")) == "gemma3:4b"

    def test_custom_default_provider(self):
        assert (
            ensure_provider_prefix("gemma3:4b", default_provider="hosted_vllm")
            == "hosted_vllm/gemma3:4b"
        )
