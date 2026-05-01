"""Tests for create_dspy_lm.

The factory passes ``LLMEndpointConfig.model`` through to dspy.LM
verbatim — no string manipulation, no provider rewriting. The chart
(or whatever else builds the LLMEndpointConfig) is responsible for
populating ``model`` with whatever litellm-recognised string should be
sent. The factory's only job is to wire api_base / api_key /
extra_body / sampling onto the dspy.LM and emit one well-formed log
line per construction.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


class TestModelPassthrough:
    """Whatever model id the caller hands in is what dspy.LM gets."""

    @pytest.mark.parametrize(
        "model",
        [
            "qwen3:4b",
            "ollama_chat/qwen3:4b",
            "hosted_vllm/google/gemma-4-e4b-it",
            "hosted_vllm/cyankiwi/Qwen3.6-27B-AWQ-INT4",
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-sonnet-20241022",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
    )
    def test_model_passes_through_unchanged(self, model):
        endpoint = LLMEndpointConfig(model=model, api_base="http://endpoint:8000/v1")
        lm = create_dspy_lm(endpoint)
        assert lm.model == model

    def test_endpoint_kwargs_are_wired_onto_dspy_lm(self):
        endpoint = LLMEndpointConfig(
            model="hosted_vllm/test/Model",
            api_base="http://endpoint:8000/v1",
            api_key="sk-test",
            temperature=0.42,
            max_tokens=2048,
            extra_body={"reasoning": "auto"},
        )
        lm = create_dspy_lm(endpoint)
        assert lm.model == "hosted_vllm/test/Model"
        assert lm.kwargs["api_base"] == "http://endpoint:8000/v1"
        assert lm.kwargs["api_key"] == "sk-test"
        assert lm.kwargs["temperature"] == 0.42
        assert lm.kwargs["max_tokens"] == 2048
        assert lm.kwargs["extra_body"] == {"reasoning": "auto"}

    def test_optional_kwargs_omitted_when_none(self):
        endpoint = LLMEndpointConfig(model="test/Model")
        lm = create_dspy_lm(endpoint)
        assert "api_base" not in lm.kwargs
        assert "api_key" not in lm.kwargs
        assert "extra_body" not in lm.kwargs

    def test_empty_model_raises(self):
        endpoint = LLMEndpointConfig(model="", api_base="http://x")
        with pytest.raises(ValueError, match="model is required"):
            create_dspy_lm(endpoint)
