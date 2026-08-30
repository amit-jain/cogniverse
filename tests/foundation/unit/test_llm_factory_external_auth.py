"""An external LLM endpoint must receive the real bearer, not the placeholder.

The chart renders ``api_key: placeholder-no-auth-needed`` for the primary and
teacher LLMs because a ConfigMap cannot hold a secret. Once those endpoints move
to Modal, which enforces auth, every call fails with AuthenticationError. The
factory is the single documented chokepoint for building a dspy.LM, so it is
where the environment bearer is resolved.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

PLACEHOLDER = "placeholder-no-auth-needed"
MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
IN_CLUSTER = "http://cogniverse-vllm-llm-student:8000/v1"


def _config(api_base: str, api_key: str | None) -> LLMEndpointConfig:
    return LLMEndpointConfig(
        model="openai/google/gemma-4-e4b-it", api_base=api_base, api_key=api_key
    )


def test_external_endpoint_gets_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    lm = create_dspy_lm(_config(MODAL, PLACEHOLDER))

    assert lm.kwargs["api_key"] == "real-bearer"


def test_in_cluster_endpoint_keeps_the_placeholder(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    lm = create_dspy_lm(_config(IN_CLUSTER, PLACEHOLDER))

    assert lm.kwargs["api_key"] == PLACEHOLDER


def test_an_explicit_key_is_not_overridden(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    lm = create_dspy_lm(_config(MODAL, "an-explicitly-configured-key"))

    assert lm.kwargs["api_key"] == "an-explicitly-configured-key"


def test_external_endpoint_without_a_bearer_fails_loudly(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(Exception) as excinfo:
        create_dspy_lm(_config(MODAL, PLACEHOLDER))

    assert "COGNIVERSE_INFERENCE_API_KEY" in str(excinfo.value)
