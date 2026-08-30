"""Every OpenAI-compatible client resolves its key by one rule.

``create_dspy_lm`` is the chokepoint for dspy.LM construction, but Mem0's LLM
provider, ``litellm.completion``/``rerank`` and the judges' raw
chat-completions POSTs cannot be a dspy.LM. Each of them once carried its own
``api_key`` default (``None``, ``"not-required"``) and sent it to a Modal
endpoint that enforces auth, so the 401s were logged and the feature silently
degraded. ``resolve_inference_api_key`` is the rule they all share.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_foundation.config.inference_auth import endpoint_root
from cogniverse_foundation.config.llm_factory import resolve_inference_api_key

PLACEHOLDER = "placeholder-no-auth-needed"
MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
IN_CLUSTER = "http://cogniverse-vllm-llm-student:8000/v1"


@pytest.mark.parametrize(
    "api_base, api_key, expected",
    (
        (None, None, None),
        (None, "provider-sdk-key", "provider-sdk-key"),
        (IN_CLUSTER, None, "not-required"),
        (IN_CLUSTER, PLACEHOLDER, PLACEHOLDER),
        (IN_CLUSTER, "an-explicitly-configured-key", "an-explicitly-configured-key"),
        (MODAL, "an-explicitly-configured-key", "an-explicitly-configured-key"),
    ),
)
def test_resolution_without_an_environment_bearer(
    monkeypatch, api_base, api_key, expected
):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    assert resolve_inference_api_key(api_base, api_key) == expected


@pytest.mark.parametrize(
    "api_base, api_key, expected",
    (
        (None, None, None),
        (IN_CLUSTER, None, "real-bearer"),
        (IN_CLUSTER, PLACEHOLDER, "real-bearer"),
        (MODAL, None, "real-bearer"),
        (MODAL, PLACEHOLDER, "real-bearer"),
        (MODAL, "an-explicitly-configured-key", "an-explicitly-configured-key"),
    ),
)
def test_resolution_with_an_environment_bearer(
    monkeypatch, api_base, api_key, expected
):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    assert resolve_inference_api_key(api_base, api_key) == expected


@pytest.mark.parametrize("api_key", (None, PLACEHOLDER))
def test_modal_endpoint_without_a_bearer_fails_naming_the_variable(
    monkeypatch, api_key
):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        resolve_inference_api_key(MODAL, api_key)


def test_an_untrimmed_bearer_is_a_misconfiguration_even_in_cluster(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", " real-bearer ")

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        resolve_inference_api_key(IN_CLUSTER, None)


def test_concurrent_resolution_is_exact(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    with ThreadPoolExecutor(max_workers=16) as pool:
        resolved = tuple(
            pool.map(lambda _: resolve_inference_api_key(MODAL, PLACEHOLDER), range(32))
        )

    assert resolved == ("real-bearer",) * 32


@pytest.mark.parametrize(
    "url, expected",
    (
        (MODAL, "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run"),
        (IN_CLUSTER, "http://cogniverse-vllm-llm-student:8000"),
        ("http://denseon:8000/v1/embeddings", "http://denseon:8000"),
        ("http://denseon:8000", "http://denseon:8000"),
        ("http://denseon:8000/", "http://denseon:8000"),
    ),
)
def test_endpoint_root_keeps_only_scheme_and_authority(url, expected):
    assert endpoint_root(url) == expected
