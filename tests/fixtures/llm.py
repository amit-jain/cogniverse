"""Shared DSPy LM fixture for integration tests.

Resolves the test LM endpoint from environment variables so the same
test suite can run against any OpenAI-compatible LM provider without a
code change.

Env vars:

- ``TEST_LLM_API_BASE`` — base URL of the LM endpoint.
  Default ``http://localhost:11434``.
- ``TEST_LLM_MODEL`` — bare model name. Default ``gemma3:4b``.
- ``TEST_LLM_PROVIDER`` — litellm provider prefix used when building
  the prefixed model id (``<provider>/<model>``). When unset, defaults
  to ``openai`` if ``TEST_LLM_API_BASE`` ends in ``/v1`` (pure
  OAI-compat) and the litellm-routed local-server prefix otherwise.
- ``TEST_LLM_API_KEY`` — optional. Defaults to ``not-required`` for
  local LM servers that don't authenticate.
"""

from __future__ import annotations

import os

import dspy
import httpx
import pytest

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "gemma3:4b"
_DEFAULT_LOCAL_PROVIDER = "openai"
_LITELLM_PROVIDERS = (
    "openai",
    "ollama",
    "ollama_chat",
    "hosted_vllm",
    "anthropic",
    "azure",
    "bedrock",
    "vertex_ai",
    "groq",
    "mistral",
    "cohere",
)


def resolve_base_url() -> str:
    return os.environ.get("TEST_LLM_API_BASE") or _DEFAULT_BASE_URL


def resolve_bare_model() -> str:
    """Return the model name without a litellm provider prefix.

    Some configs ship the model with a prefix that names the litellm
    provider (``openai/gpt-4o``, ``hosted_vllm/Qwen/...``); only those
    leading tokens are providers we should strip. HF-style namespaced
    names (``google/gemma-4-e4b-it``, ``meta-llama/Llama-3-8B``) are
    the actual model identifier and must be preserved verbatim.
    """
    raw = os.environ.get("TEST_LLM_MODEL") or _DEFAULT_MODEL
    head, _, _ = raw.partition("/")
    if head in _LITELLM_PROVIDERS:
        return raw.split("/", 1)[1]
    return raw


def resolve_provider() -> str:
    explicit = os.environ.get("TEST_LLM_PROVIDER")
    if explicit:
        return explicit
    if resolve_base_url().rstrip("/").endswith("/v1"):
        return "openai"
    return _DEFAULT_LOCAL_PROVIDER


def resolve_prefixed_model() -> str:
    return f"{resolve_provider()}/{resolve_bare_model()}"


def resolve_api_key() -> str:
    return os.environ.get("TEST_LLM_API_KEY") or "not-required"


def is_test_lm_available() -> bool:
    """Return True if the configured test LM endpoint is reachable.

    Probes ``GET /api/tags`` (native LM-server tag listing) and falls
    back to ``GET /v1/models`` (pure OAI-compat). Either returning
    HTTP 200 is enough — both are cheap, idempotent.

    Strips a trailing ``/v1`` from the base before probing so callers
    can pass the full endpoint URL (with or without the suffix) and
    the OAI probe still resolves to ``/v1/models`` rather than the
    nonsensical ``/v1/v1/models``.
    """
    base = resolve_base_url().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    for path in ("/api/tags", "/v1/models"):
        try:
            r = httpx.get(f"{base}{path}", timeout=5.0)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            continue
    return False


def make_dspy_lm() -> dspy.LM:
    """Construct a ``dspy.LM`` from the resolved env config."""
    return dspy.LM(
        model=resolve_prefixed_model(),
        api_base=resolve_base_url(),
        api_key=resolve_api_key(),
    )


@pytest.fixture
def dspy_test_lm():
    """Configure dspy with a real LM, skipping when the endpoint is down.

    Yields the configured ``dspy.LM`` instance and cleans up the global
    DSPy LM context on teardown.
    """
    if not is_test_lm_available():
        pytest.skip(f"Test LM not available at {resolve_base_url()}")
    lm = make_dspy_lm()
    dspy.configure(lm=lm)
    yield lm
    dspy.configure(lm=None)
