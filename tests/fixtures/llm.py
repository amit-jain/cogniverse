"""Shared DSPy LM fixture for integration tests.

Resolves the test LM endpoint from environment variables so the same
test suite can run against any OpenAI-compatible provider — Ollama,
vLLM, llama.cpp, OpenAI, hosted endpoints — without code change.

Env vars:

- ``TEST_LLM_API_BASE`` — base URL of the LM endpoint.
  Default ``http://localhost:11434``. If the URL ends in ``/v1``
  the fixture treats it as pure OAI-compat (uses the ``openai/``
  litellm prefix); otherwise it uses the ``ollama/`` prefix so
  litellm hits Ollama's native ``/api/`` surface.
- ``TEST_LLM_MODEL`` — bare model name. Default ``gemma3:4b``.
- ``TEST_LLM_PROVIDER`` — explicit override (``ollama`` or
  ``openai``). Wins over the URL heuristic when set.
- ``TEST_LLM_API_KEY`` — optional. Defaults to ``not-required`` for
  local backends that don't authenticate.

Backwards-compatible env vars that still work:

- ``OLLAMA_BASE_URL`` — fallback for ``TEST_LLM_API_BASE``.
- ``OLLAMA_TEST_MODEL`` — fallback for ``TEST_LLM_MODEL``. May include
  a litellm provider prefix (``ollama/gemma3:4b``); it gets stripped
  before re-prepending the resolved prefix.
"""

from __future__ import annotations

import os

import dspy
import httpx
import pytest


def resolve_base_url() -> str:
    return (
        os.environ.get("TEST_LLM_API_BASE")
        or os.environ.get("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    )


def resolve_bare_model() -> str:
    raw = (
        os.environ.get("TEST_LLM_MODEL")
        or os.environ.get("OLLAMA_TEST_MODEL")
        or "gemma3:4b"
    )
    if "/" in raw:
        return raw.split("/", 1)[1]
    return raw


def resolve_provider() -> str:
    explicit = os.environ.get("TEST_LLM_PROVIDER")
    if explicit:
        return explicit
    if resolve_base_url().rstrip("/").endswith("/v1"):
        return "openai"
    return "ollama"


def resolve_prefixed_model() -> str:
    return f"{resolve_provider()}/{resolve_bare_model()}"


def resolve_api_key() -> str:
    return os.environ.get("TEST_LLM_API_KEY") or "not-required"


def is_test_lm_available() -> bool:
    """Return True if the configured test LM endpoint is reachable.

    Probes ``GET /api/tags`` for Ollama-compatible servers and falls
    back to ``GET /v1/models`` for pure OAI-compat endpoints. Either
    returning HTTP 200 is enough — both are cheap, idempotent.
    """
    base = resolve_base_url().rstrip("/")
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
