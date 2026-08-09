"""Shared DSPy LM fixture for integration tests.

Resolves the test LM endpoint from a complete explicit environment pair or a
valid application config so the same suite can target any OpenAI-compatible
provider without silently changing model identity.

Env vars:

- ``TEST_LLM_API_BASE`` — base URL of the LM endpoint.
- ``TEST_LLM_MODEL`` — bare model name. It must be set together with
  ``TEST_LLM_API_BASE``.
- ``TEST_LLM_PROVIDER`` — litellm provider prefix used when building
  the prefixed model id (``<provider>/<model>``). When unset, defaults
  to ``openai`` if ``TEST_LLM_API_BASE`` ends in ``/v1`` (pure
  OAI-compat) and the litellm-routed local-server prefix otherwise.
- ``TEST_LLM_API_KEY`` — optional. Defaults to ``not-required`` for
  local LM servers that don't authenticate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import dspy
import httpx
import pytest


def _config_llm_defaults() -> tuple[str, str]:
    """Return the exact primary LM pair from a valid application config."""
    env_path = os.environ.get("COGNIVERSE_CONFIG")
    src = (
        Path(env_path)
        if env_path
        else Path(__file__).resolve().parent.parent.parent / "configs" / "config.json"
    )
    if not src.exists():
        raise ValueError(f"Test LM config file does not exist: {src}")
    try:
        config = json.loads(src.read_text())
    except OSError as exc:
        raise ValueError(f"Test LM config file could not be read: {src}") from exc
    except ValueError as exc:
        raise ValueError(f"Test LM config file is not valid JSON: {src}") from exc
    primary = (
        config.get("llm_config", {}).get("primary", {})
        if isinstance(config, dict)
        else {}
    )
    api_base = primary.get("api_base") if isinstance(primary, dict) else None
    model = primary.get("model") if isinstance(primary, dict) else None
    if (
        not isinstance(api_base, str)
        or not api_base.strip()
        or api_base != api_base.strip()
        or not isinstance(model, str)
        or not model.strip()
        or model != model.strip()
    ):
        raise ValueError(
            "Test LM config requires non-empty llm_config.primary.api_base "
            f"and model: {src}"
        )
    return api_base, model


def _explicit_llm_config() -> tuple[str, str] | None:
    api_base = os.environ.get("TEST_LLM_API_BASE")
    model = os.environ.get("TEST_LLM_MODEL")
    if api_base is None and model is None:
        return None
    if (
        not isinstance(api_base, str)
        or not api_base.strip()
        or api_base != api_base.strip()
        or not isinstance(model, str)
        or not model.strip()
        or model != model.strip()
    ):
        raise ValueError(
            "Test LM environment requires both TEST_LLM_API_BASE and TEST_LLM_MODEL"
        )
    return api_base, model


def _resolved_llm_config() -> tuple[str, str]:
    return _explicit_llm_config() or _config_llm_defaults()


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
    api_base, _ = _resolved_llm_config()
    return api_base


def resolve_bare_model() -> str:
    """Return the model name without a litellm provider prefix.

    Some configs ship the model with a prefix that names the litellm
    provider (``openai/gpt-4o``, ``hosted_vllm/Qwen/...``); only those
    leading tokens are providers we should strip. HF-style namespaced
    names (``google/gemma-4-e4b-it``, ``meta-llama/Llama-3-8B``) are
    the actual model identifier and must be preserved verbatim.
    """
    _, raw = _resolved_llm_config()
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
def dspy_test_lm(ensure_host_ollama):
    """Configure dspy with the provisioned test LM.

    Depends on ``ensure_host_ollama`` so the endpoint is provisioned and
    exported via ``TEST_LLM_*`` before ``make_dspy_lm`` resolves them —
    a collection-time injected provisioner is set up after function
    fixtures, too late for the env read here. The post-setup gate in
    tests/conftest.py fails the test when the endpoint is unreachable.

    Yields the configured ``dspy.LM`` instance and cleans up the global
    DSPy LM context on teardown.
    """
    lm = make_dspy_lm()
    dspy.configure(lm=lm)
    yield lm
    dspy.configure(lm=None)
