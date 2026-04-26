"""Auto-skip integration tests when their managed services aren't available,
plus the provider-agnostic ``llm_endpoint`` fixture used by Inspect-AI-driven
tests.

The eval framework runs Inspect AI tasks via ``inspect_ai.eval(task,
model=<provider_uri>)``. The provider URI (e.g. ``ollama/qwen3:4b``,
``openai/gpt-4o``, ``vllm/...``) is configuration, not test code: tests must
not hard-code one. ``llm_endpoint`` resolves it from, in order:

1. Env vars ``COGNIVERSE_TEST_LLM_PROVIDER_URI`` (required) and
   ``COGNIVERSE_TEST_LLM_BASE_URL`` (optional, sets ``OPENAI_BASE_URL`` /
   ``OLLAMA_HOST`` etc. for the chosen provider).
2. JSON file at ``tests/evaluation/integration/resources/test_llm.json``
   (same keys as the env vars, lower-cased).
3. Otherwise, the fixture skips the test — there is no built-in default
   provider, since different tests need chat vs. vision models and forcing
   one would couple the test class to an implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.utils.markers import is_docker_available

_TEST_LLM_CONFIG = Path(__file__).resolve().parent / "resources" / "test_llm.json"


def pytest_collection_modifyitems(items):
    docker_ok = is_docker_available()
    for item in items:
        if "requires_docker" in item.keywords and not docker_ok:
            item.add_marker(
                pytest.mark.skip(reason="Docker not available in this environment")
            )


def _load_llm_config() -> dict[str, Any] | None:
    """Resolve LLM endpoint from env vars or the resources/test_llm.json file."""
    provider_uri = os.environ.get("COGNIVERSE_TEST_LLM_PROVIDER_URI")
    base_url = os.environ.get("COGNIVERSE_TEST_LLM_BASE_URL")
    if provider_uri:
        return {"provider_uri": provider_uri, "base_url": base_url}

    if _TEST_LLM_CONFIG.exists():
        with _TEST_LLM_CONFIG.open() as fh:
            data = json.load(fh)
        if data.get("provider_uri"):
            return {
                "provider_uri": data["provider_uri"],
                "base_url": data.get("base_url"),
            }
    return None


@pytest.fixture(scope="module")
def llm_endpoint(monkeypatch_module):
    """Provider-agnostic LLM endpoint for ``inspect_ai.eval(task, model=...)``.

    The fixture skips the test when no endpoint is configured — the test
    class does not reference any specific provider, model, or default URL.
    """
    config = _load_llm_config()
    if config is None:
        pytest.skip(
            "No LLM endpoint configured. Set "
            "COGNIVERSE_TEST_LLM_PROVIDER_URI (and optionally "
            "COGNIVERSE_TEST_LLM_BASE_URL), or create "
            f"{_TEST_LLM_CONFIG.relative_to(Path(__file__).resolve().parents[3])} "
            "with a 'provider_uri' key."
        )

    # If a base_url was supplied, propagate it via the env-var conventions
    # Inspect AI / its provider plugins read. The provider is encoded in
    # provider_uri so each provider's expected env var is set without the
    # test class needing to know which one.
    if config["base_url"]:
        provider_prefix = config["provider_uri"].split("/", 1)[0].lower()
        env_var = {
            "openai": "OPENAI_BASE_URL",
            "ollama": "OLLAMA_HOST",
            "vllm": "OPENAI_BASE_URL",
            "anthropic": "ANTHROPIC_BASE_URL",
        }.get(provider_prefix)
        if env_var:
            monkeypatch_module.setenv(env_var, config["base_url"])

    return config


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest's built-in is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        yield mp
    finally:
        mp.undo()
