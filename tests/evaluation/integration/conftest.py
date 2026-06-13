"""Auto-skip integration tests when their managed services aren't available,
plus the provider-agnostic ``llm_endpoint`` fixture used by Inspect-AI-driven
tests.

The eval framework runs Inspect AI tasks via ``inspect_ai.eval(task,
model=<provider_uri>)``. The provider URI (e.g. ``openai/gpt-4o``,
``openai/gpt-4o``, ``vllm/...``) is configuration, not test code: tests must
not hard-code one. ``llm_endpoint`` resolves it from, in order:

1. Env vars ``COGNIVERSE_TEST_LLM_PROVIDER_URI`` (required) and
   ``COGNIVERSE_TEST_LLM_BASE_URL`` (optional, sets ``OPENAI_BASE_URL`` /
   ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY`` etc. for the chosen provider).
2. JSON file at ``tests/evaluation/integration/resources/test_llm.json``
   (same keys as the env vars, lower-cased).
3. ``llm_config.primary`` from ``configs/config.json`` — the same model the
   rest of the project uses (typically ``openai/gpt-4o`` at
   ``http://localhost:11434``). Skipping was the historical default but
   the project's other LLM tests all default to this same source, so
   diverging here just buried the suite behind an env var nobody set.
4. Otherwise the fixture skips with a reason explaining how to configure.
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


_PROJECT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "config.json"


def _config_path() -> Path:
    """The session config when ``COGNIVERSE_CONFIG`` is exported (the
    hermetic test-LM sidecar writes one via ``ensure_llm``), otherwise the
    project's ``configs/config.json``."""
    override = os.environ.get("COGNIVERSE_CONFIG")
    return Path(override) if override else _PROJECT_CONFIG


def _load_from_project_config() -> dict[str, Any] | None:
    """Read ``llm_config.primary`` from the active config and shape it
    into the same ``{provider_uri, base_url}`` dict the env-var path returns.

    The config carries ``model`` as the LiteLLM-style provider URI
    (``openai/gpt-4o``) and ``api_base`` as the endpoint. Inspect AI's
    provider plugins read base URLs from per-provider env vars
    (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...), which the ``llm_endpoint``
    fixture already wires up downstream — same plumbing, lower friction."""
    config_path = _config_path()
    if not config_path.exists():
        return None
    try:
        with config_path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    primary = (data.get("llm_config") or {}).get("primary") or {}
    model = primary.get("model")
    if not model:
        return None
    return {
        "provider_uri": model,
        "base_url": primary.get("api_base"),
    }


def _load_llm_config() -> dict[str, Any] | None:
    """Resolve LLM endpoint from env vars, the resources/test_llm.json file,
    or the project's configs/config.json (in that priority order)."""
    provider_uri = os.environ.get("COGNIVERSE_TEST_LLM_PROVIDER_URI")
    base_url = os.environ.get("COGNIVERSE_TEST_LLM_BASE_URL")
    if provider_uri:
        return {"provider_uri": provider_uri, "base_url": base_url}

    test_model = os.environ.get("TEST_LLM_MODEL")
    test_api_base = os.environ.get("TEST_LLM_API_BASE")
    if test_model and test_api_base:
        # ``TEST_LLM_MODEL`` from ``tests/conftest.py`` is the bare
        # model id (e.g. ``google/gemma-4-e4b-it``) — the ``openai/``
        # litellm prefix is already stripped at that layer. Don't
        # strip again: vLLM serves the model under its full
        # ``google/gemma-4-e4b-it`` id and 404s on the bare name.
        return {
            "provider_uri": f"openai/{test_model}",
            "base_url": test_api_base,
        }

    if _TEST_LLM_CONFIG.exists():
        with _TEST_LLM_CONFIG.open() as fh:
            data = json.load(fh)
        if data.get("provider_uri"):
            return {
                "provider_uri": data["provider_uri"],
                "base_url": data.get("base_url"),
            }

    return _load_from_project_config()


@pytest.fixture(scope="module")
def llm_endpoint(monkeypatch_module):
    """Provider-agnostic LLM endpoint for ``inspect_ai.eval(task, model=...)``.

    The fixture skips the test when no endpoint is configured — the test
    class does not reference any specific provider, model, or default URL.
    """
    if not os.environ.get("COGNIVERSE_TEST_LLM_PROVIDER_URI"):
        # Self-provision the hermetic test-LM sidecar; ensure_llm exports
        # COGNIVERSE_CONFIG pointing llm_config.primary at it, which the
        # config-file resolution path below then picks up. Integration
        # tests must not depend on the k3d cluster's student pod.
        from tests.utils.hermetic_llm import ensure_llm

        ensure_llm()
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
            "vllm": "OPENAI_BASE_URL",
            "anthropic": "ANTHROPIC_BASE_URL",
        }.get(provider_prefix)
        if env_var:
            monkeypatch_module.setenv(env_var, config["base_url"])
        # Inspect AI's provider client refuses to initialise without an API
        # key even against a keyless local endpoint (vLLM/Ollama).
        key_var = env_var.replace("_BASE_URL", "_API_KEY") if env_var else None
        if key_var and not os.environ.get(key_var):
            monkeypatch_module.setenv(key_var, "not-required")

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
