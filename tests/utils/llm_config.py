"""Shared config utilities for integration tests.

Reads values from the active config (``COGNIVERSE_CONFIG`` env var when set,
else ``configs/config.json``) so tests use whatever the test session's
materialised config dictates. The ``cogniverse_test_config`` autouse
fixture in tests/conftest.py rewrites ``llm_config.primary`` to point at
local Ollama (or whatever ``TEST_LLM_*`` envs override to) and exports
``COGNIVERSE_CONFIG``; this module honours that.

Never hardcode model / base URL in test fixtures — pass through here.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


def _load_config() -> Dict[str, Any]:
    env_path = os.environ.get("COGNIVERSE_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            with open(path) as f:
                return json.load(f)
    config_path = Path("configs/config.json")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


def get_llm_model() -> str:
    """Return the Ollama-native LLM model name (no provider prefix)."""
    model = (
        _load_config()
        .get("llm_config", {})
        .get("primary", {})
        .get("model", "ollama/qwen3:4b")
    )
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


def get_llm_base_url() -> str:
    """Return the Ollama API base URL with /v1 suffix for OpenAI compatibility."""
    api_base = (
        _load_config()
        .get("llm_config", {})
        .get("primary", {})
        .get("api_base", "http://localhost:11434")
    )
    return api_base if api_base.endswith("/v1") else f"{api_base}/v1"


def get_backend_host() -> str:
    """Return the backend host URL (e.g. http://localhost)."""
    return _load_config().get("backend", {}).get("url", "http://localhost")


def get_memory_embedding_model() -> str:
    """Return the embedding model used by Mem0 for memory storage.

    Mem0 needs a model that produces 768-dim vectors to match the
    agent_memories Vespa schema. DenseOn (ModernBERT-based,
    768-dim, CLS pooling) served by the deploy/pylate sidecar in
    mode=dense.
    """
    memory_section = _load_config().get("memory", {})
    return memory_section.get("embedding_model", "lightonai/DenseOn")
