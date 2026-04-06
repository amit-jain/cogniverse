"""Shared LLM config utility for integration tests.

Reads the LLM model name from configs/config.json so tests use the same
model as production. Never hardcode model names in test fixtures.
"""

import json
from pathlib import Path


def get_llm_model() -> str:
    """Get the LLM model name from config.json.

    Returns the Ollama-native model name (e.g., 'qwen3:4b'), stripping
    the 'ollama/' prefix that LiteLLM uses.
    """
    config_path = Path("configs/config.json")
    if not config_path.exists():
        return "qwen3:4b"  # sensible default if config missing

    with open(config_path) as f:
        config = json.load(f)

    model = (
        config.get("llm_config", {})
        .get("primary", {})
        .get("model", "ollama/qwen3:4b")
    )

    # Strip LiteLLM provider prefix — Mem0 uses Ollama native SDK
    if "/" in model:
        model = model.split("/", 1)[1]

    return model
