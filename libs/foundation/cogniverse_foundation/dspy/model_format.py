"""DSPy/litellm model-string formatting for the chosen serving engine.

The chart's ``llm.engine`` deployment switch (``ollama`` / ``vllm`` /
``external``) decides both which pod runs and which DSPy/litellm prefix
the runtime applies to ``llm.model`` when constructing ``dspy.LM(...)``.

This module centralises the prefix logic so the runtime, agents,
optimizers, and tests all agree on one mapping. Idempotent — passing a
model that already carries a known prefix strips and re-prepends, so a
config carrying ``ollama/qwen3:4b`` re-emerges correctly under either
engine.
"""

from __future__ import annotations

# Litellm/DSPy provider prefixes the runtime knows. Strings here
# correspond to the substring before the FIRST slash in a litellm model
# id. ``ollama`` and ``ollama_chat`` are both valid litellm providers
# (the latter routes through the OpenAI-compatible chat endpoint, which
# is what DSPy's ChainOfThought modules need).
_KNOWN_PREFIXES = frozenset({"ollama", "ollama_chat", "hosted_vllm", "openai"})

_ENGINE_TO_PREFIX = {
    "ollama": "ollama_chat",
    "vllm": "hosted_vllm",
    "external": "openai",
}


def format_dspy_model(model: str, engine: str) -> str:
    """Return the litellm/DSPy model string for ``model`` under ``engine``.

    Strips any leading known DSPy prefix (so a config carrying
    ``ollama/qwen3:4b`` doesn't double-prefix into
    ``ollama_chat/ollama/qwen3:4b``), then prepends the engine's prefix.
    HuggingFace-style ``Org/Name`` model ids (e.g.
    ``Qwen/Qwen2.5-7B-Instruct``) pass through untouched because their
    leading segment isn't in the known prefix set.
    """
    if not model:
        raise ValueError("model must be a non-empty string")
    if engine not in _ENGINE_TO_PREFIX:
        raise ValueError(
            f"engine must be one of {sorted(_ENGINE_TO_PREFIX)}, got {engine!r}"
        )

    bare = bare_model_name(model)
    return f"{_ENGINE_TO_PREFIX[engine]}/{bare}"


def bare_model_name(model: str) -> str:
    """Strip a leading known DSPy prefix from ``model`` if present.

    ``ollama/qwen3:4b`` → ``qwen3:4b``
    ``hosted_vllm/Qwen/Qwen2.5-7B-Instruct`` → ``Qwen/Qwen2.5-7B-Instruct``
    ``Qwen/Qwen2.5-7B-Instruct`` → unchanged (HF org, not a DSPy prefix)
    ``qwen3:4b`` → unchanged (no slash)

    Used by Mem0's bare-model normalisation and any other site that
    talks to the OpenAI-compatible API directly (which expects bare
    model names, not prefixed ids).
    """
    if "/" not in model:
        return model
    head, tail = model.split("/", 1)
    if head in _KNOWN_PREFIXES:
        return tail
    return model
