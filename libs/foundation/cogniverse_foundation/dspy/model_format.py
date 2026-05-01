"""Helpers for stripping a litellm provider prefix from a model id.

Some sites talk to the OpenAI-compatible HTTP API directly (Mem0's
embedder/llm wiring, the dashboard's memory tab) and expect a bare
model name in the request body, not a litellm-prefixed id. This
module owns the list of provider prefixes the codebase recognises
when stripping.
"""

from __future__ import annotations

# litellm provider prefixes recognised by this codebase. Used only as
# the lookup key for ``bare_model_name``; the chart picks the wire
# prefix the runtime emits via ``cogniverse.llmProviderPrefix`` in
# templates/_helpers.tpl.
_KNOWN_PREFIXES = frozenset({"ollama", "ollama_chat", "hosted_vllm", "openai"})


def bare_model_name(model: str) -> str:
    """Strip a leading known litellm provider prefix from ``model``.

    ``ollama/qwen3:4b`` → ``qwen3:4b``
    ``hosted_vllm/Qwen/Qwen2.5-7B-Instruct`` → ``Qwen/Qwen2.5-7B-Instruct``
    ``Qwen/Qwen2.5-7B-Instruct`` → unchanged (HF org, not a known prefix)
    ``qwen3:4b`` → unchanged (no slash)
    """
    if "/" not in model:
        return model
    head, tail = model.split("/", 1)
    if head in _KNOWN_PREFIXES:
        return tail
    return model
