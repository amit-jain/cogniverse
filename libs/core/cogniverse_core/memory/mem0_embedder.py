"""Mem0 embedder adapter for the DenseOn dense path.

Mem0's stock ``openai`` provider posts raw text to ``/v1/embeddings`` and
returns the server's vector verbatim — no sentence-transformers prompt
prefix, no L2-normalization. The retired pylate sidecar always applied
both, so routing Mem0 through it without this adapter silently changes
every stored memory vector and drifts Mem0's ``closeness`` ranking
against existing ``agent_memories`` rows.

This adapter wraps :class:`RemoteOpenAIEmbedder` (which restores the
DenseOn prompt + normalization client-side) and maps Mem0's
``memory_action`` to the query/document distinction: a ``search`` embeds
a query, an ``add``/``update`` embeds a document.
"""

from __future__ import annotations

from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

from cogniverse_core.common.models.semantic_embedder import RemoteOpenAIEmbedder

# Registered provider name passed to Mem0's embedder config block.
DENSEON_PROVIDER = "cogniverse_denseon"
_CLASS_PATH = "cogniverse_core.memory.mem0_embedder.DenseOnMem0Embedder"


def register_denseon_provider() -> None:
    """Register the DenseOn adapter with Mem0's embedder factory (idempotent).

    Two registrations are required, in order of when Mem0 consults them:

    1. ``EmbedderConfig.validate_config`` (mem0/embeddings/configs.py) is a
       pydantic ``@field_validator`` whose allowed-provider list is hardcoded
       inline; it runs first, when ``MemoryConfig`` is constructed in
       ``Memory.from_config``, and rejects any name not in that list with
       "Unsupported embedding provider". Mem0 exposes no extension hook for
       it, so we wrap the validator to also accept ``cogniverse_denseon``.
    2. ``EmbedderFactory.provider_to_class`` (mem0/utils/factory.py) maps the
       provider name to a class-path string; ``EmbedderFactory.create``
       consults it later, after validation passes.
    """
    from mem0.embeddings.configs import EmbedderConfig
    from mem0.utils.factory import EmbedderFactory

    # EmbedderFactory maps provider name -> class-path string (unlike
    # LlmFactory which uses a (path, config) tuple); load_class rsplits the
    # value directly, so it must be a bare string.
    EmbedderFactory.provider_to_class[DENSEON_PROVIDER] = _CLASS_PATH

    _allow_denseon_in_embedder_config(EmbedderConfig)


def _allow_denseon_in_embedder_config(embedder_config_cls) -> None:
    """Patch Mem0's ``EmbedderConfig`` field validator to accept DenseOn.

    The stock ``validate_config`` validator raises on any provider outside a
    hardcoded list. We replace it with a classmethod that returns early for
    ``cogniverse_denseon`` and otherwise delegates to the original, keeping the
    original ``(cls, v, values)`` v1-style signature pydantic inspects, then
    rebuild the model so the new validator is compiled into the core schema.
    Idempotent: re-running won't re-wrap.
    """
    decorators = embedder_config_cls.__pydantic_decorators__.field_validators
    decorator = decorators.get("validate_config")
    if decorator is None or getattr(decorator.func, "_denseon_patched", False):
        return

    original = decorator.func

    def validate_config(cls, v, values):
        if values.data.get("provider") == DENSEON_PROVIDER:
            return v
        return original(v, values)

    validate_config._denseon_patched = True
    bound = classmethod(validate_config)
    decorator.func = bound.__get__(None, embedder_config_cls)
    embedder_config_cls.validate_config = bound
    embedder_config_cls.model_rebuild(force=True)

    # MemoryConfig inlines EmbedderConfig's core schema at class-definition
    # time, so rebuilding the child alone leaves the parent validating against
    # the old allowlist. Memory.from_config constructs MemoryConfig, so force a
    # rebuild to propagate the new validator.
    from mem0.configs.base import MemoryConfig

    MemoryConfig.model_rebuild(force=True)


class DenseOnMem0Embedder(EmbeddingBase):
    """Mem0 embedder backed by :class:`RemoteOpenAIEmbedder`."""

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)
        if not self.config.model:
            raise ValueError("DenseOnMem0Embedder requires `model` in config")
        base_url = self.config.openai_base_url
        if not base_url:
            raise ValueError("DenseOnMem0Embedder requires `openai_base_url` in config")
        # RemoteOpenAIEmbedder appends "/v1/embeddings"; strip the "/v1"
        # suffix Mem0's config convention carries so we don't double it.
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        self._embedder = RemoteOpenAIEmbedder(base_url, self.config.model)

    def embed(
        self,
        text,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ):
        is_query = memory_action == "search"
        vec = self._embedder.encode(text, is_query=is_query)
        return vec.tolist()
