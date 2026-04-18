"""Pluggable semantic text embedder: local SentenceTransformer or remote Ollama.

The runtime used to instantiate `SentenceTransformer` per agent
(`AudioEmbeddingGenerator`, `DocumentAgent`, ...), loading a ~400MB
all-mpnet-base-v2 copy into process memory for each instance. Across
a multi-tenant suite that exhausts the container's memory limit.

This module exposes a single `get_semantic_embedder()` that:

* Delegates to Ollama (or any OpenAI-compatible `/v1/embeddings`
  endpoint) when `COGNIVERSE_SEMANTIC_EMBED_URL` is set. Inference
  happens out of process; the runtime only holds a lightweight HTTP
  wrapper.
* Falls back to an in-process SentenceTransformer otherwise,
  preserving the existing default for single-box dev.

Instances are cached module-level — concurrent agents share one
embedder per (backend, model) pair instead of loading independent
copies. This mirrors the existing `get_or_load_model` / `get_or_load_gliner`
pattern elsewhere in this package.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

TextsT = Union[str, List[str]]


_DEFAULT_LOCAL_MODEL = "sentence-transformers/all-mpnet-base-v2"
_DEFAULT_REMOTE_MODEL = "nomic-embed-text"


class SemanticEmbedder:
    """Common interface mirroring SentenceTransformer's ``encode``.

    - Input may be a single ``str`` or a ``list[str]``.
    - Returns shape ``(D,)`` for a single string, ``(N, D)`` for a list.
    - ``normalize_embeddings`` and ``convert_to_numpy`` are accepted for
      call-site compatibility; backends always return ``np.ndarray``.
    """

    def encode(self, texts: TextsT, **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def _to_list(texts: TextsT) -> tuple[List[str], bool]:
    """Normalize to list + remember whether caller passed a single string."""
    if isinstance(texts, str):
        return [texts], True
    return list(texts), False


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32, copy=False)


class LocalSentenceTransformerEmbedder(SemanticEmbedder):
    """In-process SentenceTransformer wrapper (fallback when no remote URL)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local semantic model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def encode(self, texts: TextsT, **kwargs) -> np.ndarray:
        # SentenceTransformer handles single-string + kwargs natively.
        return self._model.encode(texts, **kwargs)


class RemoteOllamaEmbedder(SemanticEmbedder):
    """HTTP client targeting Ollama's ``/api/embed`` endpoint.

    Ollama is already running in the cluster with ``nomic-embed-text``
    loaded for Mem0; reusing it keeps the runtime container lean.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        import requests

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._session = requests.Session()
        logger.info(
            "Remote semantic embedder: %s via %s (model=%s)",
            self.__class__.__name__,
            self._base_url,
            self._model,
        )

    def encode(self, texts: TextsT, **kwargs) -> np.ndarray:
        items, single = _to_list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        resp = self._session.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": items},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(
                f"Ollama /api/embed returned no embeddings; response={data!r}"
            )
        arr = np.asarray(embeddings, dtype=np.float32)

        # Call sites pass ``normalize_embeddings=True`` to match
        # SentenceTransformer behavior. Ollama doesn't L2-normalize its
        # output, so we do it here when requested.
        if kwargs.get("normalize_embeddings"):
            arr = _l2_normalize(arr)

        if single:
            return arr[0]
        return arr


_cache: dict[str, SemanticEmbedder] = {}
_lock = threading.Lock()


def get_semantic_embedder(
    model_name: Optional[str] = None,
    remote_url: Optional[str] = None,
) -> SemanticEmbedder:
    """Return a cached semantic embedder, remote-preferred.

    Resolution order for the backend:
    1. Explicit `remote_url` argument
    2. Env var ``COGNIVERSE_SEMANTIC_EMBED_URL``
    3. Local SentenceTransformer

    Resolution order for the model name:
    1. Explicit `model_name` argument
    2. Env var ``COGNIVERSE_SEMANTIC_EMBED_MODEL``
    3. A default that matches the chosen backend
    """
    remote_url = remote_url or os.environ.get("COGNIVERSE_SEMANTIC_EMBED_URL")
    env_model = os.environ.get("COGNIVERSE_SEMANTIC_EMBED_MODEL")

    if remote_url:
        model_name = model_name or env_model or _DEFAULT_REMOTE_MODEL
        key = f"remote|{remote_url}|{model_name}"
    else:
        model_name = model_name or env_model or _DEFAULT_LOCAL_MODEL
        key = f"local|{model_name}"

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        if remote_url:
            embedder: SemanticEmbedder = RemoteOllamaEmbedder(remote_url, model_name)
        else:
            embedder = LocalSentenceTransformerEmbedder(model_name)
        _cache[key] = embedder
        return embedder


def reset_semantic_embedder_cache() -> None:
    """Clear the module-level cache (test helper)."""
    with _lock:
        _cache.clear()
