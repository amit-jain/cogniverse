"""Pluggable semantic text embedder: local SentenceTransformer or remote Ollama.

`get_semantic_embedder()`:

* Delegates to Ollama (or any OpenAI-compatible `/v1/embeddings`
  endpoint) when `COGNIVERSE_SEMANTIC_EMBED_URL` is set. Inference
  runs out of process; the runtime only holds a lightweight HTTP
  wrapper.
* Falls back to an in-process SentenceTransformer otherwise.

Instances are cached module-level so concurrent agents share one
embedder per (backend, model) pair, mirroring `get_or_load_model`
and `get_or_load_gliner` elsewhere in this package.
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


class RemoteOpenAIEmbedder(SemanticEmbedder):
    """HTTP client targeting an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works against any server that speaks the OpenAI embeddings API shape:
    vLLM (continuous batching, parallel requests), Ollama 0.5+,
    text-embeddings-inference, Infinity. We default to this rather than
    Ollama's native ``/api/embed`` so the same client works whether the
    runtime points at a dedicated vLLM embed pod or a shared Ollama.
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
            f"{self._base_url}/v1/embeddings",
            json={"model": self._model, "input": items},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        # OpenAI-compatible response: {"data": [{"embedding": [...], "index": i}, ...]}
        rows = payload.get("data")
        if not rows:
            raise RuntimeError(
                f"/v1/embeddings returned no embeddings; response={payload!r}"
            )
        # Preserve input order (providers return "index" but often in-order already).
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        arr = np.asarray(
            [r["embedding"] for r in rows], dtype=np.float32
        )

        # Call sites pass ``normalize_embeddings=True`` to match
        # SentenceTransformer behavior. Some backends don't L2-normalize
        # their output, so we do it here when requested.
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
            embedder: SemanticEmbedder = RemoteOpenAIEmbedder(remote_url, model_name)
        else:
            embedder = LocalSentenceTransformerEmbedder(model_name)
        _cache[key] = embedder
        return embedder


def reset_semantic_embedder_cache() -> None:
    """Clear the module-level cache (test helper)."""
    with _lock:
        _cache.clear()
