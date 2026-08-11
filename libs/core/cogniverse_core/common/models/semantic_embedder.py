"""Pluggable semantic text embedder: local SentenceTransformer or remote OAI-compat embedder.

`get_semantic_embedder()`:

* Delegates to any OpenAI-compatible `/v1/embeddings`
  endpoint) when `COGNIVERSE_SEMANTIC_EMBED_URL` is set. Inference
  runs out of process; the runtime only holds a lightweight HTTP
  wrapper.
* Falls back to an in-process SentenceTransformer otherwise.

Instances are cached module-level so concurrent agents share one
embedder per (backend, model) pair, mirroring `get_or_load_model`
and `get_or_load_gliner` elsewhere in this package.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import List, Mapping, Optional, Union

import numpy as np

from cogniverse_foundation.config.inference_auth import inference_headers
from cogniverse_foundation.config.inference_service import (
    require_in_process_backend,
)

logger = logging.getLogger(__name__)

TextsT = Union[str, List[str]]


_DEFAULT_LOCAL_MODEL = "sentence-transformers/all-mpnet-base-v2"
_DEFAULT_REMOTE_MODEL = "lightonai/DenseOn"

# DenseOn's config_sentence_transformers.json carries
# prompts={"query": "query: ", "document": "document: "} and the pylate
# sidecar always applied them plus normalize_embeddings=True. Stock vLLM
# /v1/embeddings applies neither, so the remote path must restore both
# client-side or every stored memory vector silently drifts.
_DENSEON_QUERY_PROMPT = "query: "
_DENSEON_DOCUMENT_PROMPT = "document: "


def _canonical_bearer_headers(
    headers: Optional[Mapping[str, str]],
) -> dict[str, str]:
    if not headers:
        return {}
    if set(headers) != {"Authorization"}:
        raise ValueError("headers must contain only Authorization")
    authorization = headers["Authorization"]
    scheme, separator, token = authorization.partition(" ")
    if scheme != "Bearer" or not separator or not token or token != token.strip():
        raise ValueError("headers Authorization must be a canonical bearer value")
    return {"Authorization": authorization}


def _resolved_inference_headers(
    base_url: str,
    headers: Optional[Mapping[str, str]],
) -> Mapping[str, str]:
    configured_headers = inference_headers(base_url)
    explicit_headers = _canonical_bearer_headers(headers)
    if configured_headers and explicit_headers:
        raise ValueError("headers must not be supplied for a Modal endpoint")
    return configured_headers or explicit_headers


class SemanticEmbedder:
    """Common interface mirroring SentenceTransformer's ``encode``.

    - Input may be a single ``str`` or a ``list[str]``.
    - Returns shape ``(D,)`` for a single string, ``(N, D)`` for a list.
    - ``normalize_embeddings`` and ``convert_to_numpy`` are accepted for
      call-site compatibility; backends always return ``np.ndarray``.
    """

    def encode(
        self, texts: TextsT, is_query: bool = False, **kwargs
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _close(self) -> None:
        """Release resources owned by the embedder."""


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
        require_in_process_backend("denseon", module="sentence_transformers")

        from sentence_transformers import SentenceTransformer

        logger.info("Loading local semantic model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: TextsT, is_query: bool = False, **kwargs) -> np.ndarray:
        # SentenceTransformer reads the model's own prompts from
        # config_sentence_transformers.json via prompt_name, matching the
        # query/document prefixes DenseOn ships with.
        prompt_name = "query" if is_query else "document"
        try:
            return self._model.encode(texts, prompt_name=prompt_name, **kwargs)
        except (ValueError, KeyError):
            # Models without registered prompts (e.g. all-mpnet-base-v2)
            # reject prompt_name; fall back to plain encode.
            return self._model.encode(texts, **kwargs)


class RemoteOpenAIEmbedder(SemanticEmbedder):
    """HTTP client targeting an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works against any server that speaks the OpenAI embeddings API shape:
    vLLM (continuous batching, parallel requests), OAI-compat embedding servers,
    text-embeddings-inference, Infinity. We default to this rather than
    the same OAI-compat client works whether the
    runtime points at a dedicated vLLM embed pod or a shared embedding service.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        query_prompt: str = _DENSEON_QUERY_PROMPT,
        document_prompt: str = _DENSEON_DOCUMENT_PROMPT,
        headers: Optional[Mapping[str, str]] = None,
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ):
        import requests

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._query_prompt = query_prompt
        self._document_prompt = document_prompt
        if _resolved_headers is not None:
            if headers is not None:
                raise ValueError("headers and _resolved_headers are mutually exclusive")
            self._headers = _resolved_headers
        else:
            self._headers = _resolved_inference_headers(self._base_url, headers)
        self._session = requests.Session()
        logger.info(
            "Remote semantic embedder: %s via %s (model=%s)",
            self.__class__.__name__,
            self._base_url,
            self._model,
        )

    def _close(self) -> None:
        self._session.close()

    def encode(
        self,
        texts: TextsT,
        is_query: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        # convert_to_numpy / normalize_embeddings are accepted for
        # SentenceTransformer call-site compatibility; this backend always
        # returns a normalized np.ndarray, so they are no-ops. Anything else
        # would be silently dropped here (the local sibling forwards its
        # kwargs to SentenceTransformer), so reject it loudly.
        if kwargs:
            raise TypeError(
                "RemoteOpenAIEmbedder.encode() got unexpected keyword "
                f"arguments: {sorted(kwargs)}"
            )
        items, single = _to_list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        # Restore the sentence-transformers prompt prefix the pylate sidecar
        # applied. Stock vLLM /v1/embeddings does not, so prepend it here so
        # the stored/queried vectors match the historical DenseOn embeddings.
        prompt = self._query_prompt if is_query else self._document_prompt
        prefixed = [f"{prompt}{text}" for text in items]

        import requests

        url = f"{self._base_url}/v1/embeddings"
        try:
            resp = self._session.post(
                url,
                json={"model": self._model, "input": prefixed},
                headers=self._headers or None,
                timeout=self._timeout,
            )
        except requests.Timeout as exc:
            raise type(exc)(
                f"{self._model} request to {url} timed out after "
                f"{self._timeout} seconds: {exc}"
            ) from exc
        except requests.ConnectionError as exc:
            raise requests.ConnectionError(
                f"{self._model} request to {url} failed: {exc}"
            ) from exc
        resp.raise_for_status()
        payload = resp.json()

        response_context = f"{self._model} response from {url}"
        if not isinstance(payload, dict):
            raise RuntimeError(f"{response_context}: expected a JSON object")
        rows = payload.get("data")
        if rows is None or rows == []:
            raise RuntimeError(
                f"{response_context}: /v1/embeddings returned no embeddings; "
                f"response={payload!r}"
            )
        if not isinstance(rows, list):
            raise RuntimeError(f"{response_context}: data must be a list")
        if len(rows) != len(items):
            raise RuntimeError(
                f"{response_context}: expected {len(items)} rows, got {len(rows)}"
            )

        indexed_vectors: list[tuple[int, list[float]]] = []
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"{response_context}: row {position} must be a JSON object"
                )
            index = row.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RuntimeError(
                    f"{response_context}: row {position} index must be an integer"
                )
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError(
                    f"{response_context}: every embedding must have the same "
                    "non-zero dimension"
                )
            indexed_vectors.append((index, embedding))

        expected_indices = list(range(len(items)))
        actual_indices = sorted(index for index, _ in indexed_vectors)
        if actual_indices != expected_indices:
            raise RuntimeError(
                f"{response_context}: indices must be exactly {expected_indices}; "
                f"got {actual_indices}"
            )
        dimensions = {len(vector) for _, vector in indexed_vectors}
        if len(dimensions) != 1:
            raise RuntimeError(
                f"{response_context}: every embedding must have the same "
                "non-zero dimension"
            )
        indexed_vectors.sort(key=lambda item: item[0])
        try:
            arr = np.asarray(
                [vector for _, vector in indexed_vectors],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{response_context}: embeddings must contain numeric values"
            ) from exc
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{response_context}: embeddings must be finite")
        if np.any(np.linalg.norm(arr, axis=1) == 0):
            raise RuntimeError(
                f"{response_context}: embeddings must have non-zero norm"
            )

        # DenseOn was always served with normalize_embeddings=True. Stock
        # vLLM may not normalize, so always L2-normalize here; it is
        # idempotent on an already-unit vector and safe if vLLM later does.
        arr = _l2_normalize(arr)

        if single:
            return arr[0]
        return arr


_CACHE_CAPACITY = 16
_cache: OrderedDict[str, SemanticEmbedder] = OrderedDict()
_lock = threading.Lock()


def _close_embedder(embedder: SemanticEmbedder) -> None:
    embedder._close()


def _store_cached_embedder(key: str, embedder: SemanticEmbedder) -> None:
    displaced_key: Optional[str] = None
    displaced: Optional[SemanticEmbedder] = None
    restore_as_lru = False
    if key in _cache:
        displaced_key = key
        displaced = _cache.pop(key)
    elif len(_cache) >= _CACHE_CAPACITY:
        displaced_key, displaced = _cache.popitem(last=False)
        restore_as_lru = True

    if displaced is not None:
        try:
            _close_embedder(displaced)
        except Exception as exc:
            _cache[displaced_key] = displaced
            if restore_as_lru:
                _cache.move_to_end(displaced_key, last=False)
            replacement_error = None
            try:
                _close_embedder(embedder)
            except Exception as cleanup_exc:
                replacement_error = cleanup_exc
            detail = (
                "semantic embedder cache eviction failed to close "
                f"{displaced_key!r}: {exc}"
            )
            if replacement_error is not None:
                detail += f"; replacement cleanup failed: {replacement_error}"
            raise RuntimeError(detail) from exc

    _cache[key] = embedder
    _cache.move_to_end(key)


def get_semantic_embedder(
    model_name: Optional[str] = None,
    remote_url: Optional[str] = None,
    headers: Optional[Mapping[str, str]] = None,
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
    canonical_headers: Mapping[str, str] = {}

    if remote_url:
        canonical_headers = _resolved_inference_headers(remote_url, headers)
        model_name = model_name or env_model or _DEFAULT_REMOTE_MODEL
        authorization = canonical_headers.get("Authorization", "")
        credential_fingerprint = hashlib.sha256(authorization.encode()).hexdigest()
        key = f"remote|{remote_url}|{model_name}|{credential_fingerprint}"
    else:
        if headers:
            raise ValueError("headers require a remote_url")
        model_name = model_name or env_model or _DEFAULT_LOCAL_MODEL
        key = f"local|{model_name}"

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
        if remote_url:
            embedder: SemanticEmbedder = RemoteOpenAIEmbedder(
                remote_url,
                model_name,
                _resolved_headers=canonical_headers,
            )
        else:
            embedder = LocalSentenceTransformerEmbedder(model_name)
        _store_cached_embedder(key, embedder)
        return embedder


def reset_semantic_embedder_cache() -> None:
    """Clear the module-level cache (test helper)."""
    with _lock:
        failures: list[tuple[str, Exception]] = []
        retained: OrderedDict[str, SemanticEmbedder] = OrderedDict()
        for key, embedder in _cache.items():
            try:
                _close_embedder(embedder)
            except Exception as exc:
                retained[key] = embedder
                failures.append((key, exc))
        _cache.clear()
        _cache.update(retained)
        if failures:
            details = "; ".join(f"{key!r}: {exc}" for key, exc in failures)
            raise RuntimeError(
                f"semantic embedder cache reset failed to close: {details}"
            ) from failures[0][1]
