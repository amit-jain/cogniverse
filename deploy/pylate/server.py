"""FastAPI sidecar serving sentence-transformers-family embedding models.

Two modes selected by the ``MODEL_MODE`` env var (default ``colbert``):

- ``colbert`` — loads via ``pylate.models.ColBERT`` and exposes
  ``POST /pooling`` returning per-token embeddings ``(N, D)``. Used by
  the document-retrieval path and ``RemoteColBERTLoader``.
- ``dense`` — loads via ``sentence_transformers.SentenceTransformer``
  and exposes ``POST /v1/embeddings`` returning a single dense vector
  per input (OpenAI-compatible response shape). Used by Mem0,
  ``SemanticEmbedder``, and any other consumer that speaks the OAI
  embeddings API.

One image, one process, one model — but the right backend and the
right endpoint for each call site. Two pod instances of this image
deploy: one in colbert mode (LateOn), one in dense mode (DenseOn).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("pylate_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class PoolingRequest(BaseModel):
    """ColBERT mode — per-token multi-vector via PyLate."""

    input: list[str] = Field(..., description="Texts to encode.")
    model: str | None = Field(
        default=None,
        description="Optional model identifier; ignored when the server is pinned to one model.",
    )
    is_query: bool = Field(
        default=False,
        description="If true, encode as a query (short, with query markers); else as a document.",
    )


class PoolingResponseItem(BaseModel):
    object: str = "pooling"
    index: int
    data: list[list[float]]


class PoolingResponse(BaseModel):
    object: str = "list"
    data: list[PoolingResponseItem]
    model: str


# Resolve the forward reference to PoolingResponseItem when the module is loaded
# via importlib.spec_from_file_location (the class lookup would otherwise fail).
PoolingResponse.model_rebuild()


class EmbeddingsRequest(BaseModel):
    """Dense mode — OpenAI-compatible ``/v1/embeddings`` request."""

    input: list[str] | str = Field(
        ..., description="Single text or list of texts to encode."
    )
    model: str | None = Field(
        default=None,
        description="Optional model identifier; ignored when the server is pinned to one model.",
    )
    # OpenAI's spec doesn't carry query/document intent. DenseOn uses
    # different prompt prefixes for the two — callers that care about
    # retrieval quality must set this. Default ``document`` matches the
    # storage path (the more frequent case for Mem0).
    is_query: bool = Field(
        default=False,
        description="If true, prepend the query: prefix; otherwise document:.",
    )


class EmbeddingsResponseItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingsResponseItem]
    model: str
    usage: EmbeddingsUsage = Field(default_factory=EmbeddingsUsage)


EmbeddingsResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _load_colbert(model_name: str, device: str):
    from pylate.models import ColBERT

    logger.info("Loading PyLate ColBERT model=%s device=%s", model_name, device)
    return ColBERT(model_name_or_path=model_name, device=device)


def _load_dense(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    logger.info(
        "Loading SentenceTransformer model=%s device=%s",
        model_name,
        device,
    )
    return SentenceTransformer(model_name, device=device)


def _encode_colbert(model: Any, texts: list[str], is_query: bool):
    """ColBERT path — list of (N_tokens, D) per text."""
    result = model.encode(texts, is_query=is_query, show_progress_bar=False)
    embeddings: list[list[list[float]]] = []
    for tokens in result:
        arr = np.asarray(tokens, dtype=np.float32)
        embeddings.append(arr.tolist())
    return embeddings


def _encode_dense(model: Any, texts: list[str], is_query: bool):
    """Dense path — one (D,) per text. Uses the model's named prompt
    table (``query`` / ``document``) when defined; falls back to
    unprompted encoding for models that don't ship a prompt table."""
    prompt_name = "query" if is_query else "document"
    try:
        result = model.encode(
            texts,
            prompt_name=prompt_name,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except (ValueError, KeyError):
        # Model has no prompt table — encode without one.
        result = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    arr = np.asarray(result, dtype=np.float32)
    return [arr[i].tolist() for i in range(arr.shape[0])]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(
    model_name: str, device: str, mode: Literal["colbert", "dense"]
) -> FastAPI:
    if mode == "colbert":
        app = FastAPI(title="PyLate ColBERT inference", version="1.0")
        model = _load_colbert(model_name, device)
    elif mode == "dense":
        app = FastAPI(title="Sentence-Transformers dense inference", version="1.0")
        model = _load_dense(model_name, device)
    else:
        raise ValueError(f"MODEL_MODE must be 'colbert' or 'dense', got {mode!r}")

    app.state.model = model
    app.state.model_name = model_name
    app.state.mode = mode

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name, "mode": mode}

    if mode == "colbert":

        @app.post("/pooling", response_model=PoolingResponse)
        def pooling(req: PoolingRequest) -> PoolingResponse:
            if not req.input:
                raise HTTPException(
                    status_code=400, detail="`input` must be a non-empty list"
                )
            embeddings = _encode_colbert(app.state.model, req.input, req.is_query)
            return PoolingResponse(
                data=[
                    PoolingResponseItem(index=i, data=emb)
                    for i, emb in enumerate(embeddings)
                ],
                model=app.state.model_name,
            )

    else:  # dense

        @app.post("/v1/embeddings", response_model=EmbeddingsResponse)
        def embeddings(req: EmbeddingsRequest) -> EmbeddingsResponse:
            texts = [req.input] if isinstance(req.input, str) else req.input
            if not texts:
                raise HTTPException(
                    status_code=400, detail="`input` must be a non-empty string or list"
                )
            vecs = _encode_dense(app.state.model, texts, req.is_query)
            return EmbeddingsResponse(
                data=[
                    EmbeddingsResponseItem(index=i, embedding=v)
                    for i, v in enumerate(vecs)
                ],
                model=app.state.model_name,
            )

    return app


def _main() -> None:
    import uvicorn

    model_name = os.environ.get("MODEL_NAME", "lightonai/LateOn")
    device = os.environ.get("DEVICE", "cpu")
    mode = os.environ.get("MODEL_MODE", "colbert").lower()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    app = build_app(model_name, device, mode)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    _main()
