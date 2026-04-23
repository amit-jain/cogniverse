"""FastAPI wrapper exposing a PyLate ColBERT model behind a /pooling endpoint.

Runs as a standalone pod. Response shape matches vLLM's ``/pooling`` output
so ``RemoteColBERTLoader`` speaks to either backend without branching.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("pylate_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


class PoolingRequest(BaseModel):
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


def _load_model(model_name: str, device: str):
    from pylate.models import ColBERT

    logger.info("Loading PyLate ColBERT model=%s device=%s", model_name, device)
    return ColBERT(model_name_or_path=model_name, device=device)


def _encode(model: Any, texts: list[str], is_query: bool) -> list[list[list[float]]]:
    """Return per-token embeddings for each text: list of (N_tokens, dim) float arrays.

    PyLate returns a list of per-text 2D arrays; coerce each to a plain Python
    nested list for JSON serialization.
    """
    result = model.encode(texts, is_query=is_query, show_progress_bar=False)
    embeddings: list[list[list[float]]] = []
    for tokens in result:
        arr = np.asarray(tokens, dtype=np.float32)
        embeddings.append(arr.tolist())
    return embeddings


def build_app(model_name: str, device: str) -> FastAPI:
    app = FastAPI(title="PyLate ColBERT inference", version="1.0")
    model = _load_model(model_name, device)
    app.state.model = model
    app.state.model_name = model_name

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": app.state.model_name}

    @app.post("/pooling", response_model=PoolingResponse)
    def pooling(req: PoolingRequest) -> PoolingResponse:
        if not req.input:
            raise HTTPException(
                status_code=400, detail="`input` must be a non-empty list"
            )
        embeddings = _encode(app.state.model, req.input, req.is_query)
        return PoolingResponse(
            data=[
                PoolingResponseItem(index=i, data=emb)
                for i, emb in enumerate(embeddings)
            ],
            model=app.state.model_name,
        )

    return app


def _main() -> None:
    import uvicorn

    model_name = os.environ.get("MODEL_NAME", "lightonai/LateOn")
    device = os.environ.get("DEVICE", "cpu")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    app = build_app(model_name, device)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    _main()
