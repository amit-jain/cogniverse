"""FastAPI server for fine-tuned query classifier checkpoints.

The gateway runs a query classifier beside its rules; the runtime image has
no torch, so the model runs here. ``MODEL_ROOT`` holds one checkpoint
directory per model, written by ``cogniverse_finetuning``'s query classifier
trainer; a request names its checkpoint by directory name.

``POST /classify`` answers ``{model, heads, predictions}`` in the shape
``RemoteQueryClassifierClient`` validates.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cogniverse_foundation.query_classifier import QueryClassifier

logger = logging.getLogger("query_classifier_server")

_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEVICE = os.environ.get("DEVICE", "cpu")


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Query text")
    model: str = Field(..., min_length=1, description="Checkpoint directory name")


class Prediction(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    model: str
    heads: dict[str, list[str]]
    predictions: dict[str, Prediction]


_models: dict[str, QueryClassifier] = {}
_model_lock = threading.Lock()


def model_root() -> Path:
    root = os.environ.get("MODEL_ROOT")
    if not root:
        raise RuntimeError("MODEL_ROOT must name the checkpoint directory root")
    return Path(root)


def _get_model(name: str) -> QueryClassifier:
    if not _MODEL_NAME.match(name):
        raise ValueError(f"model {name!r} must be a checkpoint directory name")
    cached = _models.get(name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _models.get(name)
        if cached is not None:
            return cached
        logger.info("Loading query classifier %s on %s", name, _DEVICE)
        instance = QueryClassifier.load(model_root() / name, device=_DEVICE)
        _models[name] = instance
        return instance


app = FastAPI(title="cogniverse-query-classifier", version="1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    root = model_root()
    if not root.is_dir():
        raise HTTPException(status_code=503, detail=f"MODEL_ROOT {root} is missing")
    return {
        "status": "ready",
        "models": sorted(p.name for p in root.iterdir() if p.is_dir()),
        "loaded_models": sorted(_models),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    try:
        model = _get_model(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("query classifier load failed for %s", req.model)
        raise HTTPException(
            status_code=503,
            detail=(
                f"query classifier {req.model} load failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc
    try:
        predictions = model.classify(req.text)
    except Exception as exc:
        logger.exception("query classifier %s inference failed", req.model)
        raise HTTPException(
            status_code=500,
            detail=(
                f"query classifier {req.model} inference failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc
    return ClassifyResponse(
        model=req.model,
        heads=model.heads,
        predictions={name: Prediction(**pred) for name, pred in predictions.items()},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
