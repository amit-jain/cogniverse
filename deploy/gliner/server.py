"""FastAPI sidecar serving GLiNER zero-shot entity extraction.

GatewayAgent classifies queries by modality + generation_type using
GLiNER's zero-shot NER. The runtime image excludes torch/gliner by
design (heavy ML stack); this sidecar runs the model in its own pod
so the runtime stays slim.

One endpoint, ``POST /predict_entities``, mirroring the in-process
``model.predict_entities(text, labels, threshold)`` shape so
``RemoteGlinerLoader`` can replace the local loader transparently.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("gliner_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Query text")
    labels: List[str] = Field(..., min_length=1, description="Candidate label set")
    threshold: float = Field(0.4, ge=0.0, le=1.0, description="Min entity score")
    model: Optional[str] = Field(
        None,
        description=(
            "Optional GLiNER HF id to use for this request "
            "(e.g. urchade/gliner_large-v2.1). Defaults to the sidecar's "
            "MODEL_NAME env. Models are loaded on first use and cached."
        ),
    )


class EntityOut(BaseModel):
    text: str
    label: str
    score: float
    start: Optional[int] = None
    end: Optional[int] = None


class PredictResponse(BaseModel):
    entities: List[EntityOut]
    model: str


_models: dict = {}
_model_lock = threading.Lock()
_DEFAULT_MODEL = os.environ.get("MODEL_NAME", "urchade/gliner_mediumv2.1")


def _get_model(name: str):
    cached = _models.get(name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _models.get(name)
        if cached is not None:
            return cached
        from gliner import GLiNER

        logger.info("Loading GLiNER model=%s", name)
        instance = GLiNER.from_pretrained(name)
        _models[name] = instance
        logger.info("GLiNER loaded: %s", name)
        return instance


app = FastAPI(title="cogniverse-gliner", version="1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "default_model": _DEFAULT_MODEL,
        "loaded_models": sorted(_models),
    }


@app.post("/predict_entities", response_model=PredictResponse)
def predict_entities(req: PredictRequest) -> PredictResponse:
    model_name = req.model or _DEFAULT_MODEL
    try:
        model = _get_model(model_name)
    except Exception as exc:
        logger.exception("model load failed for %s", model_name)
        raise HTTPException(status_code=503, detail=f"model load failed: {exc}")

    try:
        raw = model.predict_entities(req.text, req.labels, threshold=req.threshold)
    except Exception as exc:
        logger.exception("predict_entities failed (model=%s)", model_name)
        raise HTTPException(status_code=500, detail=f"predict failed: {exc}")

    entities = [
        EntityOut(
            text=e["text"],
            label=e["label"],
            score=float(e["score"]),
            start=e.get("start"),
            end=e.get("end"),
        )
        for e in raw
    ]
    return PredictResponse(entities=entities, model=model_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
