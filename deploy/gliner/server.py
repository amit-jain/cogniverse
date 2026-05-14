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


class EntityOut(BaseModel):
    text: str
    label: str
    score: float
    start: Optional[int] = None
    end: Optional[int] = None


class PredictResponse(BaseModel):
    entities: List[EntityOut]
    model: str


_model: Any = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from gliner import GLiNER

        name = os.environ.get("MODEL_NAME", "urchade/gliner_large-v2.1")
        logger.info("Loading GLiNER model=%s", name)
        _model = GLiNER.from_pretrained(name)
        logger.info("GLiNER loaded: %s", name)
        return _model


app = FastAPI(title="cogniverse-gliner", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": os.environ.get("MODEL_NAME", "")}


@app.post("/predict_entities", response_model=PredictResponse)
def predict_entities(req: PredictRequest) -> PredictResponse:
    try:
        model = _get_model()
    except Exception as exc:
        logger.exception("model load failed")
        raise HTTPException(status_code=503, detail=f"model load failed: {exc}")

    try:
        raw = model.predict_entities(req.text, req.labels, threshold=req.threshold)
    except Exception as exc:
        logger.exception("predict_entities failed")
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
    return PredictResponse(
        entities=entities, model=os.environ.get("MODEL_NAME", "")
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
