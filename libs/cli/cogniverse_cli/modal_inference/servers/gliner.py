"""FastAPI sidecar serving GLiNER zero-shot entity extraction.

GatewayAgent classifies queries by modality + generation_type using
GLiNER's zero-shot NER. The runtime image excludes torch/gliner by
design (heavy ML stack); this sidecar runs the model in its own pod
so the runtime stays slim.

One endpoint, ``POST /predict_entities``, mirroring the in-process
``model.predict_entities(text, labels, threshold)`` shape so
``RemoteGlinerClient`` can replace the local loader transparently.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("gliner_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


MODEL_ID = "urchade/gliner_large-v2.1"
MODEL_REVISION = "abd49a1f1ebc12af1be84d06f6848221cf96dcad"


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Query text")
    labels: list[str] = Field(..., min_length=1, description="Candidate label set")
    threshold: float = Field(0.4, ge=0.0, le=1.0, description="Min entity score")
    model: str | None = Field(
        None,
        description="Optional canonical model identifier pinned by this service.",
    )

    @field_validator("model")
    @classmethod
    def require_pinned_model(cls, model: str | None) -> str | None:
        if model is not None and model != MODEL_ID:
            raise ValueError(f"model must equal {MODEL_ID}")
        return model


class EntityOut(BaseModel):
    text: str
    label: str
    score: float
    start: int | None = None
    end: int | None = None


class PredictResponse(BaseModel):
    entities: list[EntityOut]
    model: str


_models: dict[str, Any] = {}
_model_lock = threading.Lock()

if configured_model := os.environ.get("MODEL_NAME"):
    if configured_model != MODEL_ID:
        raise RuntimeError(f"MODEL_NAME must equal pinned model {MODEL_ID}")
_DEVICE = os.environ.get("DEVICE", "cpu")
if _DEVICE not in {"cpu", "cuda"}:
    raise RuntimeError("DEVICE must equal cpu or cuda")


def _get_model(name: str) -> Any:
    if name != MODEL_ID:
        raise ValueError(f"model must equal pinned model {MODEL_ID}")
    cached = _models.get(name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _models.get(name)
        if cached is not None:
            return cached
        from gliner import GLiNER

        logger.info("Loading GLiNER model=%s", name)
        instance = GLiNER.from_pretrained(
            name,
            revision=MODEL_REVISION,
            map_location=_DEVICE,
        )
        _models[name] = instance
        logger.info("GLiNER loaded: %s", name)
        return instance


app = FastAPI(title="cogniverse-gliner", version="1.0")


@app.get("/health")
def health() -> dict:
    try:
        _get_model(MODEL_ID)
    except Exception as exc:
        logger.exception("readiness model load failed for %s", MODEL_ID)
        raise HTTPException(
            status_code=503,
            detail=(
                f"gliner: model {MODEL_ID} load failed ({type(exc).__name__}): {exc}"
            ),
        ) from exc
    return {
        # ``model`` is the key the runtime's boot probe reads to identify the
        # served model (inference_health_check._extract_model_from_health);
        # a payload without it fails startup validation for every profile
        # bound to this service.
        "status": "ready",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "loaded_models": sorted(_models),
    }


@app.post("/predict_entities", response_model=PredictResponse)
def predict_entities(req: PredictRequest) -> PredictResponse:
    model_name = req.model or MODEL_ID
    try:
        model = _get_model(model_name)
    except Exception as exc:
        logger.exception("model load failed for %s", model_name)
        raise HTTPException(
            status_code=503,
            detail=(
                f"gliner: model {model_name} load failed ({type(exc).__name__}): {exc}"
            ),
        ) from exc

    try:
        raw = model.predict_entities(req.text, req.labels, threshold=req.threshold)
    except Exception as exc:
        logger.exception("predict_entities failed (model=%s)", model_name)
        raise HTTPException(
            status_code=500,
            detail=(
                f"gliner: model {model_name} inference failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc

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
