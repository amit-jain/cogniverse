"""FastAPI sidecar serving GLiNER zero-shot entity extraction.

GatewayAgent classifies queries by modality + generation_type using
GLiNER's zero-shot NER. The runtime image excludes torch/gliner by
design (heavy ML stack); this sidecar runs the model in its own pod
so the runtime stays slim.

One endpoint, ``POST /predict_entities``, mirroring the in-process
``model.predict_entities(text, labels, threshold)`` shape so
``RemoteGlinerClient`` can replace the local loader transparently.

With ``ONNX_MODEL_DIR`` set (the CPU image exports the pinned model there at
build time), inference runs on ONNX Runtime, whose graph optimizer folds the
input-independent relative-position projections DeBERTa otherwise recomputes
in every layer of every request. Otherwise the PyTorch model loads on
``DEVICE``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
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
_ONNX_MODEL_DIR = os.environ.get("ONNX_MODEL_DIR") or None
if _ONNX_MODEL_DIR is not None and _DEVICE != "cpu":
    raise RuntimeError(
        "ONNX_MODEL_DIR is served on the ONNX Runtime CPU provider; "
        "DEVICE must equal cpu"
    )
_CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
# Written next to the exported graph; names the checkpoint it came from.
ONNX_SOURCE_FILE = "source.json"


def cpu_quota_threads(cpu_max: Path) -> int | None:
    """Whole CPUs the cgroup v2 quota grants this process; None without one.

    PyTorch and ONNX Runtime size their thread pools from the host's cores,
    not from the container's quota. Sixteen threads sharing a four-CPU quota
    use it up in a quarter of each period and wait out the rest throttled.
    """
    try:
        quota, period = cpu_max.read_text().split()
    except FileNotFoundError:
        return None
    if quota == "max":
        return None
    return max(1, math.ceil(int(quota) / int(period)))


def export_onnx(model_dir: Path) -> None:
    """Export the pinned checkpoint for ONNX Runtime serving (image build step)."""
    from gliner import GLiNER

    model = GLiNER.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        map_location="cpu",
    )
    model.export_to_onnx(model_dir)
    (model_dir / ONNX_SOURCE_FILE).write_text(
        json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION})
    )


def _load_onnx(model_dir: Path, threads: int | None) -> Any:
    source = json.loads((model_dir / ONNX_SOURCE_FILE).read_text())
    if source != {"model": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError(
            f"ONNX artifact in {model_dir} was exported from "
            f"{source.get('model')}@{source.get('revision')}, not "
            f"{MODEL_ID}@{MODEL_REVISION}"
        )
    import onnxruntime as ort
    from gliner import GLiNER

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.inter_op_num_threads = 1
    if threads is not None:
        options.intra_op_num_threads = threads
    return GLiNER.from_pretrained(
        str(model_dir),
        load_onnx_model=True,
        load_tokenizer=True,
        local_files_only=True,
        session_options=options,
    )


def _load_torch(name: str, threads: int | None) -> Any:
    if threads is not None:
        import torch

        torch.set_num_threads(threads)
    from gliner import GLiNER

    return GLiNER.from_pretrained(
        name,
        revision=MODEL_REVISION,
        map_location=_DEVICE,
    )


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
        threads = cpu_quota_threads(_CGROUP_CPU_MAX)
        backend = "onnxruntime" if _ONNX_MODEL_DIR is not None else "torch"
        logger.info("Loading GLiNER model=%s backend=%s", name, backend)
        if _ONNX_MODEL_DIR is not None:
            instance = _load_onnx(Path(_ONNX_MODEL_DIR), threads)
        else:
            instance = _load_torch(name, threads)
        _models[name] = instance
        logger.info(
            "GLiNER loaded: %s backend=%s device=%s threads=%s",
            name,
            backend,
            _DEVICE,
            threads if threads is not None else "default",
        )
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
