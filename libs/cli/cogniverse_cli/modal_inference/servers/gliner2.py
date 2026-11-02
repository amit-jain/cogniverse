"""FastAPI server for GLiNER2: entity spans and classifications in one pass.

``POST /extract`` takes ``{model, text, entity_types: {type: description},
classifications: {task: {label: description}}, threshold}`` and answers
``{model, entities: [{text, label, score, start, end}], classifications:
{task: {label, score}}}``, the shape ``RemoteGliner2Client`` validates.

``MODEL_NAME`` is the model served: the ``gliner2`` inference spec's pinned
checkpoint by default (loaded at its pinned revision), or a local checkpoint
directory.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from cogniverse_foundation.inference_specs import get_inference_service_spec

logger = logging.getLogger("gliner2_server")

_SPEC = get_inference_service_spec("gliner2")
MODEL_ID = _SPEC.model_id
MODEL_REVISION = _SPEC.model_revision

_SERVED_MODEL = os.environ.get("MODEL_NAME") or MODEL_ID
_DEVICE = os.environ.get("DEVICE", "cpu")
if _DEVICE not in {"cpu", "cuda"}:
    raise RuntimeError("DEVICE must equal cpu or cuda")


class ExtractRequest(BaseModel):
    model: str = Field(..., description="The model this service serves")
    text: str = Field(..., min_length=1)
    entity_types: dict[str, str] = Field(
        ..., description="Entity type -> description the model reads"
    )
    classifications: dict[str, dict[str, str]] = Field(
        ..., description="Task -> {label: description}"
    )
    threshold: float = Field(..., ge=0.0, le=1.0, description="Min entity score")

    @field_validator("model")
    @classmethod
    def served_model(cls, model: str) -> str:
        if model != _SERVED_MODEL:
            raise ValueError(f"model must equal {_SERVED_MODEL}")
        return model

    @field_validator("classifications")
    @classmethod
    def two_labels_per_task(
        cls, classifications: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        for task, labels in classifications.items():
            if len(labels) < 2:
                raise ValueError(f"classification {task!r} needs at least two labels")
        return classifications


class EntityOut(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class ChoiceOut(BaseModel):
    label: str
    score: float


class ExtractResponse(BaseModel):
    model: str
    entities: list[EntityOut]
    classifications: dict[str, ChoiceOut]


def checkpoint_dir(model: str) -> str:
    """The local directory holding ``model``'s weights.

    ``AutoExtractor.from_pretrained`` forwards a ``revision`` to the config
    download only, so the pinned checkpoint is fetched as one snapshot at
    ``MODEL_REVISION`` and loaded from disk.
    """
    if model != MODEL_ID:
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=MODEL_REVISION)


_model: Any = None
_model_lock = threading.Lock()


def _get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from gliner2 import AutoExtractor

            logger.info("Loading GLiNER2 %s on %s", _SERVED_MODEL, _DEVICE)
            _model = AutoExtractor.from_pretrained(
                checkpoint_dir(_SERVED_MODEL), map_location=_DEVICE
            )
        return _model


def build_schema(model: Any, req: ExtractRequest) -> Any:
    schema = model.create_schema()
    if req.entity_types:
        schema = schema.entities(req.entity_types)
    for task, labels in req.classifications.items():
        schema = schema.classification(task, labels)
    return schema


def _query_span(span: dict[str, Any], text: str) -> tuple[int, int] | None:
    """``span``'s offsets within the request text, or None when it lies in the
    period GLiNER2 appends to text that lacks closing punctuation."""
    start, end = span["start"], min(span["end"], len(text))
    if start >= end:
        return None
    if text[start:end] != span["text"][: end - start]:
        raise ValueError(
            f"span {span['text']!r} at [{span['start']}, {span['end']}) is not "
            "in the request text"
        )
    return start, end


def to_response(raw: dict[str, Any], req: ExtractRequest) -> ExtractResponse:
    """The model's formatted output as an :class:`ExtractResponse`: entities in
    text order with offsets into ``req.text``, one per (span, label) at its
    highest score."""
    best: dict[tuple[int, int, str], float] = {}
    for label, spans in raw.get("entities", {}).items():
        for span in spans:
            offsets = _query_span(span, req.text)
            if offsets is None:
                continue
            key = (*offsets, label)
            best[key] = max(best.get(key, 0.0), float(span["confidence"]))
    entities = [
        EntityOut(
            text=req.text[start:end], label=label, score=score, start=start, end=end
        )
        for (start, end, label), score in sorted(best.items())
    ]
    classifications = {}
    for task in req.classifications:
        choice = raw.get(task)
        if (
            not isinstance(choice, dict)
            or choice.get("label") not in (req.classifications[task])
        ):
            raise ValueError(f"classification {task!r} returned {choice!r}")
        score = float(choice["confidence"])
        if not math.isfinite(score):
            raise ValueError(f"classification {task!r} score is {score}")
        classifications[task] = ChoiceOut(label=choice["label"], score=score)
    return ExtractResponse(
        model=req.model, entities=entities, classifications=classifications
    )


app = FastAPI(title="cogniverse-gliner2", version="1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _get_model()
    except Exception as exc:
        logger.exception("readiness model load failed for %s", _SERVED_MODEL)
        raise HTTPException(
            status_code=503,
            detail=(
                f"gliner2: model {_SERVED_MODEL} load failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc
    return {
        "status": "ready",
        "model": _SERVED_MODEL,
        "model_revision": MODEL_REVISION if _SERVED_MODEL == MODEL_ID else None,
    }


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    try:
        model = _get_model()
    except Exception as exc:
        logger.exception("model load failed for %s", _SERVED_MODEL)
        raise HTTPException(
            status_code=503,
            detail=(
                f"gliner2: model {_SERVED_MODEL} load failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc
    try:
        raw = model.extract(
            req.text,
            build_schema(model, req),
            threshold=req.threshold,
            include_confidence=True,
            include_spans=True,
        )
        return to_response(raw, req)
    except Exception as exc:
        logger.exception("extract failed (model=%s)", _SERVED_MODEL)
        raise HTTPException(
            status_code=500,
            detail=(
                f"gliner2: model {_SERVED_MODEL} inference failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
