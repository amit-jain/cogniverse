"""FastAPI sidecar serving PyLate ColBERT per-token embeddings.

LateOn-family retrieval needs PyLate's exact encode contract — query
expansion over masked padding positions and document skiplist masking —
which generic token-embedding servers cannot reproduce because their
request schema carries no attention mask. This server runs
``pylate.models.ColBERT`` itself and exposes ``POST /pooling`` accepting
text plus ``is_query``, returning the exact per-token matrix for both
encode directions.

One process serves one pinned model; ``colbert_pylate`` (LateOn, 128-dim)
and ``code_colbert_pylate`` (LateOn-Code-edge, 48-dim) each deploy their
own instance. The module stays free of cogniverse imports so service
images can copy it alone.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("pylate_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

_MUTABLE_REVISIONS = frozenset({"latest", "main", "master"})
_DEVICES = frozenset({"cpu", "cuda", "mps"})

# Peak device memory for an encode follows the request, so the request has to
# be bounded. The canonical client chunks at 32 texts per call
# (RemoteColBERT.encode), so 256 accepts eight times its chunk while still
# rejecting a payload that could exhaust the pool. The character ceiling
# covers the other shape: few items, each enormous.
_DEFAULT_MAX_INPUT_ITEMS = 256
_DEFAULT_MAX_INPUT_CHARS = 2_000_000
# Matches sentence-transformers' own default, so this pins current behaviour
# rather than changing it, and makes the working set a configured property
# instead of a library default.
_DEFAULT_ENCODE_BATCH_SIZE = 32


class PoolingRequest(BaseModel):
    input: list[str] = Field(..., description="Texts to encode.")
    model: str | None = Field(
        None,
        description="Optional model identifier; must equal the pinned model.",
    )
    is_query: bool = Field(
        False,
        description=(
            "If true, encode with PyLate's query contract (fixed-length "
            "expansion); otherwise as a document (skiplist-masked tokens)."
        ),
    )


class PoolingResponseItem(BaseModel):
    object: str = "pooling"
    index: int
    data: list[list[float]]


class PoolingResponse(BaseModel):
    object: str = "list"
    data: list[PoolingResponseItem]
    model: str


def _load_colbert(model_name: str, model_revision: str, device: str) -> Any:
    from pylate.models import ColBERT

    logger.info(
        "Loading PyLate ColBERT model=%s revision=%s device=%s",
        model_name,
        model_revision,
        device,
    )
    model = ColBERT(
        model_name_or_path=model_name,
        revision=model_revision,
        device=device,
    )
    logger.info("PyLate ColBERT loaded: %s", model_name)
    return model


class _ModelHolder:
    """Load the pinned PyLate model once across concurrent requests."""

    def __init__(self, model_name: str, model_revision: str, device: str) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self._lock = threading.Lock()
        self._model: Any = None

    def get(self) -> Any:
        model = self._model
        if model is not None:
            return model
        with self._lock:
            if self._model is None:
                self._model = _load_colbert(
                    self.model_name,
                    self.model_revision,
                    self.device,
                )
            return self._model


def build_app(
    model_name: str,
    model_revision: str,
    device: str,
    max_input_items: int = _DEFAULT_MAX_INPUT_ITEMS,
    max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
    encode_batch_size: int = _DEFAULT_ENCODE_BATCH_SIZE,
) -> FastAPI:
    """Build the served app for one pinned model, revision, and device."""

    if not model_name or model_name != model_name.strip():
        raise ValueError("MODEL_NAME must be a non-empty canonical identifier")
    if (
        not model_revision
        or model_revision != model_revision.strip()
        or model_revision in _MUTABLE_REVISIONS
    ):
        raise ValueError("MODEL_REVISION must identify an immutable artifact")
    if device not in _DEVICES:
        raise ValueError(f"DEVICE must be one of {sorted(_DEVICES)}, got {device!r}")
    for label, bound in (
        ("MAX_INPUT_ITEMS", max_input_items),
        ("MAX_INPUT_CHARS", max_input_chars),
        ("ENCODE_BATCH_SIZE", encode_batch_size),
    ):
        if not isinstance(bound, int) or isinstance(bound, bool) or bound < 1:
            raise ValueError(f"{label} must be a positive integer, got {bound!r}")

    holder = _ModelHolder(model_name, model_revision, device)
    app = FastAPI(title="cogniverse-pylate", version="1.0")

    def _loaded_model() -> Any:
        try:
            return holder.get()
        except Exception as exc:
            logger.exception("model load failed for %s", model_name)
            raise HTTPException(
                status_code=503,
                detail=(
                    f"pylate: model {model_name} load failed "
                    f"({type(exc).__name__}): {exc}"
                ),
            ) from exc

    @app.get("/health")
    def health() -> dict:
        _loaded_model()
        # ``model`` is the key the runtime's boot probe reads to identify the
        # served model (inference_health_check._extract_model_from_health);
        # a payload without it fails startup validation for every profile
        # bound to this service.
        return {
            "status": "ready",
            "model": model_name,
            "model_revision": model_revision,
            "loaded_models": [model_name],
        }

    @app.post("/pooling", response_model=PoolingResponse)
    def pooling(request: PoolingRequest) -> PoolingResponse:
        if not request.input:
            raise HTTPException(
                status_code=400, detail="`input` must be a non-empty list"
            )
        # Reject before loading or encoding. Truncating to the limit would
        # return embeddings for a subset under a success status, which the
        # caller cannot distinguish from a complete result.
        if len(request.input) > max_input_items:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"pylate: `input` holds {len(request.input)} texts, "
                    f"limit is {max_input_items}"
                ),
            )
        total_chars = sum(len(text) for text in request.input)
        if total_chars > max_input_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"pylate: `input` holds {total_chars} characters, "
                    f"limit is {max_input_chars}"
                ),
            )
        if request.model is not None and request.model != model_name:
            raise HTTPException(
                status_code=400,
                detail=f"model must equal pinned model {model_name}",
            )
        model = _loaded_model()
        try:
            encoded = model.encode(
                request.input,
                is_query=request.is_query,
                show_progress_bar=False,
                batch_size=encode_batch_size,
            )
        except Exception as exc:
            logger.exception(
                "pooling failed (model=%s, is_query=%s)",
                model_name,
                request.is_query,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    f"pylate: model {model_name} inference failed "
                    f"({type(exc).__name__}): {exc}"
                ),
            ) from exc
        return PoolingResponse(
            data=[
                PoolingResponseItem(
                    index=index,
                    data=np.asarray(tokens, dtype=np.float32).tolist(),
                )
                for index, tokens in enumerate(encoded)
            ],
            model=model_name,
        )

    return app


def _main() -> None:
    import uvicorn

    app = build_app(
        os.environ["MODEL_NAME"],
        os.environ["MODEL_REVISION"],
        os.environ.get("DEVICE", "cpu"),
        int(os.environ.get("MAX_INPUT_ITEMS", _DEFAULT_MAX_INPUT_ITEMS)),
        int(os.environ.get("MAX_INPUT_CHARS", _DEFAULT_MAX_INPUT_CHARS)),
        int(os.environ.get("ENCODE_BATCH_SIZE", _DEFAULT_ENCODE_BATCH_SIZE)),
    )
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )


if __name__ == "__main__":
    _main()
