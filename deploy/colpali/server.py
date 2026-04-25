"""FastAPI wrapper exposing a ColPali / ColIdefics3 / ColQwen model behind
``POST /v1/embeddings``.

Response shape matches the ``RemoteInferenceClient.process_images`` contract
in ``libs/core/cogniverse_core/common/models/model_loaders.py`` — request
carries base64-encoded images, response returns one multi-vector per image.

Pinned to one model per pod; ``MODEL_NAME`` env var selects it. Same one-pod-
one-model pattern as ``deploy/pylate``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("colpali_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


# ---------------------------------------------------------------------------
# Request / response shapes (must match RemoteInferenceClient)
# ---------------------------------------------------------------------------


class EmbeddingsRequest(BaseModel):
    images: list[str] = Field(
        default_factory=list,
        description="Base64-encoded image bytes (PNG/JPEG/...). One per item.",
    )
    input: list[str] = Field(
        default_factory=list,
        description="Optional text inputs — encoded as queries for cross-modal retrieval.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model identifier; ignored when the server is pinned to one model.",
    )


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]
    embeddings: list[list[list[float]]]
    model: str
    processing_time: float
    usage: dict[str, int]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_model_classes(model_name: str):
    """Pick the colpali-engine model + processor pair for the model name."""
    from colpali_engine.models import (
        ColIdefics3,
        ColIdefics3Processor,
        ColPali,
        ColPaliProcessor,
        ColQwen2,
        ColQwen2Processor,
    )

    name = model_name.lower()
    if "smol" in name or "idefics3" in name:
        return ColIdefics3, ColIdefics3Processor
    if "qwen" in name:
        return ColQwen2, ColQwen2Processor
    return ColPali, ColPaliProcessor


def _load_model(model_name: str, device: str):
    cls, proc_cls = _resolve_model_classes(model_name)
    logger.info(
        "Loading %s / %s for %s on %s",
        cls.__name__,
        proc_cls.__name__,
        model_name,
        device,
    )

    # bfloat16 on x86 CPU is software-emulated unless the chip has
    # AVX512_BF16 (Sapphire Rapids and later); on Apple Silicon ARM the
    # bf16 path is also slower than fp32. Default to float32 on CPU,
    # float16 on GPU. This drops a ColSmol-500m image inference from
    # 10+ min to ~30s on a laptop CPU.
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = cls.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    processor = proc_cls.from_pretrained(model_name)

    # Cap intra-op threads to the pod's CPU limit (4) so the model
    # doesn't oversubscribe on systems with many host cores. Without
    # this PyTorch grabs all detected cores and thrashes the cgroup.
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    return model, processor


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_images(model, processor, images: list[Image.Image]) -> np.ndarray:
    batch = processor.process_images(images=images).to(model.device)
    with torch.no_grad():
        out = model(**batch)
    if hasattr(out, "embeddings"):
        out = out.embeddings
    return out.to(torch.float32).cpu().numpy()


def _encode_queries(model, processor, queries: list[str]) -> np.ndarray:
    batch = processor.process_queries(queries=queries).to(model.device)
    with torch.no_grad():
        out = model(**batch)
    if hasattr(out, "embeddings"):
        out = out.embeddings
    return out.to(torch.float32).cpu().numpy()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(model_name: str, device: str) -> FastAPI:
    app = FastAPI(title="ColPali inference", version="1.0")
    model, processor = _load_model(model_name, device)
    app.state.model = model
    app.state.processor = processor
    app.state.model_name = model_name

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": app.state.model_name}

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse)
    def embeddings(req: EmbeddingsRequest) -> EmbeddingsResponse:
        if not req.images and not req.input:
            raise HTTPException(
                status_code=400,
                detail="`images` or `input` must be a non-empty list",
            )
        start = time.perf_counter()
        if req.images:
            pil_images = [_decode_image(b) for b in req.images]
            arr = _encode_images(app.state.model, app.state.processor, pil_images)
        else:
            arr = _encode_queries(app.state.model, app.state.processor, req.input)

        embeddings_list: list[list[list[float]]] = [a.tolist() for a in arr]
        elapsed = time.perf_counter() - start

        return EmbeddingsResponse(
            data=[
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(embeddings_list)
            ],
            embeddings=embeddings_list,
            model=app.state.model_name,
            processing_time=round(elapsed, 4),
            usage={
                "prompt_tokens": int(arr.shape[1]),
                "total_tokens": int(arr.shape[1]),
            },
        )

    return app


def _main() -> None:
    import uvicorn

    model_name = os.environ.get("MODEL_NAME", "vidore/colsmol-500m")
    device = os.environ.get("DEVICE", "cpu")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7997"))

    app = build_app(model_name, device)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    _main()
