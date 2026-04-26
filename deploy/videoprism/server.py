"""FastAPI wrapper exposing VideoPrism behind ``POST /v1/video/embeddings``.

Request / response shape matches ``RemoteInferenceClient.process_video_segment``
in ``libs/core/cogniverse_core/common/models/model_loaders.py``:

  Request:  { video: <base64 mp4>, start_time, end_time, model }
  Response: { embeddings: [...], processing_time, model, frames_processed }

One pod per VideoPrism model — ``MODEL_NAME`` env var picks it (mirrors
the ``deploy/colpali`` pattern). VideoPrism's frame count is configurable
via ``NUM_FRAMES`` (default 16, what the public checkpoint was trained
with — temporal positional embeddings interpolate but quality drops
sharply past 32).
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("videoprism_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


# ---------------------------------------------------------------------------
# Request / response shapes (must match RemoteInferenceClient)
# ---------------------------------------------------------------------------


class VideoEmbeddingRequest(BaseModel):
    video: str = Field(..., description="Base64-encoded MP4 video segment.")
    start_time: float = Field(default=0.0, description="Segment start (seconds).")
    end_time: float = Field(default=0.0, description="Segment end (seconds).")
    model: str | None = Field(
        default=None,
        description="Optional model identifier; the server is pinned to one model.",
    )


class VideoEmbeddingResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        ..., description="Per-patch multi-vector embeddings, [num_patches, dim]."
    )
    processing_time: float
    model: str
    frames_processed: int


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


_MODEL: dict[str, Any] = {}


def _load_videoprism(model_name: str) -> None:
    """Build the JAX model + load pretrained weights once at startup. Heavy
    work — runs once and caches into module-level ``_MODEL``."""
    import jax
    from videoprism import models as vp

    logger.info("Loading VideoPrism model: %s", model_name)
    model = vp.get_model(model_name)
    state = vp.load_pretrained_weights(model_name)

    def _forward(frames):
        embeddings, _ = model.apply(state, frames, train=False)
        return embeddings

    try:
        forward = jax.jit(_forward)
        logger.info("JIT-compiled forward pass")
    except Exception as exc:  # pragma: no cover - JIT can fail on edge platforms
        logger.warning("JIT compilation failed (%s); using eager forward", exc)
        forward = _forward

    # Embedding dim and patch count are determined by the checkpoint —
    # base = 768 dim / 4096 patches, large = 1024 dim / 2048 patches.
    if "large" in model_name:
        embedding_dim, num_patches = 1024, 2048
    else:
        embedding_dim, num_patches = 768, 4096

    _MODEL.update(
        {
            "name": model_name,
            "model": model,
            "state": state,
            "forward": forward,
            "embedding_dim": embedding_dim,
            "num_patches": num_patches,
        }
    )
    logger.info("VideoPrism ready (dim=%d, patches=%d)", embedding_dim, num_patches)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _sample_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Decode ``video_path`` and sample ``num_frames`` evenly across its
    duration. Returns float32 [num_frames, 288, 288, 3] normalized to
    [0, 1]. Mirrors ``SimpleVideoPrismModel.preprocess_video``."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise HTTPException(
            status_code=400,
            detail=f"Could not read frames from video at {video_path}",
        )

    indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            # Pad with the previous frame (or zeros) so we always emit
            # exactly ``num_frames`` — VideoPrism's input shape is fixed.
            frame = (
                frames[-1].copy() if frames else np.zeros((288, 288, 3), dtype=np.uint8)
            )
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (288, 288), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return arr


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.environ.get("MODEL_NAME", "videoprism_public_v1_base_hf")
    _load_videoprism(model_name)
    yield


app = FastAPI(title="VideoPrism inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    if not _MODEL:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model": _MODEL["name"],
        "embedding_dim": _MODEL["embedding_dim"],
        "num_patches": _MODEL["num_patches"],
    }


@app.post("/v1/video/embeddings", response_model=VideoEmbeddingResponse)
def embeddings(req: VideoEmbeddingRequest) -> VideoEmbeddingResponse:
    if not _MODEL:
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()

    try:
        video_bytes = base64.b64decode(req.video)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}")

    num_frames = int(os.environ.get("NUM_FRAMES", "16"))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        frames = _sample_frames(tmp_path, num_frames)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    import jax.numpy as jnp

    batch = np.expand_dims(frames, axis=0)  # (1, T, H, W, C)
    out = _MODEL["forward"](jnp.array(batch))
    embeddings_np = np.asarray(out[0])  # drop batch dim → (P, D)

    return VideoEmbeddingResponse(
        embeddings=embeddings_np.tolist(),
        processing_time=time.perf_counter() - started,
        model=_MODEL["name"],
        frames_processed=num_frames,
    )
