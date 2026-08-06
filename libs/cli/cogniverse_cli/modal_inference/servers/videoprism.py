"""FastAPI wrapper exposing VideoPrism behind ``POST /v1/video/embeddings``.

Request / response shape matches ``RemoteInferenceClient.process_video_segment``
in ``libs/core/cogniverse_core/common/models/model_loaders.py``:

  Request:  { video: <base64 mp4>, start_time, end_time, model }
  Response: { embeddings: [...], processing_time, model, frames_processed }

The service accepts only the pinned public base checkpoint. VideoPrism's frame
count is configurable via ``NUM_FRAMES`` (default 16, what the public checkpoint
was trained with).
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("videoprism_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


MODEL_ID = "videoprism_public_v1_base_hf"
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_REVISION = "d481d91b9bf8c9d330d1e526e511a359c799bbe1"
UPSTREAM_MODEL_ID = "videoprism_public_v1_base"
CHECKPOINT_REPO_ID = "google/videoprism-base-f16r288"
CHECKPOINT_FILENAME = "flax_base_f16r288_repeated.npz"

if configured_model := os.environ.get("MODEL_NAME"):
    if configured_model != MODEL_ID:
        raise RuntimeError(f"MODEL_NAME must equal pinned model {MODEL_ID}")


class VideoEmbeddingRequest(BaseModel):
    video: str = Field(..., description="Base64-encoded MP4 video segment.")
    start_time: float = Field(default=0.0, description="Segment start (seconds).")
    end_time: float = Field(default=0.0, description="Segment end (seconds).")
    model: str | None = Field(
        default=None,
        description="Optional model identifier; the server is pinned to one model.",
    )

    @field_validator("model")
    @classmethod
    def require_pinned_model(cls, model: str | None) -> str | None:
        if model is not None and model != MODEL_ID:
            raise ValueError(f"model must equal {MODEL_ID}")
        return model


class VideoEmbeddingResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        ..., description="Per-patch multi-vector embeddings, [num_patches, dim]."
    )
    processing_time: float
    model: str
    frames_processed: int


_MODEL: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_videoprism(model_name: str) -> dict[str, Any]:
    """Build the pinned JAX model and load its pretrained weights."""
    if model_name != MODEL_ID:
        raise ValueError(f"model must equal pinned model {MODEL_ID}")

    import jax
    from huggingface_hub import hf_hub_download
    from videoprism import models as vp

    logger.info(
        "Loading VideoPrism model: %s (upstream model: %s)",
        model_name,
        UPSTREAM_MODEL_ID,
    )
    model = vp.get_model(UPSTREAM_MODEL_ID)
    checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO_ID,
        filename=CHECKPOINT_FILENAME,
        revision=MODEL_REVISION,
    )
    state = vp.load_pretrained_weights(
        UPSTREAM_MODEL_ID,
        checkpoint_path=checkpoint_path,
    )

    def _forward(frames):
        embeddings, _ = model.apply(state, frames, train=False)
        return embeddings

    loaded = {
        "name": model_name,
        "model": model,
        "state": state,
        "forward": jax.jit(_forward),
        "embedding_dim": 768,
        "num_patches": 4096,
    }
    logger.info("VideoPrism ready (dim=768, patches=4096)")
    return loaded


def _get_videoprism(model_name: str) -> dict[str, Any]:
    if model_name != MODEL_ID:
        raise ValueError(f"model must equal pinned model {MODEL_ID}")
    if _MODEL:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL:
            return _MODEL
        _MODEL.update(_load_videoprism(model_name))
        return _MODEL


def _sample_frames(video_path: str, num_frames: int):
    """Decode ``video_path`` and sample ``num_frames`` evenly across its
    duration. Returns float32 [num_frames, 288, 288, 3] normalized to
    [0, 1]. Mirrors ``VideoPrismModel.preprocess_video``."""
    import cv2
    import numpy as np

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


app = FastAPI(title="VideoPrism inference")


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        loaded = _get_videoprism(MODEL_ID)
    except Exception as exc:
        logger.exception("model load failed for health probe")
        raise HTTPException(
            status_code=503,
            detail=(
                f"videoprism_jax: model {MODEL_ID} load failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc
    return {
        "status": "ready",
        "model": loaded["name"],
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "embedding_dim": loaded["embedding_dim"],
        "num_patches": loaded["num_patches"],
    }


@app.post("/v1/video/embeddings", response_model=VideoEmbeddingResponse)
def embeddings(req: VideoEmbeddingRequest) -> VideoEmbeddingResponse:
    import numpy as np

    started = time.perf_counter()
    model_name = req.model or MODEL_ID

    try:
        loaded = _get_videoprism(model_name)
    except Exception as exc:
        logger.exception("model load failed for %s", model_name)
        raise HTTPException(
            status_code=503,
            detail=(
                f"videoprism_jax: model {model_name} load failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc

    try:
        video_bytes = base64.b64decode(req.video, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=400,
            detail="videoprism_jax: video is not valid base64",
        ) from exc

    num_frames = int(os.environ.get("NUM_FRAMES", "16"))

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            frames = _sample_frames(tmp_path, num_frames)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                raise RuntimeError(
                    f"failed to remove temporary video {tmp_path} "
                    f"({type(exc).__name__}): {exc}"
                ) from exc

        batch = np.expand_dims(frames, axis=0)
        out = loaded["forward"](batch)
        embeddings_np = np.asarray(out[0])
        expected_shape = (loaded["num_patches"], loaded["embedding_dim"])
        if embeddings_np.shape != expected_shape:
            raise ValueError(
                f"expected embedding shape {expected_shape}, got {embeddings_np.shape}"
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("video inference failed (model=%s)", model_name)
        raise HTTPException(
            status_code=500,
            detail=(
                f"videoprism_jax: model {model_name} inference failed "
                f"({type(exc).__name__}): {exc}"
            ),
        ) from exc

    return VideoEmbeddingResponse(
        embeddings=embeddings_np.tolist(),
        processing_time=time.perf_counter() - started,
        model=loaded["name"],
        frames_processed=num_frames,
    )
