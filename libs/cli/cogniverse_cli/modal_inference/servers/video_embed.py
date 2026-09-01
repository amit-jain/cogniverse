"""FastAPI sidecar serving temporal video embeddings.

Two embedding endpoints, one joint video-text space (the model is trained
contrastively, so text vectors are directly comparable to video ones):

* ``POST /embed/video`` — body ``{"video_b64": "..."}`` (an MP4 segment's raw
  bytes). Frames are sampled evenly across the segment and encoded with
  cross-frame attention, so the vector describes motion across the clip rather
  than a single frame. Returns ``{"vec": [768 floats]}``.
* ``POST /embed/text`` — body ``{"text": "..."}``. Returns the matching
  768-dim text vector in the same space, which is what makes text→video
  search possible.
* ``GET /health`` — readiness probe that loads the pinned model.

The model is named only in configuration; nothing here depends on which
checkpoint is served beyond its embedding width, which is asserted on every
response.

One model, one process. Model dependencies remain lazy imports so Modal
operators can load this module without importing model runtimes locally.
"""

import base64
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from importlib import import_module
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("video_embed_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


class VideoEmbedRequest(BaseModel):
    video_b64: str = Field(
        ...,
        description="Base64-encoded video segment bytes (MP4 or anything ffmpeg reads).",
    )


class TextEmbedRequest(BaseModel):
    text: str = Field(..., description="Query text to embed into the video space.")


class EmbedResponse(BaseModel):
    vec: List[float] = Field(
        ..., description="Pooled embedding in the joint video-text space."
    )


@dataclass(frozen=True)
class VideoEmbedConfig:
    model_name: str = "microsoft/xclip-large-patch14"
    model_revision: str = "a9dd1429a16cf305df2aaea232d5e8dceba1c675"
    # The checkpoint's own vision_config.num_frames; the processor stacks
    # exactly this many, so sampling must produce exactly this many.
    num_frames: int = 8
    embedding_dim: int = 768
    device: str = "cpu"
    host: str = "0.0.0.0"
    port: int = 8080


_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = threading.Lock()


def _load_model(cfg: VideoEmbedConfig):
    """Load the video encoder lazily on first request.

    Readiness and inference share this path, so a pod cannot join its Service
    until the pinned processor and model are usable. The lock serialises
    concurrent first requests so only one downloads/deserialises the
    checkpoint.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _PROCESSOR
        transformers = import_module("transformers")

        logger.info(
            "Loading video embedder model=%s revision=%s device=%s (cold start)",
            cfg.model_name,
            cfg.model_revision,
            cfg.device,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            cfg.model_name,
            revision=cfg.model_revision,
        )
        model = transformers.AutoModel.from_pretrained(
            cfg.model_name,
            revision=cfg.model_revision,
        )
        model.to(cfg.device)
        model.eval()
        _PROCESSOR = processor
        _MODEL = model
        logger.info("video embedder ready")
        return model, processor


def _coerce_dim(vec, expected_dim: int) -> List[float]:
    """Squeeze to a flat ``expected_dim`` list.

    A width other than the configured one means the served checkpoint does not
    match the schema the vectors are fed into, which Vespa would reject far
    from here; fail at the boundary that knows why.
    """
    np = import_module("numpy")

    arr = np.asarray(vec, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != expected_dim:
        raise HTTPException(
            status_code=500,
            detail=(
                f"video_embed: model emitted {arr.shape[0]} dims, expected "
                f"{expected_dim}; the served checkpoint does not match the "
                "configured embedding width"
            ),
        )
    return [float(value) for value in arr]


def _sample_frames(video_path: str, num_frames: int):
    """Sample ``num_frames`` frames evenly across the clip, as RGB uint8.

    Returned frames are unresized: the processor applies the checkpoint's own
    resize and normalisation, so doing it here would risk a second, different
    transform.
    """
    cv2 = import_module("cv2")
    np = import_module("numpy")

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise HTTPException(
            status_code=400, detail=f"could not read frames from video at {video_path}"
        )

    indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        read_ok, frame = capture.read()
        if not read_ok or frame is None:
            # Repeat the previous frame so the stack is always exactly
            # num_frames; the model's input shape is fixed.
            if not frames:
                capture.release()
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode any frame from video at {video_path}",
                )
            frames.append(frames[-1].copy())
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def _decode_video(video_b64: str, num_frames: int):
    try:
        raw = base64.b64decode(video_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"video_b64 decode failed: {exc}"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        handle.write(raw)
        handle.flush()
        return _sample_frames(handle.name, num_frames)


def _inputs_to_device(inputs, device: str):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }


def _model_error(
    cfg: VideoEmbedConfig,
    operation: str,
    status_code: int,
    exc: Exception,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=(
            f"video_embed: model {cfg.model_name} {operation} failed "
            f"({type(exc).__name__}): {exc}"
        ),
    )


def build_app(cfg: VideoEmbedConfig) -> FastAPI:
    app = FastAPI(title="cogniverse video-embed sidecar", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        try:
            _load_model(cfg)
        except Exception as exc:
            raise _model_error(cfg, "load", 503, exc) from exc
        return {
            "status": "ready",
            "model": cfg.model_name,
            "model_revision": cfg.model_revision,
            "embedding_dim": cfg.embedding_dim,
            "num_frames": cfg.num_frames,
        }

    @app.post("/embed/video", response_model=EmbedResponse)
    def embed_video(req: VideoEmbedRequest) -> EmbedResponse:
        frames = _decode_video(req.video_b64, cfg.num_frames)
        try:
            model, processor = _load_model(cfg)
        except Exception as exc:
            raise _model_error(cfg, "load", 503, exc) from exc

        torch = import_module("torch")

        try:
            inputs = _inputs_to_device(
                processor(videos=[frames], return_tensors="pt"),
                cfg.device,
            )
            with torch.no_grad():
                video_embeds = model.get_video_features(**inputs)
        except HTTPException:
            raise
        except Exception as exc:
            raise _model_error(cfg, "inference", 500, exc) from exc
        return EmbedResponse(
            vec=_coerce_dim(video_embeds.squeeze().cpu().numpy(), cfg.embedding_dim)
        )

    @app.post("/embed/text", response_model=EmbedResponse)
    def embed_text(req: TextEmbedRequest) -> EmbedResponse:
        try:
            model, processor = _load_model(cfg)
        except Exception as exc:
            raise _model_error(cfg, "load", 503, exc) from exc

        torch = import_module("torch")

        try:
            inputs = _inputs_to_device(
                processor(text=[req.text], return_tensors="pt", padding=True),
                cfg.device,
            )
            with torch.no_grad():
                text_embeds = model.get_text_features(**inputs)
        except HTTPException:
            raise
        except Exception as exc:
            raise _model_error(cfg, "inference", 500, exc) from exc
        return EmbedResponse(
            vec=_coerce_dim(text_embeds.squeeze().cpu().numpy(), cfg.embedding_dim)
        )

    return app


# Default-config app for in-process consumers (tests import ``app`` and
# patch ``_MODEL``/``_PROCESSOR``). The deployed entrypoint is ``main()``,
# which parses the container env once and builds its own app from it.
app = build_app(VideoEmbedConfig())


def main() -> None:
    """Deployed entrypoint. The container contract (Dockerfile ENV + Helm
    values) configures the sidecar via environment — parsed here, once, and
    nowhere else. Defaults are single-sourced from the dataclass."""
    uvicorn = import_module("uvicorn")

    defaults = VideoEmbedConfig()
    cfg = VideoEmbedConfig(
        model_name=os.environ.get("VIDEO_EMBED_MODEL", defaults.model_name),
        model_revision=os.environ.get(
            "VIDEO_EMBED_MODEL_REVISION",
            defaults.model_revision,
        ),
        num_frames=int(
            os.environ.get("VIDEO_EMBED_NUM_FRAMES", str(defaults.num_frames))
        ),
        embedding_dim=int(
            os.environ.get("VIDEO_EMBED_DIM", str(defaults.embedding_dim))
        ),
        device=os.environ.get("VIDEO_EMBED_DEVICE", defaults.device),
        host=os.environ.get("HOST", defaults.host),
        port=int(os.environ.get("PORT", str(defaults.port))),
    )
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
