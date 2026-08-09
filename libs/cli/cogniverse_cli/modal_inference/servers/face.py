"""FastAPI service serving InsightFace (Buffalo_L) face embeddings.

Two endpoints:

* ``POST /embed`` — body ``{"image_url": "http://..."}`` OR
  ``{"image_b64": "..."}``. Returns ``{"faces": [{bbox, vec}], "n": int}``
  where ``vec`` is a 512-dim L2-normalised ArcFace embedding (the same
  space the face-cluster consumer operates in) and ``bbox`` is the
  detected face rectangle as ``[x1, y1, x2, y2]`` in image pixels.
* ``GET /health`` — readiness for the pinned model artifact.

One model, one process. InsightFace's ``Buffalo_L`` bundles the
``RetinaFace`` detector + the ``ArcFace`` ``w600k_r50`` recogniser. The
verified model pack is present in the image before the process starts;
first-request initialization only opens those local ONNX files.

The face-cluster consumer POSTs one image per keyframe and clusters
the returned vectors per ``source_doc_id`` to discover anonymous
identity groups. The sidecar does not persist any state — it's a
pure compute service.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("face_embed_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

FACE_MODEL_NAME = "buffalo_l"
FACE_MODEL_REVISION = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
FACE_MODEL_ROOT = "/opt/insightface"
FACE_MODEL_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


class EmbedRequest(BaseModel):
    image_url: Optional[str] = Field(
        default=None,
        description="HTTP(S) URL to fetch the image from.",
    )
    image_b64: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded image bytes (PNG/JPEG). Mutually exclusive "
            "with image_url; exactly one must be supplied."
        ),
    )


class FaceRecord(BaseModel):
    bbox: List[int] = Field(
        ...,
        description="Detected face rectangle: [x1, y1, x2, y2] in image pixels.",
    )
    vec: List[float] = Field(
        ...,
        description=(
            "Normalized 512-dim ArcFace embedding. Cosine similarity in "
            "this space is well-calibrated for identity grouping."
        ),
    )
    det_score: float = Field(
        ...,
        description="RetinaFace detection confidence in [0, 1].",
    )


class EmbedResponse(BaseModel):
    n: int = Field(..., description="Number of faces detected in the image.")
    faces: List[FaceRecord]


@dataclass(frozen=True)
class FaceEmbedConfig:
    model_name: str = FACE_MODEL_NAME
    model_revision: str = FACE_MODEL_REVISION
    model_root: str = FACE_MODEL_ROOT
    ctx_id: int = -1  # -1 = CPU; a GPU index for CUDA
    url_timeout_s: float = 5.0
    host: str = "0.0.0.0"
    port: int = 8080


_MODEL = None
_MODEL_LOCK = threading.Lock()


def _require_model_artifact(cfg: FaceEmbedConfig) -> None:
    model_dir = Path(cfg.model_root) / "models" / cfg.model_name
    missing = [
        str(model_dir / filename)
        for filename in FACE_MODEL_FILES
        if not (model_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "face model artifact is incomplete; missing: " + ", ".join(missing)
        )


def _load_model(cfg: FaceEmbedConfig):
    """Load InsightFace lazily on first request.

    Readiness and inference share this path, so the pod cannot join its Service
    until the pinned local ONNX artifacts open successfully. The lock
    serialises concurrent first requests; initialization never downloads
    artifacts.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        _require_model_artifact(cfg)
        face_analysis = import_module("insightface.app")

        logger.info(
            "Loading InsightFace model=%s ctx_id=%s (this takes ~5s on cold start)",
            cfg.model_name,
            cfg.ctx_id,
        )
        if cfg.ctx_id >= 0:
            app_ = face_analysis.FaceAnalysis(
                name=cfg.model_name,
                root=cfg.model_root,
                providers=["CUDAExecutionProvider"],
            )
        else:
            app_ = face_analysis.FaceAnalysis(
                name=cfg.model_name,
                root=cfg.model_root,
            )
        app_.prepare(ctx_id=cfg.ctx_id, det_size=(640, 640))
        _MODEL = app_
        logger.info("InsightFace ready")
        return app_


def _bytes_from_request(req: EmbedRequest, url_timeout_s: float) -> bytes:
    """Resolve the EmbedRequest to raw image bytes — URL fetch or b64 decode."""
    if (req.image_url is None) == (req.image_b64 is None):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of image_url, image_b64 must be supplied.",
        )
    if req.image_b64 is not None:
        try:
            return base64.b64decode(req.image_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail=f"image_b64 decode failed: {exc}"
            ) from exc

    httpx = import_module("httpx")

    try:
        resp = httpx.get(req.image_url, timeout=url_timeout_s)
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=400, detail=f"image_url fetch failed: {exc}"
        ) from exc


def _decode_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode arbitrary image bytes to the BGR ndarray InsightFace expects."""
    np = import_module("numpy")
    image_module = import_module("PIL.Image")

    try:
        img = image_module.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # PIL raises a hodge-podge of exception types
        raise HTTPException(
            status_code=400, detail=f"image decode failed: {exc}"
        ) from exc
    rgb = np.array(img)
    # InsightFace wants BGR (OpenCV convention) — just reverse the channel
    # axis without paying for an OpenCV install.
    return rgb[:, :, ::-1].copy()


def _model_error(
    cfg: FaceEmbedConfig,
    operation: str,
    status_code: int,
    exc: Exception,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=(
            f"face_embed: model {cfg.model_name} {operation} failed "
            f"({type(exc).__name__}): {exc}"
        ),
    )


def build_app(cfg: FaceEmbedConfig) -> FastAPI:
    app = FastAPI(title="cogniverse face-embed sidecar", version="1.0.0")

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
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        image_bytes = _bytes_from_request(req, cfg.url_timeout_s)
        image = _decode_to_bgr(image_bytes)
        try:
            model = _load_model(cfg)
        except Exception as exc:
            raise _model_error(cfg, "load", 503, exc) from exc

        try:
            raw_faces = model.get(image)
            faces: List[FaceRecord] = []
            for f in raw_faces:
                # f.normed_embedding is L2-normalised already, which is what we
                # want for cosine clustering on the consumer side.
                faces.append(
                    FaceRecord(
                        bbox=[int(c) for c in f.bbox.astype(int).tolist()],
                        vec=[float(v) for v in f.normed_embedding.tolist()],
                        det_score=float(f.det_score),
                    )
                )
            return EmbedResponse(n=len(faces), faces=faces)
        except Exception as exc:
            raise _model_error(cfg, "inference", 500, exc) from exc

    return app


# Default-config app for in-process consumers (tests import ``app`` and
# patch ``_MODEL``). The deployed entrypoint is ``main()``, which parses
# the container env once and builds its own app from it.
app = build_app(FaceEmbedConfig())


def main() -> None:
    """Deployed entrypoint. The container contract (Dockerfile ENV +
    Helm values) configures the sidecar via environment — parsed here,
    once, and nowhere else. Defaults are single-sourced from the
    dataclass."""
    uvicorn = import_module("uvicorn")

    defaults = FaceEmbedConfig()
    cfg = FaceEmbedConfig(
        model_name=os.environ.get("FACE_EMBED_MODEL", defaults.model_name),
        model_revision=os.environ.get(
            "FACE_EMBED_MODEL_REVISION", defaults.model_revision
        ),
        model_root=os.environ.get("FACE_EMBED_MODEL_ROOT", defaults.model_root),
        ctx_id=int(os.environ.get("FACE_EMBED_CTX_ID", str(defaults.ctx_id))),
        url_timeout_s=float(
            os.environ.get("FACE_EMBED_URL_TIMEOUT_S", str(defaults.url_timeout_s))
        ),
        host=os.environ.get("HOST", defaults.host),
        port=int(os.environ.get("PORT", str(defaults.port))),
    )
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
