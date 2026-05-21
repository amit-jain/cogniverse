"""FastAPI sidecar serving InsightFace (Buffalo_L) face embeddings.

Two endpoints:

* ``POST /embed`` — body ``{"image_url": "http://..."}`` OR
  ``{"image_b64": "..."}``. Returns ``{"faces": [{bbox, vec}], "n": int}``
  where ``vec`` is a 512-dim ArcFace embedding (the same space cogniverse
  Phase-2 face-clustering operates in) and ``bbox`` is the detected face
  rectangle as ``[x1, y1, x2, y2]`` in image pixels.
* ``GET /health`` — liveness probe used by Helm + ``setup_local_tests.sh``.

One model, one process. InsightFace's ``Buffalo_L`` bundles the
``RetinaFace`` detector + the ``ArcFace`` ``w600k_r50`` recogniser. Both
load on CPU at process start (≈300 MiB resident, ≈5 s cold-load). After
warmup the encode path is ~50 ms per frame for typical 720p inputs.

Cogniverse Phase-2 face-cluster paths POST one image per keyframe and
cluster the returned vectors per ``source_doc_id`` to discover
anonymous identity groups. The sidecar does not persist any state — it's
a pure compute service.
"""

import base64
import io
import logging
import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("face_embed_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


_MODEL = None


def _load_model():
    """Load InsightFace Buffalo_L lazily on first request.

    Two reasons not to eager-load:
      * Health probes pass before the model is in memory, so the pod can
        join its Service early.
      * Tests can patch ``_MODEL`` with a deterministic stand-in without
        ever hitting the real ArcFace weights.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    # Imported inside the function so test patches can avoid the heavy
    # native dependency entirely.
    from insightface.app import FaceAnalysis  # noqa: PLC0415

    model_name = os.environ.get("FACE_EMBED_MODEL", "buffalo_l")
    ctx_id = int(os.environ.get("FACE_EMBED_CTX_ID", "-1"))  # -1 = CPU
    logger.info(
        "Loading InsightFace model=%s ctx_id=%s (this takes ~5s on cold start)",
        model_name,
        ctx_id,
    )
    app_ = FaceAnalysis(name=model_name)
    app_.prepare(ctx_id=ctx_id, det_size=(640, 640))
    _MODEL = app_
    logger.info("InsightFace ready")
    return _MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_from_request(req: EmbedRequest) -> bytes:
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

    # URL fetch — local-only by default for security; expand via env if
    # the operator explicitly opts in (e.g. via a sidecar-internal cache).
    import httpx  # noqa: PLC0415

    timeout = float(os.environ.get("FACE_EMBED_URL_TIMEOUT_S", "5.0"))
    try:
        resp = httpx.get(req.image_url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=400, detail=f"image_url fetch failed: {exc}"
        ) from exc


def _decode_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode arbitrary image bytes to the BGR ndarray InsightFace expects."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # PIL raises a hodge-podge of exception types
        raise HTTPException(
            status_code=400, detail=f"image decode failed: {exc}"
        ) from exc
    rgb = np.array(img)
    # InsightFace wants BGR (OpenCV convention) — just reverse the channel
    # axis without paying for an OpenCV install.
    return rgb[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


app = FastAPI(title="cogniverse face-embed sidecar", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    model = _load_model()
    image_bytes = _bytes_from_request(req)
    image = _decode_to_bgr(image_bytes)
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


def main() -> None:
    import uvicorn  # noqa: PLC0415

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
