"""FastAPI sidecar serving CLAP acoustic embeddings.

Two embedding endpoints, one joint audio-text space (CLAP is trained
contrastively, so text vectors are directly comparable to audio ones):

* ``POST /embed/audio`` — body ``{"audio_b64": "..."}`` (an audio file's
  raw bytes, any libsndfile-readable container). Returns
  ``{"vec": [512 floats]}`` from ``ClapModel.get_audio_features``.
* ``POST /embed/text`` — body ``{"text": "..."}``. Returns the matching
  512-dim text vector from ``get_text_features`` — used by the
  audio-analysis agent to encode acoustic-mode queries.
* ``GET /health`` — liveness probe.

Preprocessing mirrors the in-process generator byte-for-byte: audio is
loaded via ``librosa.load(sr=48000, mono=True)`` before ClapProcessor.

One model, one process. The module must stay free of cogniverse imports
so the image can COPY it alone (see deploy/clap_embed/Dockerfile; build
from the repo root).
"""

import base64
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("clap_embed_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class AudioEmbedRequest(BaseModel):
    audio_b64: str = Field(
        ...,
        description=(
            "Base64-encoded audio file bytes (WAV/FLAC/OGG — anything "
            "libsndfile reads)."
        ),
    )


class TextEmbedRequest(BaseModel):
    text: str = Field(..., description="Query text to embed into CLAP space.")


class EmbedResponse(BaseModel):
    vec: List[float] = Field(
        ..., description="512-dim CLAP embedding (joint audio-text space)."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClapEmbedConfig:
    model_name: str = "laion/clap-htsat-unfused"
    sample_rate: int = 48000
    host: str = "0.0.0.0"
    port: int = 8080


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = threading.Lock()


def _load_model(cfg: ClapEmbedConfig):
    """Load CLAP lazily on first request.

    Health probes pass before the model is in memory so the pod joins its
    Service early; tests patch ``_MODEL``/``_PROCESSOR`` with stand-ins.
    The lock serialises concurrent first requests so only one downloads/
    deserialises the checkpoint.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _PROCESSOR
        # Imported inside the function so test patches can avoid the heavy
        # native dependencies entirely.
        from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

        logger.info("Loading CLAP model=%s (cold start)", cfg.model_name)
        _PROCESSOR = ClapProcessor.from_pretrained(cfg.model_name)
        _MODEL = ClapModel.from_pretrained(cfg.model_name)
        _MODEL.eval()
        logger.info("CLAP ready")
        return _MODEL, _PROCESSOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_512(vec) -> List[float]:
    """Squeeze to a flat 512-dim list, padding/truncating defensively —
    the same guard the in-process generator applied."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(vec, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != 512:
        logger.warning("CLAP embedding has %s dims, expected 512", arr.shape[0])
        if arr.shape[0] > 512:
            arr = arr[:512]
        else:
            arr = np.concatenate([arr, np.zeros(512 - arr.shape[0], arr.dtype)])
    return [float(v) for v in arr]


def _decode_audio(audio_b64: str, sample_rate: int):
    """Decode request bytes to a mono float array at the target rate —
    mirrors ``librosa.load(path, sr=..., mono=True)`` exactly."""
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"audio_b64 decode failed: {exc}"
        ) from exc

    import librosa  # noqa: PLC0415

    suffix = ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix) as fh:
        fh.write(raw)
        fh.flush()
        try:
            audio_array, _sr = librosa.load(fh.name, sr=sample_rate, mono=True)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"audio decode failed: {exc}"
            ) from exc
    return audio_array


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def build_app(cfg: ClapEmbedConfig) -> FastAPI:
    app = FastAPI(title="cogniverse clap-embed sidecar", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/embed/audio", response_model=EmbedResponse)
    def embed_audio(req: AudioEmbedRequest) -> EmbedResponse:
        model, processor = _load_model(cfg)
        audio_array = _decode_audio(req.audio_b64, cfg.sample_rate)

        import torch  # noqa: PLC0415

        inputs = processor(
            audios=audio_array,
            sampling_rate=cfg.sample_rate,
            return_tensors="pt",
        )
        with torch.no_grad():
            audio_embeds = model.get_audio_features(**inputs)
        return EmbedResponse(vec=_coerce_512(audio_embeds.squeeze().cpu().numpy()))

    @app.post("/embed/text", response_model=EmbedResponse)
    def embed_text(req: TextEmbedRequest) -> EmbedResponse:
        model, processor = _load_model(cfg)

        import torch  # noqa: PLC0415

        inputs = processor(text=[req.text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
        return EmbedResponse(vec=_coerce_512(text_embeds.squeeze().cpu().numpy()))

    return app


# Default-config app for in-process consumers (tests import ``app`` and
# patch ``_MODEL``/``_PROCESSOR``). The deployed entrypoint is ``main()``,
# which parses the container env once and builds its own app from it.
app = build_app(ClapEmbedConfig())


def main() -> None:
    """Deployed entrypoint. The container contract (Dockerfile ENV +
    Helm values) configures the sidecar via environment — parsed here,
    once, and nowhere else. Defaults are single-sourced from the
    dataclass."""
    import uvicorn  # noqa: PLC0415

    defaults = ClapEmbedConfig()
    cfg = ClapEmbedConfig(
        model_name=os.environ.get("CLAP_EMBED_MODEL", defaults.model_name),
        sample_rate=int(
            os.environ.get("CLAP_EMBED_SAMPLE_RATE", str(defaults.sample_rate))
        ),
        host=os.environ.get("HOST", defaults.host),
        port=int(os.environ.get("PORT", str(defaults.port))),
    )
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
