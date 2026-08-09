"""FastAPI sidecar serving CLAP acoustic embeddings.

Two embedding endpoints, one joint audio-text space (CLAP is trained
contrastively, so text vectors are directly comparable to audio ones):

* ``POST /embed/audio`` — body ``{"audio_b64": "..."}`` (an audio file's
  raw bytes, any libsndfile-readable container). Returns
  ``{"vec": [512 floats]}`` from ``ClapModel.get_audio_features``.
* ``POST /embed/text`` — body ``{"text": "..."}``. Returns the matching
  512-dim text vector from ``get_text_features`` — used by the
  audio-analysis agent to encode acoustic-mode queries.
* ``GET /health`` — readiness probe that loads the pinned model.

Preprocessing mirrors the in-process generator byte-for-byte: audio is
loaded via ``librosa.load(sr=48000, mono=True)`` before ClapProcessor.

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

logger = logging.getLogger("clap_embed_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


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


@dataclass(frozen=True)
class ClapEmbedConfig:
    model_name: str = "laion/clap-htsat-unfused"
    model_revision: str = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
    sample_rate: int = 48000
    device: str = "cpu"
    host: str = "0.0.0.0"
    port: int = 8080


_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = threading.Lock()


def _load_model(cfg: ClapEmbedConfig):
    """Load CLAP lazily on first request.

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
            "Loading CLAP model=%s revision=%s device=%s (cold start)",
            cfg.model_name,
            cfg.model_revision,
            cfg.device,
        )
        processor = transformers.ClapProcessor.from_pretrained(
            cfg.model_name,
            revision=cfg.model_revision,
        )
        model = transformers.ClapModel.from_pretrained(
            cfg.model_name,
            revision=cfg.model_revision,
        )
        model.to(cfg.device)
        model.eval()
        _PROCESSOR = processor
        _MODEL = model
        logger.info("CLAP ready")
        return model, processor


def _coerce_512(vec) -> List[float]:
    """Squeeze to a flat 512-dim list, padding/truncating defensively —
    the same guard the in-process generator applied."""
    np = import_module("numpy")

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

    librosa = import_module("librosa")

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


def _inputs_to_device(inputs, device: str):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }


def _model_error(
    cfg: ClapEmbedConfig,
    operation: str,
    status_code: int,
    exc: Exception,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=(
            f"clap_embed: model {cfg.model_name} {operation} failed "
            f"({type(exc).__name__}): {exc}"
        ),
    )


def build_app(cfg: ClapEmbedConfig) -> FastAPI:
    app = FastAPI(title="cogniverse clap-embed sidecar", version="1.0.0")

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

    @app.post("/embed/audio", response_model=EmbedResponse)
    def embed_audio(req: AudioEmbedRequest) -> EmbedResponse:
        audio_array = _decode_audio(req.audio_b64, cfg.sample_rate)
        try:
            model, processor = _load_model(cfg)
        except Exception as exc:
            raise _model_error(cfg, "load", 503, exc) from exc

        torch = import_module("torch")

        try:
            inputs = _inputs_to_device(
                processor(
                    audios=audio_array,
                    sampling_rate=cfg.sample_rate,
                    return_tensors="pt",
                ),
                cfg.device,
            )
            with torch.no_grad():
                audio_embeds = model.get_audio_features(**inputs)
            return EmbedResponse(vec=_coerce_512(audio_embeds.squeeze().cpu().numpy()))
        except Exception as exc:
            raise _model_error(cfg, "inference", 500, exc) from exc

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
            return EmbedResponse(vec=_coerce_512(text_embeds.squeeze().cpu().numpy()))
        except Exception as exc:
            raise _model_error(cfg, "inference", 500, exc) from exc

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
    uvicorn = import_module("uvicorn")

    defaults = ClapEmbedConfig()
    cfg = ClapEmbedConfig(
        model_name=os.environ.get("CLAP_EMBED_MODEL", defaults.model_name),
        model_revision=os.environ.get(
            "CLAP_EMBED_MODEL_REVISION",
            defaults.model_revision,
        ),
        sample_rate=int(
            os.environ.get("CLAP_EMBED_SAMPLE_RATE", str(defaults.sample_rate))
        ),
        device=os.environ.get("CLAP_EMBED_DEVICE", defaults.device),
        host=os.environ.get("HOST", defaults.host),
        port=int(os.environ.get("PORT", str(defaults.port))),
    )
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
