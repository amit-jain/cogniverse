"""FastAPI wrapper exposing a Whisper ASR model behind ``POST /v1/transcribe``.

Pluggable engine selected at startup via the ``WHISPER_ENGINE`` env var:

  - ``faster-whisper`` (default): CTranslate2 backend, runs well on CPU
    and CUDA. Image: ``cogniverse/whisper-fw``.
  - ``whisperx``: PyTorch-based, used on AMD ROCm where CTranslate2 has
    no backend. Image: ``cogniverse/whisper-wx``. (Stub raises until the
    image and engine wiring are built.)
  - ``whisper-cpp``: whisper.cpp C++ binding, used on Apple Silicon /
    constrained environments. Image: ``cogniverse/whisper-cpp``. (Stub.)

Mirrors the ``deploy/colpali`` shape so the chart treats both the same:
one pod, one model, env-var configuration, ``GET /health`` returns the
model identifier so the runtime's inference health check can verify it.

Request shape (canonical):

    POST /v1/transcribe
    {
      "audio_b64": "<base64-encoded audio bytes>",
      "language": "en"        # optional, auto-detect when absent
    }

Future: accept ``source_url`` to fetch from MinIO directly so workers
don't have to base64-encode.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("whisper_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class TranscribeRequest(BaseModel):
    audio_b64: str = Field(
        ...,
        description="Base64-encoded audio bytes (any format ffmpeg can read).",
    )
    language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code, e.g. 'en'. Omit for auto-detect.",
    )
    initial_prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt to bias the decoder; helpful for domain terms.",
    )


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration_seconds: float
    processing_time: float
    model: str
    segments: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine loaders
# ---------------------------------------------------------------------------


class _Engine:
    """Minimal interface every Whisper backend must satisfy."""

    name: str
    model_id: str

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: Optional[str],
        initial_prompt: Optional[str],
    ) -> dict[str, Any]:
        """Return a dict with text, language, duration, segments."""
        raise NotImplementedError


def _load_faster_whisper(model_size: str, device: str) -> _Engine:
    """Load the faster-whisper backend (CTranslate2)."""
    from faster_whisper import WhisperModel

    # CTranslate2 compute_type: int8 on CPU is the standard small-footprint
    # default; float16 on CUDA. ROCm isn't supported by CTranslate2 — that
    # path raises NotImplementedError below and the deployer must use the
    # whisperx engine.
    if device == "cpu":
        compute_type = "int8"
    elif device == "cuda":
        compute_type = "float16"
    else:
        raise NotImplementedError(
            f"faster-whisper engine does not support device={device!r}. "
            f"Use WHISPER_ENGINE=whisperx for ROCm, or "
            f"WHISPER_ENGINE=whisper-cpp for Apple Silicon."
        )

    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        model_size,
        device,
        compute_type,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    class FasterWhisperEngine(_Engine):
        name = "faster-whisper"
        model_id = model_size

        def transcribe(
            self,
            audio_bytes: bytes,
            *,
            language: Optional[str],
            initial_prompt: Optional[str],
        ) -> dict[str, Any]:
            # WhisperModel.transcribe accepts a file-like or path; BytesIO
            # avoids touching disk for short audio. ffmpeg-style demuxing
            # is handled internally.
            buf = io.BytesIO(audio_bytes)
            segments_iter, info = model.transcribe(
                buf,
                language=language,
                initial_prompt=initial_prompt,
                # vad_filter trims silence; helps cut spurious tokens on
                # podcasts/recordings with long pauses.
                vad_filter=True,
            )
            segments = []
            text_parts = []
            for seg in segments_iter:
                segments.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                )
                text_parts.append(seg.text)
            return {
                "text": "".join(text_parts).strip(),
                "language": info.language,
                "duration": info.duration,
                "segments": segments,
            }

    return FasterWhisperEngine()


def _load_whisperx(model_size: str, device: str) -> _Engine:
    """Load the whisperx backend (PyTorch). Required on ROCm.

    Stub for now — image cogniverse/whisper-wx isn't built yet. When you
    need ROCm, build the image with PyTorch-ROCm + whisperx and replace
    this stub. The interface above (`_Engine`) is what the rest of the
    server depends on.
    """
    raise NotImplementedError(
        "whisperx engine is a stub. Build the cogniverse/whisper-wx image "
        "(PyTorch-ROCm + whisperx) and implement this loader. See "
        "deploy/whisper/Dockerfile.wx for the build recipe scaffold."
    )


def _load_whisper_cpp(model_size: str, device: str) -> _Engine:
    """Load the whisper.cpp backend. Lightweight; useful on Apple Silicon.

    Stub for now — image cogniverse/whisper-cpp isn't built yet.
    """
    raise NotImplementedError(
        "whisper-cpp engine is a stub. Build the cogniverse/whisper-cpp image "
        "(whisper.cpp + Python binding) and implement this loader. See "
        "deploy/whisper/Dockerfile.cpp for the build recipe scaffold."
    )


_ENGINE_LOADERS = {
    "faster-whisper": _load_faster_whisper,
    "whisperx": _load_whisperx,
    "whisper-cpp": _load_whisper_cpp,
}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(engine_name: str, model_size: str, device: str) -> FastAPI:
    loader = _ENGINE_LOADERS.get(engine_name)
    if loader is None:
        raise ValueError(
            f"Unknown WHISPER_ENGINE={engine_name!r}. "
            f"Valid choices: {sorted(_ENGINE_LOADERS)}"
        )

    app = FastAPI(title="Whisper transcription", version="1.0")
    engine = loader(model_size, device)
    app.state.engine = engine

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": engine.model_id,
            "engine": engine.name,
        }

    @app.post("/v1/transcribe", response_model=TranscribeResponse)
    def transcribe(req: TranscribeRequest) -> TranscribeResponse:
        try:
            audio = base64.b64decode(req.audio_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid base64: {exc}"
            ) from exc

        start = time.perf_counter()
        result = engine.transcribe(
            audio,
            language=req.language,
            initial_prompt=req.initial_prompt,
        )
        elapsed = time.perf_counter() - start

        return TranscribeResponse(
            text=result["text"],
            language=result["language"],
            duration_seconds=float(result.get("duration") or 0.0),
            processing_time=round(elapsed, 4),
            model=engine.model_id,
            segments=result.get("segments", []),
        )

    return app


def _main() -> None:
    import uvicorn

    engine_name = os.environ.get("WHISPER_ENGINE", "faster-whisper")
    model_size = os.environ.get("MODEL_NAME", "base")
    device = os.environ.get("DEVICE", "cpu")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7998"))

    app = build_app(engine_name, model_size, device)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    _main()
