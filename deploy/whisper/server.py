"""FastAPI wrapper exposing a Whisper ASR model behind ``POST /v1/transcribe``.

Each Whisper backend ships in its own image so engine selection happens
at chart-template render time, not at runtime via env var. This file is
the **faster-whisper variant** (image: ``cogniverse/whisper-fw``,
``deploy/whisper/Dockerfile``) — it loads CTranslate2 directly and
doesn't try to know about whisperx or whisper.cpp. Sibling images get
their own ``server.py`` specialisations:

  - ``cogniverse/whisper-wx`` (``Dockerfile.wx``) — PyTorch-based whisperx,
    required on AMD ROCm where CTranslate2 has no backend.
  - ``cogniverse/whisper-cpp`` (``Dockerfile.cpp``) — whisper.cpp C++
    binding, lightweight for Apple Silicon / constrained environments.

The chart's ``whisper.engine`` field picks which image to pull (see
``charts/cogniverse/templates/all-resources.yaml``). A pod runs exactly
one engine; ``GET /health`` returns the engine identifier so the runtime's
inference health check can verify nobody mismatched image and config.

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

ENGINE_NAME = "faster-whisper"


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
# Engine
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
    # default; float16 on CUDA. ROCm has no CTranslate2 backend — deploy the
    # whisperx image (Dockerfile.wx) instead.
    if device == "cpu":
        compute_type = "int8"
    elif device == "cuda":
        compute_type = "float16"
    else:
        raise NotImplementedError(
            f"faster-whisper does not support device={device!r}. "
            f"Use the whisperx image (cogniverse/whisper-wx) for ROCm, or "
            f"the whisper-cpp image (cogniverse/whisper-cpp) for Apple Silicon."
        )

    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        model_size,
        device,
        compute_type,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    class FasterWhisperEngine(_Engine):
        name = ENGINE_NAME
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(model_size: str, device: str) -> FastAPI:
    app = FastAPI(title="Whisper transcription", version="1.0")
    engine = _load_faster_whisper(model_size, device)
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

    model_size = os.environ.get("MODEL_NAME", "base")
    device = os.environ.get("DEVICE", "cpu")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7998"))

    app = build_app(model_size, device)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    _main()
