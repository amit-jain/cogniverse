#!/usr/bin/env python3
"""
Audio Processor - Pluggable audio transcription.

Transcribes audio from videos using Whisper. Two modes:

- Remote (default in production): when ``endpoint`` is set, POSTs the
  audio multipart to the vLLM Whisper pod's OpenAI-compatible
  ``/v1/audio/transcriptions`` endpoint. The pod owns model selection
  via its ``--model`` arg.
- Local: loads ``openai-whisper`` in-process. Requires the
  ``cogniverse-runtime[whisper-local]`` extra; useful only for offline
  dev hosts without a cluster ASR sidecar.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..processor_base import BaseProcessor

REMOTE_TRANSCRIBE_TIMEOUT_SECONDS = 600.0


class AudioProcessor(BaseProcessor):
    """Handles audio transcription from videos."""

    PROCESSOR_NAME = "audio"

    def __init__(
        self,
        logger: logging.Logger,
        model: str = "base",
        language: str = "auto",
        endpoint: str | None = None,
    ):
        """
        Initialize audio processor.

        Args:
            logger: Logger instance
            model: Whisper model to use (local mode only; remote pod owns its
                model selection via the ``MODEL_NAME`` env var)
            language: Language for transcription (auto for detection)
            endpoint: When set, the processor runs in remote mode and POSTs
                audio multipart to ``{endpoint}/v1/audio/transcriptions``
                instead of loading Whisper locally.
        """
        super().__init__(logger)
        self.model = model
        self.language = language
        self.endpoint = endpoint
        self._whisper = None

    @classmethod
    def from_config(
        cls, config: dict[str, Any], logger: logging.Logger
    ) -> "AudioProcessor":
        """Create audio processor from configuration."""
        return cls(
            logger=logger,
            model=config.get("model", "base"),
            language=config.get("language", "auto"),
            endpoint=config.get("endpoint"),
        )

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper is None:
            try:
                import whisper  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "openai-whisper is not installed. Either install the "
                    "in-process audio extra (`pip install "
                    "cogniverse-runtime[whisper-local]`) or pass `endpoint=` "
                    "so AudioProcessor POSTs to the vLLM ASR sidecar instead."
                ) from exc

            try:
                self.logger.info(f"Loading Whisper model: {self.model}")
                # Map our model names to actual Whisper model names
                model_map = {
                    "whisper-large-v3": "large-v3",
                    "whisper-large-v2": "large-v2",
                    "whisper-medium": "medium",
                    "whisper-small": "small",
                    "whisper-base": "base",
                    "whisper-tiny": "tiny",
                }
                whisper_model_name = model_map.get(self.model, self.model)
                from cogniverse_core.common.models import model_load_lock

                with model_load_lock:
                    self._whisper = whisper.load_model(whisper_model_name)
                self.logger.info(f"   ✅ Whisper model loaded: {whisper_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                raise

    def transcribe_audio(
        self, video_path: Path, output_dir: Path = None, cache=None
    ) -> dict[str, Any]:
        """Transcribe audio from video."""
        self.logger.info(f"🎤 Transcribing audio from: {video_path.name}")

        video_id = video_path.stem

        # Check cache first if available
        if cache:
            self.logger.debug(f"Checking cache for transcript: {video_id}")
            cached_transcript = cache.get_transcript(video_path, video_id)
            if cached_transcript:
                self.logger.info(f"✅ Using cached transcript for {video_path.name}")
                return cached_transcript

        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from cogniverse_core.common.utils.output_manager import get_output_manager

            output_manager = get_output_manager()
            transcript_file = (
                output_manager.get_processing_dir("transcripts")
                / f"{video_id}_transcript.json"
            )
        else:
            # For testing - should migrate tests to use OutputManager
            transcript_file = output_dir / "transcripts" / f"{video_id}_transcript.json"

        transcript_file.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            if self.endpoint:
                transcript_data = self._transcribe_remote(video_path, video_id)
            else:
                self._load_whisper()
                transcript_data = self._transcribe_local(video_path, video_id)

            transcription_time = time.time() - start_time
            transcript_data["transcription_time"] = transcription_time
            segments = transcript_data.get("segments", [])

            # Save transcript to file
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)

            # Save to cache if available
            if cache:
                cache.set_transcript(video_path, video_id, transcript_data)
                self.logger.debug(f"Cached transcript for {video_id}")

            self.logger.info(
                f"   ✅ Audio transcribed in {transcription_time:.2f}s ({len(segments)} segments)"
            )

            return transcript_data

        except Exception as e:
            self.logger.error(f"   ❌ Audio transcription failed: {e}")
            return {
                "video_id": video_id,
                "error": str(e),
                "full_text": "",
                "segments": [],
            }

    def _transcribe_local(self, video_path: Path, video_id: str) -> dict[str, Any]:
        """Run transcription via the in-process openai-whisper model."""
        options = {
            "language": None if self.language == "auto" else self.language,
            "task": "transcribe",
        }
        result = self._whisper.transcribe(str(video_path), **options)

        segments = [
            {
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip(),
            }
            for segment in result.get("segments", [])
        ]

        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "model": self.model,
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0.0),
            "full_text": result.get("text", "").strip(),
            "segments": segments,
        }

    @staticmethod
    def _extract_audio_wav(video_path: Path) -> bytes:
        """Extract the audio stream from ``video_path`` and return it as
        16 kHz mono PCM WAV bytes.

        Required because vLLM's Whisper endpoint rejects MP4 / MKV
        containers (it expects pure audio in the OpenAI-Whisper-expected
        16 kHz mono format). pyav decodes the source container and
        resamples on the fly without a temp file.
        """
        import io

        import av

        container = av.open(str(video_path))
        in_stream = next((s for s in container.streams if s.type == "audio"), None)
        if in_stream is None:
            container.close()
            raise RuntimeError(f"{video_path}: no audio stream present")

        buf = io.BytesIO()
        out = av.open(buf, "w", format="wav")
        out_stream = out.add_stream("pcm_s16le", rate=16000)
        out_stream.layout = "mono"
        resampler = av.audio.resampler.AudioResampler(
            format=out_stream.format, layout=out_stream.layout, rate=16000
        )

        try:
            for frame in container.decode(in_stream):
                for resampled in resampler.resample(frame):
                    for packet in out_stream.encode(resampled):
                        out.mux(packet)
            for packet in out_stream.encode():
                out.mux(packet)
        finally:
            out.close()
            container.close()

        return buf.getvalue()

    def _transcribe_remote(self, video_path: Path, video_id: str) -> dict[str, Any]:
        """POST audio to the vLLM Whisper sidecar's
        ``/v1/audio/transcriptions`` endpoint and return the parsed
        transcript with per-segment timestamps.

        vLLM's Whisper endpoint rejects raw video containers ("Invalid
        or unsupported audio file") and requires 16 kHz mono PCM. We
        extract the audio stream via pyav, resample on the fly, and
        send the resulting wav buffer.
        """

        import requests

        url = f"{self.endpoint.rstrip('/')}/v1/audio/transcriptions"
        audio_bytes = self._extract_audio_wav(video_path)
        files = {"file": (f"{video_id}.wav", audio_bytes, "audio/wav")}
        try:
            models_resp = requests.get(
                f"{self.endpoint.rstrip('/')}/v1/models", timeout=10
            )
            models_resp.raise_for_status()
            served = (models_resp.json().get("data") or [{}])[0].get("id", self.model)
        except Exception:
            served = self.model
        data: dict[str, Any] = {
            "model": served,
            "response_format": "verbose_json",
        }
        if self.language and self.language != "auto":
            data["language"] = self.language

        self.logger.info(f"🛰️  POST {url}  ({len(audio_bytes) / 1024:.1f} KiB audio)")
        resp = requests.post(
            url, data=data, files=files, timeout=REMOTE_TRANSCRIBE_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        body = resp.json()

        full_text = (body.get("text") or "").strip()
        raw_segments = body.get("segments") or []
        segments = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
            }
            for seg in raw_segments
            if isinstance(seg, dict)
        ]
        # vLLM omits segments on short audio; synthesize one so callers
        # always have a non-empty list to iterate.
        if not segments and full_text:
            segments = [
                {
                    "start": 0.0,
                    "end": float(body.get("duration") or 0.0),
                    "text": full_text,
                }
            ]

        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "model": body.get("model", served),
            "language": body.get("language", "unknown"),
            "duration": float(body.get("duration") or 0.0),
            "full_text": full_text,
            "segments": segments,
        }

    def process(
        self, video_path: Path, output_dir: Path = None, **kwargs
    ) -> dict[str, Any]:
        """Process video by transcribing audio."""
        cache = kwargs.get("cache")
        return self.transcribe_audio(video_path, output_dir, cache)

    def cleanup(self):
        """Clean up Whisper model."""
        if self._whisper is not None:
            del self._whisper
            self._whisper = None
