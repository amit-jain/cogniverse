#!/usr/bin/env python3
"""
Audio Processor - Pluggable audio transcription.

Transcribes audio from videos using Whisper.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from ..processor_base import BaseProcessor


class AudioProcessor(BaseProcessor):
    """Handles audio transcription from videos."""

    PROCESSOR_NAME = "audio"

    def __init__(
        self,
        logger: logging.Logger,
        model: str = "whisper-large-v3",
        language: str = "auto",
    ):
        """
        Initialize audio processor.

        Args:
            logger: Logger instance
            model: Whisper model to use
            language: Language for transcription (auto for detection)
        """
        super().__init__(logger)
        self.model = model
        self.language = language
        self._whisper = None

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], logger: logging.Logger
    ) -> "AudioProcessor":
        """Create audio processor from configuration."""
        return cls(
            logger=logger,
            model=config.get("model", "whisper-large-v3"),
            language=config.get("language", "auto"),
        )

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper is None:
            try:
                import whisper

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
                whisper_model_name = model_map.get(self.model, "large-v3")
                self._whisper = whisper.load_model(whisper_model_name)
                self.logger.info(f"   ✅ Whisper model loaded: {whisper_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                raise

    def transcribe_audio(
        self, video_path: Path, output_dir: Path = None, cache=None
    ) -> Dict[str, Any]:
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
            from src.common.utils.output_manager import get_output_manager

            output_manager = get_output_manager()
            transcript_file = (
                output_manager.get_processing_dir("transcripts")
                / f"{video_id}_transcript.json"
            )
        else:
            # Legacy path support
            transcript_file = output_dir / "transcripts" / f"{video_id}_transcript.json"

        transcript_file.parent.mkdir(parents=True, exist_ok=True)

        # Load Whisper model
        self._load_whisper()

        start_time = time.time()

        try:
            # Transcribe using Whisper
            options = {
                "language": None if self.language == "auto" else self.language,
                "task": "transcribe",
            }

            result = self._whisper.transcribe(str(video_path), **options)

            transcription_time = time.time() - start_time

            # Extract segments
            segments = []
            for segment in result.get("segments", []):
                segments.append(
                    {
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "text": segment.get("text", "").strip(),
                    }
                )

            # Create transcript data
            transcript_data = {
                "video_id": video_id,
                "video_path": str(video_path),
                "model": self.model,
                "language": result.get("language", "unknown"),
                "duration": result.get("duration", 0.0),
                "transcription_time": transcription_time,
                "full_text": result.get("text", "").strip(),
                "segments": segments,
            }

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

    def cleanup(self):
        """Clean up Whisper model."""
        if self._whisper is not None:
            del self._whisper
            self._whisper = None
