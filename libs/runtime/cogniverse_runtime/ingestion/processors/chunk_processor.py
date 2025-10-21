#!/usr/bin/env python3
"""
Chunk Processor - Pluggable video chunk extraction.

Extracts video chunks for processing with models like ColQwen.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from ..processor_base import BaseProcessor


class ChunkProcessor(BaseProcessor):
    """Handles video chunk extraction."""

    PROCESSOR_NAME = "chunk"

    def __init__(
        self,
        logger: logging.Logger,
        chunk_duration: float = 30.0,
        chunk_overlap: float = 0.0,
        cache_chunks: bool = True,
    ):
        """
        Initialize chunk processor.

        Args:
            logger: Logger instance
            chunk_duration: Duration of each chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            cache_chunks: Whether to cache extracted chunks
        """
        super().__init__(logger)
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.cache_chunks = cache_chunks

    @classmethod
    def from_config(
        cls, config: dict[str, Any], logger: logging.Logger
    ) -> "ChunkProcessor":
        """Create chunk processor from configuration."""
        return cls(
            logger=logger,
            chunk_duration=config.get("chunk_duration", 30.0),
            chunk_overlap=config.get("chunk_overlap", 0.0),
            cache_chunks=config.get("cache_chunks", True),
        )

    def extract_chunks(
        self, video_path: Path, output_dir: Path = None
    ) -> dict[str, Any]:
        """Extract video chunks."""
        self.logger.info(
            f"🎬 Extracting chunks from: {video_path.name} ({self.chunk_duration}s chunks)"
        )

        video_id = video_path.stem

        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from cogniverse_core.common.utils.output_manager import get_output_manager

            output_manager = get_output_manager()
            chunks_dir = output_manager.get_processing_dir("chunks") / video_id
            metadata_file = (
                output_manager.get_processing_dir("metadata")
                / f"{video_id}_chunks.json"
            )
        else:
            # For testing - should migrate tests to use OutputManager
            chunks_dir = output_dir / "chunks" / video_id
            metadata_file = output_dir / "metadata" / f"{video_id}_chunks.json"

        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Get video duration
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            self.logger.error("   ❌ Could not determine video duration")
            return {"chunks": [], "metadata": {}}

        # Calculate chunk positions
        chunks = []
        chunk_idx = 0
        start_time = 0.0

        while start_time < duration:
            end_time = min(start_time + self.chunk_duration, duration)

            # Generate chunk
            chunk_filename = f"{video_id}_chunk_{chunk_idx:04d}.mp4"
            chunk_path = chunks_dir / chunk_filename

            if self._extract_chunk(
                video_path, chunk_path, start_time, end_time - start_time
            ):
                chunks.append(
                    {
                        "chunk_number": chunk_idx,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "filename": chunk_filename,
                        "path": str(chunk_path),
                    }
                )
                chunk_idx += 1

            # Move to next chunk (with overlap consideration)
            start_time += self.chunk_duration - self.chunk_overlap

        # Save metadata
        metadata = {
            "video_id": video_id,
            "video_path": str(video_path),
            "video_duration": duration,
            "chunk_duration": self.chunk_duration,
            "chunk_overlap": self.chunk_overlap,
            "chunks_extracted": len(chunks),
            "chunks": chunks,
        }

        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"   ✅ Extracted {len(chunks)} chunks")

        return {
            "chunks": chunks,
            "metadata": metadata,
            "chunks_dir": str(chunks_dir),
            "video_id": video_id,
        }

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())

        except Exception as e:
            self.logger.error(f"Error getting video duration: {e}")
            return 0.0

    def _extract_chunk(
        self, video_path: Path, chunk_path: Path, start_time: float, duration: float
    ) -> bool:
        """Extract a single chunk using ffmpeg."""
        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-i",
                str(video_path),
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c",
                "copy",  # Copy without re-encoding for speed
                "-avoid_negative_ts",
                "make_zero",
                str(chunk_path),
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            return chunk_path.exists() and chunk_path.stat().st_size > 0

        except Exception as e:
            self.logger.error(f"Error extracting chunk at {start_time}s: {e}")
            return False

    def process(
        self, video_path: Path, output_dir: Path = None, **kwargs
    ) -> dict[str, Any]:
        """Process video by extracting chunks."""
        return self.extract_chunks(video_path, output_dir)

    def cleanup(self):
        """Clean up temporary files if needed."""
        pass
