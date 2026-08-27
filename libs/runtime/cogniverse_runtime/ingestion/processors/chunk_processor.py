#!/usr/bin/env python3
"""
Chunk Processor - Pluggable video chunk extraction.

Extracts video chunks for processing with models like ColQwen.
"""

import json
import logging
import math
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

        # Reuse previously extracted chunks when caching is enabled and a
        # complete, valid set is already on disk. With cache_chunks=False every
        # call re-extracts.
        if self.cache_chunks:
            cached = self._load_cached_chunks(metadata_file, chunks_dir, video_id)
            if cached is not None:
                self.logger.info(f"   ♻️  Reusing {len(cached['chunks'])} cached chunks")
                return cached

        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Get video duration
        duration = self._get_video_duration(video_path)
        if not math.isfinite(duration) or duration <= 0:
            self.logger.error(
                "   ❌ Invalid video duration %r for %s", duration, video_path
            )
            raise RuntimeError(
                f"ffprobe returned invalid duration {duration!r} for {video_path}"
            )

        # Calculate chunk positions
        chunks = []
        chunk_idx = 0
        start_time = 0.0

        while start_time < duration:
            end_time = min(start_time + self.chunk_duration, duration)

            # Generate chunk
            chunk_filename = f"{video_id}_chunk_{chunk_idx:04d}.mp4"
            chunk_path = chunks_dir / chunk_filename

            extracted = self._extract_chunk(
                video_path, chunk_path, start_time, end_time - start_time
            )
            if not extracted:
                raise RuntimeError(
                    f"ffmpeg reported no output for {video_path} at "
                    f"{start_time:.3f}s for {end_time - start_time:.3f}s"
                )
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

    def _load_cached_chunks(
        self, metadata_file: Path, chunks_dir: Path, video_id: str
    ) -> dict[str, Any] | None:
        """Return the cached chunk result if a complete valid set exists, else
        None. A set is valid only when the metadata parses and every chunk file
        it references is present and non-empty."""
        if not metadata_file.exists():
            return None
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        chunks = metadata.get("chunks")
        if not chunks:
            return None
        for chunk in chunks:
            chunk_path = Path(chunk.get("path", ""))
            if not (chunk_path.exists() and chunk_path.stat().st_size > 0):
                return None

        return {
            "chunks": chunks,
            "metadata": metadata,
            "chunks_dir": str(chunks_dir),
            "video_id": video_id,
        }

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
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

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffprobe is not available for {video_path}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                f"ffprobe failed for {video_path} with exit {exc.returncode}: {stderr}"
            ) from exc

        raw_stdout = result.stdout
        stdout = raw_stdout.strip()
        if not stdout:
            raise RuntimeError(
                f"ffprobe returned empty stdout for {video_path}: stdout={raw_stdout!r}"
            )

        try:
            duration = float(stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"ffprobe returned unparsable stdout for {video_path}: "
                f"stdout={raw_stdout!r}"
            ) from exc

        if not math.isfinite(duration):
            raise RuntimeError(
                f"ffprobe returned non-finite duration for {video_path}: "
                f"stdout={raw_stdout!r}"
            )
        if duration <= 0:
            raise RuntimeError(
                f"ffprobe returned non-positive duration {duration!r} for "
                f"{video_path}: stdout={raw_stdout!r}"
            )
        return duration

    def _extract_chunk(
        self, video_path: Path, chunk_path: Path, start_time: float, duration: float
    ) -> bool:
        """Extract one independently decodable video chunk using ffmpeg."""
        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-ss",
                str(start_time),
                "-i",
                str(video_path),
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-avoid_negative_ts",
                "make_zero",
                str(chunk_path),
            ]

            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True,
                timeout=max(120.0, duration * 10),
            )
            if not chunk_path.exists() or chunk_path.stat().st_size == 0:
                raise RuntimeError(
                    f"ffmpeg produced no bytes at {chunk_path} for {video_path} "
                    f"at {start_time:.3f}s for {duration:.3f}s"
                )
            return True

        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg failed for {video_path} at {start_time:.3f}s for "
                f"{duration:.3f}s with exit {exc.returncode}: {stderr}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffmpeg timed out for {video_path} at {start_time:.3f}s for "
                f"{duration:.3f}s after {exc.timeout}s"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"ffmpeg could not start for {video_path} at {start_time:.3f}s: {exc}"
            ) from exc

    def process(
        self, video_path: Path, output_dir: Path = None, **kwargs
    ) -> dict[str, Any]:
        """Process video by extracting chunks."""
        return self.extract_chunks(video_path, output_dir)

    def cleanup(self):
        """Clean up temporary files if needed."""
        pass
