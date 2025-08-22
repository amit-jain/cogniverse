#!/usr/bin/env python3
"""
Video Chunk Extractor - Splits videos into temporal chunks for processing.

This replaces keyframe extraction for video_chunks processing (e.g., ColQwen).
Instead of extracting individual frames, it splits the video into temporal chunks
that can be processed directly by multi-vector models.
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
import json
import time


class VideoChunkExtractor:
    """
    Extracts temporal chunks from videos for direct processing by models like ColQwen.

    This avoids the inefficiency of:
    1. Extracting keyframes from the whole video
    2. Then splitting into chunks
    3. Then extracting frames AGAIN from each chunk

    Instead, we directly split the video into chunks and cache them.
    """

    def __init__(
        self,
        chunk_duration: float = 30.0,
        chunk_overlap: float = 0.0,
        max_chunks: Optional[int] = None,
        cache_chunks: bool = True,
    ):
        """
        Initialize the video chunk extractor.

        Args:
            chunk_duration: Duration of each chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            max_chunks: Maximum number of chunks to extract (None for all)
            cache_chunks: Whether to cache extracted chunks to disk
        """
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.cache_chunks = cache_chunks
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_chunks(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Extract video chunks and save them to disk.

        Args:
            video_path: Path to the input video
            output_dir: Directory to save extracted chunks

        Returns:
            Dictionary containing chunk metadata and paths
        """
        video_id = video_path.stem
        chunks_dir = output_dir / "chunks" / video_id
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Get video duration
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            self.logger.error(f"Invalid video duration: {duration}")
            return {"chunks": [], "video_id": video_id, "error": "Invalid duration"}

        self.logger.info(
            f"Video duration: {duration:.2f}s, chunk duration: {self.chunk_duration}s"
        )

        # Calculate chunk boundaries
        chunks_metadata = []
        chunk_starts = []

        if self.chunk_overlap > 0:
            # With overlap
            stride = self.chunk_duration - self.chunk_overlap
            current = 0.0
            while current < duration:
                chunk_starts.append(current)
                current += stride
        else:
            # Without overlap
            num_chunks = int(np.ceil(duration / self.chunk_duration))
            for i in range(num_chunks):
                chunk_starts.append(i * self.chunk_duration)

        # Limit chunks if max_chunks is set
        if self.max_chunks and len(chunk_starts) > self.max_chunks:
            chunk_starts = chunk_starts[: self.max_chunks]

        self.logger.info(f"Extracting {len(chunk_starts)} chunks from video")

        # Extract each chunk
        for idx, start_time in enumerate(chunk_starts):
            end_time = min(start_time + self.chunk_duration, duration)
            chunk_duration = end_time - start_time

            # Define output path for this chunk
            chunk_filename = f"chunk_{idx:04d}_{int(start_time)}_{int(end_time)}.mp4"
            chunk_path = chunks_dir / chunk_filename

            # Check if chunk already exists (caching)
            if self.cache_chunks and chunk_path.exists():
                self.logger.info(f"Using cached chunk {idx}: {chunk_filename}")
            else:
                # Extract chunk using ffmpeg
                self.logger.info(
                    f"Extracting chunk {idx}: {start_time:.2f}s - {end_time:.2f}s"
                )
                success = self._extract_chunk_ffmpeg(
                    video_path, chunk_path, start_time, chunk_duration
                )

                if not success:
                    self.logger.error(f"Failed to extract chunk {idx}")
                    continue

            # Add chunk metadata
            chunk_metadata = {
                "chunk_id": idx,
                "chunk_path": str(chunk_path),
                "start_time": start_time,
                "end_time": end_time,
                "duration": chunk_duration,
                "filename": chunk_filename,
            }

            # Optionally extract key frame from chunk for preview
            preview_path = chunks_dir / f"chunk_{idx:04d}_preview.jpg"
            if not preview_path.exists():
                self._extract_preview_frame(chunk_path, preview_path)
            if preview_path.exists():
                chunk_metadata["preview_path"] = str(preview_path)

            chunks_metadata.append(chunk_metadata)

        # Save metadata
        metadata = {
            "video_id": video_id,
            "video_path": str(video_path),
            "total_duration": duration,
            "chunk_duration": self.chunk_duration,
            "chunk_overlap": self.chunk_overlap,
            "num_chunks": len(chunks_metadata),
            "chunks": chunks_metadata,
            "extracted_at": time.time(),
        }

        # Save metadata to JSON
        metadata_path = chunks_dir / "chunks_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Extracted {len(chunks_metadata)} chunks for {video_id}")

        return metadata

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

    def _extract_chunk_ffmpeg(
        self, input_path: Path, output_path: Path, start_time: float, duration: float
    ) -> bool:
        """Extract a video chunk using ffmpeg."""
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c",
                "copy",  # Copy codec for speed
                "-avoid_negative_ts",
                "make_zero",
                "-y",  # Overwrite output
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
            )
            return False

    def _extract_preview_frame(self, video_path: Path, output_path: Path) -> bool:
        """Extract a preview frame from the chunk."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            # Get frame from middle of chunk
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(str(output_path), frame)
                cap.release()
                return ret
            cap.release()
            return False
        except Exception as e:
            self.logger.error(f"Error extracting preview frame: {e}")
            return False

    def load_cached_chunks(
        self, video_path: Path, output_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached chunks metadata if available.

        Args:
            video_path: Path to the original video
            output_dir: Directory containing cached chunks

        Returns:
            Chunks metadata if cached, None otherwise
        """
        video_id = video_path.stem
        metadata_path = output_dir / "chunks" / video_id / "chunks_metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Verify all chunk files exist
                all_exist = all(
                    Path(chunk["chunk_path"]).exists()
                    for chunk in metadata.get("chunks", [])
                )

                if all_exist:
                    self.logger.info(f"Loaded cached chunks for {video_id}")
                    return metadata
                else:
                    self.logger.warning(f"Some cached chunks missing for {video_id}")
            except Exception as e:
                self.logger.error(f"Error loading cached chunks: {e}")

        return None
