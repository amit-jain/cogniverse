#!/usr/bin/env python3
"""
Keyframe Processor - Pluggable keyframe extraction.

Extracts representative keyframes from videos using histogram comparison.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2

from ..processor_base import BaseProcessor


class KeyframeProcessor(BaseProcessor):
    """Handles keyframe extraction from videos."""

    PROCESSOR_NAME = "keyframe"

    def __init__(
        self,
        logger: logging.Logger,
        threshold: float = 0.999,
        max_frames: int = 3000,
        fps: float | None = None,
    ):
        """
        Initialize keyframe processor.

        Args:
            logger: Logger instance
            threshold: Similarity threshold for keyframe detection
            max_frames: Maximum number of keyframes to extract
            fps: Optional FPS for time-based extraction
        """
        super().__init__(logger)
        self.threshold = threshold
        self.max_frames = max_frames
        self.fps = fps
        self.extraction_mode = "fps" if fps else "histogram"

    @classmethod
    def from_config(
        cls, config: dict[str, Any], logger: logging.Logger
    ) -> "KeyframeProcessor":
        """Create keyframe processor from configuration."""
        return cls(
            logger=logger,
            threshold=config.get("threshold", 0.999),
            max_frames=config.get("max_frames", 3000),
            fps=config.get("fps"),
        )

    def extract_keyframes(
        self, video_path: Path, output_dir: Path = None
    ) -> dict[str, Any]:
        """Extract keyframes from video using specified method."""
        self.logger.info(
            f"📸 Extracting keyframes from: {video_path.name} (mode: {self.extraction_mode})"
        )

        video_id = video_path.stem

        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from src.common.utils.output_manager import get_output_manager

            output_manager = get_output_manager()
            keyframes_dir = output_manager.get_processing_dir("keyframes") / video_id
            metadata_file = (
                output_manager.get_processing_dir("metadata")
                / f"{video_id}_keyframes.json"
            )
        else:
            # Legacy path support
            keyframes_dir = output_dir / "keyframes" / video_id
            metadata_file = output_dir / "metadata" / f"{video_id}_keyframes.json"

        keyframes_dir.mkdir(parents=True, exist_ok=True)

        if self.extraction_mode == "fps" and self.fps:
            return self._extract_keyframes_fps(
                video_path, keyframes_dir, metadata_file, video_id
            )
        else:
            return self._extract_keyframes_histogram(
                video_path, keyframes_dir, metadata_file, video_id
            )

    def _extract_keyframes_fps(
        self, video_path: Path, keyframes_dir: Path, metadata_file: Path, video_id: str
    ) -> dict[str, Any]:
        """Extract keyframes at regular FPS intervals."""
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        keyframes = []
        frame_interval = int(video_fps / self.fps) if video_fps > self.fps else 1

        frame_idx = 0
        extracted_count = 0

        while cap.isOpened() and extracted_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at specified intervals
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps

                # Save keyframe
                keyframe_filename = f"{video_id}_keyframe_{extracted_count:04d}.jpg"
                keyframe_path = keyframes_dir / keyframe_filename

                cv2.imwrite(str(keyframe_path), frame)

                keyframes.append(
                    {
                        "frame_number": frame_idx,
                        "timestamp": timestamp,
                        "filename": keyframe_filename,
                        "path": str(keyframe_path),
                    }
                )

                extracted_count += 1

            frame_idx += 1

        cap.release()

        # Save metadata
        metadata = {
            "video_id": video_id,
            "video_path": str(video_path),
            "extraction_method": "fps",
            "fps": self.fps,
            "frame_interval": frame_interval,
            "total_frames": total_frames,
            "video_duration": duration,
            "keyframes_extracted": len(keyframes),
            "keyframes": keyframes,
        }

        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"   ✅ Extracted {len(keyframes)} keyframes using FPS method")

        return {
            "keyframes": keyframes,
            "metadata": metadata,
            "keyframes_dir": str(keyframes_dir),
            "video_id": video_id,
        }

    def _extract_keyframes_histogram(
        self, video_path: Path, keyframes_dir: Path, metadata_file: Path, video_id: str
    ) -> dict[str, Any]:
        """Extract keyframes using histogram comparison method."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        keyframes = []
        prev_hist = None
        frame_idx = 0

        start_time = time.time()

        while cap.isOpened() and len(keyframes) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate histogram for current frame
            hist = cv2.calcHist(
                [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()

            # Compare with previous frame
            is_keyframe = prev_hist is None
            if prev_hist is not None:
                # Calculate correlation coefficient
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                is_keyframe = correlation < self.threshold

            if is_keyframe:
                timestamp = frame_idx / fps

                # Save keyframe
                keyframe_filename = f"{video_id}_keyframe_{len(keyframes):04d}.jpg"
                keyframe_path = keyframes_dir / keyframe_filename

                cv2.imwrite(str(keyframe_path), frame)

                keyframes.append(
                    {
                        "frame_number": frame_idx,
                        "timestamp": timestamp,
                        "filename": keyframe_filename,
                        "path": str(keyframe_path),
                        "correlation": correlation if prev_hist is not None else 0.0,
                    }
                )

            prev_hist = hist
            frame_idx += 1

        cap.release()
        extraction_time = time.time() - start_time

        # Save metadata
        metadata = {
            "video_id": video_id,
            "video_path": str(video_path),
            "extraction_method": "histogram",
            "threshold": self.threshold,
            "total_frames": total_frames,
            "video_duration": duration,
            "extraction_time": extraction_time,
            "keyframes_extracted": len(keyframes),
            "keyframes": keyframes,
        }

        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(
            f"   ✅ Extracted {len(keyframes)} keyframes using histogram method in {extraction_time:.2f}s"
        )

        return {
            "keyframes": keyframes,
            "metadata": metadata,
            "keyframes_dir": str(keyframes_dir),
            "video_id": video_id,
        }
