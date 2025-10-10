#!/usr/bin/env python3
"""
Keyframe Extraction Step

Extracts representative keyframes from videos using histogram comparison.
"""

import json
import time
from pathlib import Path
from typing import Any

import cv2


class KeyframeExtractor:
    """Handles keyframe extraction from videos"""

    def __init__(self, threshold: float = 0.999, max_frames: int = 3000):
        self.threshold = threshold
        self.max_frames = max_frames

    def extract_keyframes(
        self, video_path: Path, output_dir: Path = None
    ) -> dict[str, Any]:
        """Extract keyframes from video using histogram comparison"""
        print(f"📸 Extracting keyframes from: {video_path.name}")

        video_id = video_path.stem

        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from cogniverse_core.common.utils.output_manager import get_output_manager

            output_manager = get_output_manager()
            keyframes_dir = output_manager.get_processing_dir("keyframes") / video_id
            metadata_file = (
                output_manager.get_processing_dir("metadata")
                / f"{video_id}_keyframes.json"
            )
        else:
            # For testing - should migrate tests to use OutputManager
            keyframes_dir = output_dir / "keyframes" / video_id
            metadata_file = output_dir / "metadata" / f"{video_id}_keyframes.json"

        keyframes_dir.mkdir(parents=True, exist_ok=True)

        # Remove caching - always extract keyframes

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        keyframes = []
        prev_hist = None
        frame_count = 0
        keyframe_count = 0

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate histogram for comparison
            hist = cv2.calcHist(
                [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()

            is_keyframe = False
            if prev_hist is None:
                is_keyframe = True  # First frame is always a keyframe
            else:
                # Compare histograms using correlation
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < self.threshold:
                    is_keyframe = True

            if is_keyframe:
                # Save keyframe
                timestamp = frame_count / fps if fps > 0 else 0
                keyframe_filename = f"frame_{keyframe_count:04d}.jpg"
                keyframe_path = keyframes_dir / keyframe_filename

                cv2.imwrite(str(keyframe_path), frame)

                keyframes.append(
                    {
                        "frame_id": keyframe_count,
                        "original_frame_number": frame_count,
                        "timestamp": timestamp,
                        "path": str(keyframe_path),
                        "filename": keyframe_filename,
                    }
                )

                keyframe_count += 1
                prev_hist = hist

                # Limit max frames per video
                if keyframe_count >= self.max_frames:
                    print(f"  ⚠️ Reached max frames limit ({self.max_frames})")
                    break

            frame_count += 1

            # Progress reporting
            if frame_count % 1000 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  🔄 Progress: {progress:.1f}% ({keyframe_count} keyframes)")

        cap.release()
        processing_time = time.time() - start_time

        # Create metadata
        metadata = {
            "video_id": video_id,
            "video_path": str(video_path),
            "keyframes": keyframes,
            "stats": {
                "total_keyframes": keyframe_count,
                "total_frames": frame_count,
                "fps": fps,
                "duration_seconds": duration,
                "processing_time_seconds": processing_time,
                "threshold_used": self.threshold,
            },
            "created_at": time.time(),
        }

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✅ Extracted {keyframe_count} keyframes in {processing_time:.1f}s")
        return metadata
