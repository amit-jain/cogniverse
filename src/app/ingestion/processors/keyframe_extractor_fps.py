#!/usr/bin/env python3
"""
FPS-based Keyframe Extraction Step

Extracts keyframes at fixed frame-per-second intervals from videos.
"""

import json
import time
import cv2
from pathlib import Path
from typing import Dict, Any


class FPSKeyframeExtractor:
    """Handles FPS-based keyframe extraction from videos"""

    def __init__(self, fps: float = 1.0, max_frames: int = 3000):
        """
        Initialize FPS-based keyframe extractor

        Args:
            fps: Frames per second to extract (default 1.0 = 1 frame per second)
            max_frames: Maximum number of frames to extract per video
        """
        self.target_fps = fps
        self.max_frames = max_frames

    def extract_keyframes(
        self, video_path: Path, output_dir: Path = None
    ) -> Dict[str, Any]:
        """Extract keyframes from video at fixed FPS intervals"""
        print(
            f"📸 Extracting keyframes at {self.target_fps} FPS from: {video_path.name}"
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

        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        # Calculate frame interval based on video FPS and target FPS
        frame_interval = int(video_fps / self.target_fps) if video_fps > 0 else 1
        frame_interval = max(1, frame_interval)  # Ensure at least 1

        keyframes = []
        frame_count = 0
        keyframe_count = 0

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at FPS intervals
            if frame_count % frame_interval == 0:
                # Save keyframe
                timestamp = frame_count / video_fps if video_fps > 0 else 0
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
                "video_fps": video_fps,
                "target_fps": self.target_fps,
                "frame_interval": frame_interval,
                "duration_seconds": duration,
                "processing_time_seconds": processing_time,
                "extraction_method": "fps",
            },
            "created_at": time.time(),
        }

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✅ Extracted {keyframe_count} keyframes in {processing_time:.1f}s")
        print(
            f"  📊 Video FPS: {video_fps:.1f}, Target FPS: {self.target_fps}, Frame interval: {frame_interval}"
        )
        return metadata
