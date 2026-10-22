#!/usr/bin/env python3
"""
Single Vector Video Processor - Generic processor for single-vector embedding approaches

This processor is model-agnostic and strategy-flexible. It handles:
- Segmentation strategies (chunks, windows, global)
- Transcript alignment

The same processor can be used for:
- video_embeddinggemma2_sv_chunk_6s (6-second chunks)
- single__video_anymodel_anystrategy (any future approach)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2

from cogniverse_core.common.utils.async_bridge import run_coro_blocking

from ..processor_base import BaseProcessor

logger = logging.getLogger(__name__)


@dataclass
class VideoSegment:
    """Generic video segment - can represent any portion of video"""

    segment_id: int
    start_time: float
    end_time: float
    transcript_segments: list[dict[str, Any]]
    transcript_text: str = ""
    metadata: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary without the aligned transcript segments."""
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "transcript_text": self.transcript_text,
            "metadata": self.metadata or {},
        }


def _transcript_fingerprint(transcript_data: dict[str, Any] | None) -> str:
    """Stable digest of the transcript a segmentation was aligned against —
    part of the cache key so a re-transcription invalidates cached results."""
    if not transcript_data:
        return "none"
    digest = hashlib.sha256(
        json.dumps(transcript_data, sort_keys=True, default=str).encode()
    ).hexdigest()
    return digest[:16]


def _segment_cache_payload(segment: VideoSegment) -> dict[str, Any]:
    """Full-fidelity serialization for the segmentation cache — unlike
    ``to_dict``, keeps transcript_segments so a cache
    hit reconstructs the exact VideoSegment. Copies every mutable field so
    later caller mutation can't reach the stored entry."""
    return {
        "segment_id": segment.segment_id,
        "start_time": segment.start_time,
        "end_time": segment.end_time,
        "transcript_segments": [dict(s) for s in segment.transcript_segments],
        "transcript_text": segment.transcript_text,
        "metadata": dict(segment.metadata or {}),
    }


def _segment_from_cache(data: dict[str, Any]) -> VideoSegment:
    return VideoSegment(
        segment_id=data["segment_id"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        transcript_segments=[dict(s) for s in data["transcript_segments"]],
        transcript_text=data["transcript_text"],
        metadata=dict(data["metadata"]),
    )


class SingleVectorVideoProcessor(BaseProcessor):
    """
    Generic processor for any single-vector video embedding approach.
    Configurable for different strategies without being tied to specific models.
    """

    PROCESSOR_NAME = "single_vector"

    def __init__(
        self,
        logger: logging.Logger,
        strategy: Literal["chunks", "windows", "global"] = "chunks",
        segment_duration: float = 6.0,
        segment_overlap: float = 1.0,
        min_segment_duration: float = 2.0,
        cache: Any | None = None,
        **kwargs,
    ):
        """
        Initialize processor with strategy configuration.

        Args:
            strategy: Segmentation strategy
                - "chunks": Small overlapping segments (e.g., 6s for temporal precision)
                - "windows": Larger segments (e.g., 30s for balanced approach)
                - "global": Entire video as one segment
            segment_duration: Duration of each segment in seconds
            segment_overlap: Overlap between segments (ignored for global)
            min_segment_duration: Minimum duration for last segment
        """
        # Pass all parameters to parent using locals() to avoid hardcoding
        params = locals().copy()
        params.pop("self")  # Remove self
        params.pop("kwargs")  # Remove kwargs since we'll spread it
        params.update(kwargs)  # Add any additional kwargs

        super().__init__(**params)

        # Expose processing parameters as instance attributes
        self.strategy = strategy
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap if strategy != "global" else 0
        if segment_duration <= 0:
            raise ValueError(
                f"segment_duration must be positive, got {segment_duration}"
            )
        if not 0 <= self.segment_overlap < segment_duration:
            raise ValueError(
                "segment_overlap must be at least 0 and less than segment_duration "
                f"({segment_duration}), got {segment_overlap}"
            )
        if min_segment_duration < 0:
            raise ValueError(
                f"min_segment_duration must not be negative, got {min_segment_duration}"
            )
        self.min_segment_duration = min_segment_duration
        self.cache = cache

    def process_video(
        self,
        video_path: Path,
        transcript_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process video according to configured strategy.

        Returns:
            Dict containing:
            - segments: List of VideoSegment objects
            - metadata: Processing metadata
            - full_transcript: Combined transcript
        """
        self.logger.info(
            f"Processing video: {video_path.name} with strategy: {self.strategy}"
        )

        cache_params = {
            "strategy": self.strategy,
            "segment_duration": self.segment_duration,
            "segment_overlap": self.segment_overlap,
            "transcript_fingerprint": _transcript_fingerprint(transcript_data),
        }
        if self.cache is not None:
            cached = run_coro_blocking(
                self.cache.get_segmentation(str(video_path), **cache_params)
            )
            if cached is not None:
                result = {
                    "segments": [_segment_from_cache(d) for d in cached["segments"]],
                    # Copy: the caller-metadata merge below must not poison
                    # the cached payload for later hits.
                    "metadata": dict(cached["metadata"]),
                    "full_transcript": cached["full_transcript"],
                }
                if metadata:
                    result["metadata"].update(metadata)
                return result

        # Get video info
        video_info = self._get_video_info(video_path)
        duration = video_info["duration"]
        if duration <= 0:
            raise ValueError(
                f"{video_path.name} has no readable duration "
                f"(fps {video_info['fps']}, {video_info['total_frames']} frames)"
            )

        # Calculate segment boundaries
        segment_boundaries = self._calculate_segment_boundaries(duration)
        self.logger.info(
            f"Duration: {duration:.1f}s, segments: {len(segment_boundaries)}"
        )

        # Process each segment
        segments = []
        for i, (start_time, end_time) in enumerate(segment_boundaries):
            segment = self._process_segment(
                video_path=video_path,
                segment_id=i,
                start_time=start_time,
                end_time=end_time,
                transcript_data=transcript_data,
            )
            segments.append(segment)

        # Combine transcripts
        full_transcript = self._combine_transcripts(segments)

        # Prepare result
        result = {
            "segments": segments,
            "metadata": {
                "video_id": video_path.stem,
                "duration": duration,
                "fps": video_info["fps"],
                "total_frames": video_info["total_frames"],
                "num_segments": len(segments),
                "strategy": self.strategy,
                "segment_duration": self.segment_duration,
                "segment_overlap": self.segment_overlap,
                "processed_at": time.time(),
            },
            "full_transcript": full_transcript,
        }

        if self.cache is not None:
            # Own dicts for the payload: the caller-metadata merge below (and
            # any caller mutation) must not reach the stored entry.
            payload = {
                "segments": [_segment_cache_payload(s) for s in segments],
                "metadata": dict(result["metadata"]),
                "full_transcript": full_transcript,
            }
            run_coro_blocking(
                self.cache.set_segmentation(str(video_path), payload, **cache_params)
            )

        # Add any additional metadata
        if metadata:
            result["metadata"].update(metadata)

        return result

    def _get_video_info(self, video_path: Path) -> dict[str, Any]:
        """Get video metadata"""
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
        cap.release()
        return info

    def _calculate_segment_boundaries(
        self, duration: float
    ) -> list[tuple[float, float]]:
        """Calculate segment boundaries based on strategy"""
        if self.strategy == "global":
            return [(0.0, duration)]

        segments = []
        start = 0.0

        while start < duration:
            end = min(start + self.segment_duration, duration)

            # A remainder too short to stand alone joins the chunk before it.
            if end - start < self.min_segment_duration and segments:
                segments[-1] = (segments[-1][0], duration)
                break

            segments.append((start, end))
            if end >= duration:
                break

            # Move to next segment
            if self.segment_overlap > 0:
                start += self.segment_duration - self.segment_overlap
            else:
                start = end

        return segments

    def _process_segment(
        self,
        video_path: Path,
        segment_id: int,
        start_time: float,
        end_time: float,
        transcript_data: dict[str, Any] | None = None,
    ) -> VideoSegment:
        """Process one segment: its boundaries and the transcript aligned
        to them. The embedding stage reads the time range from the video."""
        # Align transcripts
        transcript_segments = []
        transcript_text = ""

        if transcript_data and "segments" in transcript_data:
            transcript_segments = self._align_transcript_segments(
                transcript_data["segments"], start_time, end_time
            )
            transcript_text = " ".join(
                text
                for seg in sorted(transcript_segments, key=lambda s: s.get("start", 0))
                if (text := seg.get("text", "").strip())
            )

        # Create segment
        return VideoSegment(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
            transcript_segments=transcript_segments,
            transcript_text=transcript_text,
            metadata={
                "duration": end_time - start_time,
                "has_transcript": bool(transcript_text),
            },
        )

    def _align_transcript_segments(
        self, segments: list[dict[str, Any]], start_time: float, end_time: float
    ) -> list[dict[str, Any]]:
        """Align transcript segments with video segment"""
        aligned = []

        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", seg_start)

            # Check overlap
            if seg_end > start_time and seg_start < end_time:
                aligned_seg = seg.copy()
                # Add relative timestamps
                aligned_seg["relative_start"] = max(0, seg_start - start_time)
                aligned_seg["relative_end"] = min(
                    end_time - start_time, seg_end - start_time
                )
                aligned.append(aligned_seg)

        return aligned

    def _combine_transcripts(self, segments: list[VideoSegment]) -> str:
        """Combine all segment transcripts avoiding duplicates"""
        if self.strategy == "global":
            # Global strategy - just return the transcript
            return segments[0].transcript_text if segments else ""

        # For overlapping segments, deduplicate
        seen_segments = set()
        transcript_parts = []

        for segment in segments:
            for trans_seg in segment.transcript_segments:
                seg_key = (trans_seg.get("start", 0), trans_seg.get("text", "").strip())
                if seg_key not in seen_segments:
                    seen_segments.add(seg_key)
                    transcript_parts.append(seg_key)

        # Sort by timestamp and combine
        transcript_parts.sort(key=lambda x: x[0])
        return " ".join([text for _, text in transcript_parts if text])

    def process(
        self, video_path: Path, output_dir: Path = None, **kwargs
    ) -> dict[str, Any]:
        """Process video using single vector approach.

        ``output_dir`` is accepted for BaseProcessor signature compatibility
        but unused; transcript_data/metadata are forwarded by keyword so they
        reach ``process_video`` instead of being dropped.
        """
        return self.process_video(
            video_path,
            transcript_data=kwargs.get("transcript_data"),
            metadata=kwargs.get("metadata"),
        )
