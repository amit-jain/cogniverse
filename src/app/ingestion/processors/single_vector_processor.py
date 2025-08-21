#!/usr/bin/env python3
"""
Single Vector Video Processor - Generic processor for single-vector embedding approaches

This processor is model-agnostic and strategy-flexible. It handles:
- Segmentation strategies (chunks, windows, global)
- Frame extraction
- Transcript alignment
- Document preparation

The same processor can be used for:
- single__video_videoprism_large_6s (6-second chunks)
- single__video_videoprism_large_30s (30-second windows)
- single__video_videoprism_lvt_base_global (entire video)
- single__video_anymodel_anystrategy (any future approach)
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
import cv2
from dataclasses import dataclass
import time
import asyncio

from ..processor_base import BaseProcessor

logger = logging.getLogger(__name__)


@dataclass
class VideoSegment:
    """Generic video segment - can represent any portion of video"""
    segment_id: int
    start_time: float
    end_time: float
    frames: List[np.ndarray]
    frame_timestamps: List[float]
    transcript_segments: List[Dict[str, Any]]
    transcript_text: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without frame data)"""
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "frame_count": len(self.frames),
            "transcript_text": self.transcript_text,
            "metadata": self.metadata or {}
        }


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
        sampling_fps: float = 2.0,
        max_frames_per_segment: int = 12,
        min_segment_duration: float = 2.0,
        store_as_single_doc: bool = True,
        cache: Optional[Any] = None,
        **kwargs
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
            sampling_fps: Frame sampling rate within segments
            max_frames_per_segment: Maximum frames to extract per segment
            min_segment_duration: Minimum duration for last segment
            store_as_single_doc: If True, store all segments in one document
        """
        # Pass all parameters to parent using locals() to avoid hardcoding
        params = locals().copy()
        params.pop('self')  # Remove self
        params.pop('kwargs')  # Remove kwargs since we'll spread it
        params.update(kwargs)  # Add any additional kwargs
        
        super().__init__(**params)
        
        # Set instance variables for backward compatibility with existing code
        self.strategy = strategy
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap if strategy != "global" else 0
        self.sampling_fps = sampling_fps
        self.max_frames_per_segment = max_frames_per_segment
        self.min_segment_duration = min_segment_duration
        self.store_as_single_doc = store_as_single_doc
        self.cache = cache
        
    def process_video(
        self,
        video_path: Path,
        transcript_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process video according to configured strategy.
        
        Returns:
            Dict containing:
            - segments: List of VideoSegment objects
            - metadata: Processing metadata
            - full_transcript: Combined transcript
            - document_structure: How to structure for storage
        """
        self.logger.info(f"Processing video: {video_path.name} with strategy: {self.strategy}")
        
        # Get video info
        video_info = self._get_video_info(video_path)
        duration = video_info["duration"]
        
        # Calculate segment boundaries
        segment_boundaries = self._calculate_segment_boundaries(duration)
        self.logger.info(f"Duration: {duration:.1f}s, segments: {len(segment_boundaries)}")
        
        # Process each segment
        segments = []
        for i, (start_time, end_time) in enumerate(segment_boundaries):
            segment = self._process_segment(
                video_path=video_path,
                segment_id=i,
                start_time=start_time,
                end_time=end_time,
                video_info=video_info,
                transcript_data=transcript_data
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
                "store_as_single_doc": self.store_as_single_doc,
                "processed_at": time.time()
            },
            "full_transcript": full_transcript,
            "document_structure": self._get_document_structure()
        }
        
        # Add any additional metadata
        if metadata:
            result["metadata"].update(metadata)
            
        return result
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata"""
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
        cap.release()
        return info
    
    def _calculate_segment_boundaries(self, duration: float) -> List[Tuple[float, float]]:
        """Calculate segment boundaries based on strategy"""
        if self.strategy == "global":
            return [(0.0, duration)]
            
        segments = []
        start = 0.0
        
        while start < duration:
            end = min(start + self.segment_duration, duration)
            
            # Check if remaining duration is too small
            if duration - end < self.min_segment_duration and segments:
                # Extend last segment instead of creating tiny one
                segments[-1] = (segments[-1][0], duration)
                break
            
            segments.append((start, end))
            
            # Move to next segment
            if self.segment_overlap > 0:
                start += (self.segment_duration - self.segment_overlap)
            else:
                start = end
                
        return segments
    
    def _process_segment(
        self,
        video_path: Path,
        segment_id: int,
        start_time: float,
        end_time: float,
        video_info: Dict[str, Any],
        transcript_data: Optional[Dict[str, Any]] = None
    ) -> VideoSegment:
        """Process a single video segment"""
        # Extract frames
        frames, timestamps = self._extract_frames(
            video_path, start_time, end_time, video_info["fps"], segment_id
        )
        
        # Align transcripts
        transcript_segments = []
        transcript_text = ""
        
        if transcript_data and "segments" in transcript_data:
            transcript_segments = self._align_transcript_segments(
                transcript_data["segments"], start_time, end_time
            )
            transcript_text = " ".join([
                seg["text"].strip() for seg in transcript_segments
            ])
        
        # Create segment
        return VideoSegment(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
            frames=frames,
            frame_timestamps=timestamps,
            transcript_segments=transcript_segments,
            transcript_text=transcript_text,
            metadata={
                "duration": end_time - start_time,
                "frame_count": len(frames),
                "has_transcript": bool(transcript_text)
            }
        )
    
    def _extract_frames(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        video_fps: float,
        segment_id: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames from video segment with caching"""
        # Check cache first if available
        if self.cache:
            # Handle asyncio.run() in nested event loop
            try:
                loop = asyncio.get_running_loop()
                # We're already in an event loop, use ThreadPoolExecutor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    cached_result = executor.submit(asyncio.run, self.cache.get_segment_frames(
                        str(video_path),
                        segment_id,
                        start_time,
                        end_time,
                        self.sampling_fps,
                        self.max_frames_per_segment,
                        load_images=True
                    )).result()
            except RuntimeError:
                # No event loop, safe to use asyncio.run
                cached_result = asyncio.run(self.cache.get_segment_frames(
                    str(video_path),
                    segment_id,
                    start_time,
                    end_time,
                    self.sampling_fps,
                    self.max_frames_per_segment,
                    load_images=True
                ))
            
            if cached_result:
                if isinstance(cached_result, tuple):
                    metadata, frames = cached_result
                    self.logger.info(f"Using cached frames for segment {segment_id}")
                    return frames, metadata.get("timestamps", [])
        
        # No cache or cache miss - extract frames
        cap = cv2.VideoCapture(str(video_path))
        
        # Calculate frame extraction pattern
        segment_duration = end_time - start_time
        desired_frames = int(segment_duration * self.sampling_fps)
        frames_to_extract = min(desired_frames, self.max_frames_per_segment)
        
        if frames_to_extract <= 0:
            cap.release()
            return [], []
        
        # Calculate which frames to extract
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        total_frames = end_frame - start_frame
        
        if total_frames <= frames_to_extract:
            # Extract all frames
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Sample evenly
            step = total_frames / frames_to_extract
            frame_indices = [int(start_frame + i * step) for i in range(frames_to_extract)]
        
        # Extract frames
        frames = []
        timestamps = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                timestamps.append(frame_idx / video_fps)
                
        cap.release()
        
        # Cache the frames if cache is available
        if self.cache and frames:
            # Handle asyncio.run() in nested event loop
            try:
                loop = asyncio.get_running_loop()
                # We're already in an event loop, use ThreadPoolExecutor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.submit(asyncio.run, self.cache.set_segment_frames(
                        str(video_path),
                        segment_id,
                        start_time,
                        end_time,
                        frames,
                        timestamps,
                        self.sampling_fps,
                        self.max_frames_per_segment
                    )).result()
            except RuntimeError:
                # No event loop, safe to use asyncio.run
                asyncio.run(self.cache.set_segment_frames(
                    str(video_path),
                    segment_id,
                    start_time,
                    end_time,
                    frames,
                    timestamps,
                    self.sampling_fps,
                    self.max_frames_per_segment
                ))
            self.logger.info(f"Cached frames for segment {segment_id}")
        
        return frames, timestamps
    
    def _align_transcript_segments(
        self,
        segments: List[Dict[str, Any]],
        start_time: float,
        end_time: float
    ) -> List[Dict[str, Any]]:
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
                aligned_seg["relative_end"] = min(end_time - start_time, seg_end - start_time)
                aligned.append(aligned_seg)
                
        return aligned
    
    def _combine_transcripts(self, segments: List[VideoSegment]) -> str:
        """Combine all segment transcripts avoiding duplicates"""
        if self.strategy == "global":
            # Global strategy - just return the transcript
            return segments[0].transcript_text if segments else ""
            
        # For overlapping segments, deduplicate
        seen_segments = set()
        transcript_parts = []
        
        for segment in segments:
            for trans_seg in segment.transcript_segments:
                seg_key = (
                    trans_seg.get("start", 0),
                    trans_seg.get("text", "").strip()
                )
                if seg_key not in seen_segments:
                    seen_segments.add(seg_key)
                    transcript_parts.append(seg_key)
        
        # Sort by timestamp and combine
        transcript_parts.sort(key=lambda x: x[0])
        return " ".join([text for _, text in transcript_parts if text])
    
    def _get_document_structure(self) -> Dict[str, Any]:
        """Define how documents should be structured for this strategy"""
        if self.store_as_single_doc:
            return {
                "type": "single_document",
                "description": "All segments stored in one document with tensor arrays"
            }
        else:
            return {
                "type": "multiple_documents",
                "description": "Each segment stored as separate document"
            }
    
    def prepare_for_embedding_generation(
        self,
        segments: List[VideoSegment],
        model_type: str = "videoprism"
    ) -> List[Dict[str, Any]]:
        """
        Prepare segments for embedding generation.
        Model-agnostic preparation.
        """
        prepared = []
        
        for segment in segments:
            prepared.append({
                "segment_id": segment.segment_id,
                "frames": segment.frames,
                "frame_timestamps": segment.frame_timestamps,
                "metadata": {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "transcript": segment.transcript_text,
                    "model_type": model_type
                }
            })
            
        return prepared