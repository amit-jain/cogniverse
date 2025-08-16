"""Core document structures shared across the system."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import numpy as np


class MediaType(Enum):
    """Types of media that can be processed"""
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"
    VIDEO_SEGMENT = "video_segment"
    AUDIO = "audio"
    TEXT = "text"


@dataclass
class TemporalInfo:
    """Temporal information for media segments"""
    start_time: float
    end_time: float
    fps: Optional[float] = None
    frame_number: Optional[int] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SegmentInfo:
    """Information about a segment within a larger media file"""
    segment_idx: int
    total_segments: int
    segment_type: str = "frame"  # "frame", "clip", "chunk"
    overlap_with_previous: float = 0.0
    overlap_with_next: float = 0.0


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embeddings: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension"""
        if len(self.embeddings.shape) == 1:
            return self.embeddings.shape[0]
        elif len(self.embeddings.shape) == 2:
            return self.embeddings.shape[1]
        else:
            return self.embeddings.shape[-1]
    
    @property
    def num_embeddings(self) -> int:
        """Get the number of embeddings (1 for global, N for multi-vector)"""
        if len(self.embeddings.shape) == 1:
            return 1
        else:
            return self.embeddings.shape[0]


@dataclass
class Document:
    """Core document structure for the system"""
    doc_id: str
    media_type: MediaType
    content_path: Optional[str] = None
    content_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[EmbeddingResult] = None
    temporal_info: Optional[TemporalInfo] = None
    segment_info: Optional[SegmentInfo] = None
    transcription: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization"""
        result = {
            "doc_id": self.doc_id,
            "media_type": self.media_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
        
        if self.content_path:
            result["content_path"] = self.content_path
            
        if self.temporal_info:
            result["temporal_info"] = {
                "start_time": self.temporal_info.start_time,
                "end_time": self.temporal_info.end_time,
                "duration": self.temporal_info.duration,
                "fps": self.temporal_info.fps,
                "frame_number": self.temporal_info.frame_number
            }
            
        if self.segment_info:
            result["segment_info"] = {
                "segment_idx": self.segment_info.segment_idx,
                "total_segments": self.segment_info.total_segments,
                "segment_type": self.segment_info.segment_type
            }
            
        if self.transcription:
            result["transcription"] = self.transcription
            
        return result