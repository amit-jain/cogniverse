#!/usr/bin/env python3
"""
Base Embedding Generator - Abstract base class and data structures.

This module defines the core abstractions and data structures used throughout
the embedding generation pipeline:

1. MediaType: Enum for different media types (VIDEO, IMAGE, TEXT, etc.)
2. TemporalInfo: Temporal metadata for time-based media
3. SegmentInfo: Information about video segments
4. Document: Universal document structure for all media types
5. EmbeddingGenerator: Abstract base class for generators

The Document class is the key abstraction that allows backend-agnostic processing.
It carries raw embeddings and metadata, which backends convert to their specific formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Union
from enum import Enum
import numpy as np
import logging
from dataclasses import dataclass
import time

# Import from core
from src.core import Document, MediaType, TemporalInfo, SegmentInfo, EmbeddingResult




@dataclass
class ProcessingConfig:
    """Configuration for processing"""
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    save_progress_interval: int = 10


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    video_id: str
    total_documents: int
    documents_processed: int
    documents_fed: int
    processing_time: float
    errors: List[str]
    metadata: Dict[str, Any]


class EmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.
    
    This class defines the interface that all embedding generators must implement.
    Concrete implementations handle different processing profiles (frame-based,
    direct video, etc.) while maintaining a consistent interface.
    
    The generator is responsible for:
    1. Loading appropriate models based on profile
    2. Processing videos/frames to generate embeddings
    3. Creating Document objects with metadata
    4. Feeding documents to the backend
    
    Subclasses must implement:
    - generate_embeddings(): Main entry point for processing
    - process_segment(): Process individual segments/frames
    - create_document(): Create documents from embeddings
    - _feed_single_document(): Feed document to backend
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.processing_config = ProcessingConfig()
        
    @abstractmethod
    def generate_embeddings(
        self, 
        video_data: Dict[str, Any], 
        output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for a video"""
        pass
    
    @abstractmethod
    def process_segment(
        self,
        segment_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single segment/frame and return embedding data"""
        pass
    
    @abstractmethod
    def create_document(
        self,
        segment_data: Dict[str, Any],
        embeddings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a document for the backend"""
        pass
    
    def process_segments(
        self,
        segments: List[Dict[str, Any]],
        feed_immediately: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """Process segments and yield documents"""
        for i, segment in enumerate(segments):
            try:
                # Process segment
                embedding_data = self.process_segment(segment)
                if not embedding_data:
                    self.logger.warning(f"Failed to process segment {i}")
                    continue
                
                # Create document
                document = self.create_document(segment, embedding_data)
                
                # Yield document
                yield document
                
                # Feed immediately if requested
                if feed_immediately:
                    self._feed_single_document(document)
                    
            except Exception as e:
                self.logger.error(f"Error processing segment {i}: {e}")
                continue
    
    @abstractmethod
    def _feed_single_document(self, document: Dict[str, Any]) -> bool:
        """Feed a single document to the backend"""
        pass
    
    def save_progress(
        self,
        video_id: str,
        output_dir: Path,
        progress_data: Dict[str, Any]
    ) -> None:
        """Save processing progress"""
        progress_file = output_dir / "embeddings" / f"{video_id}_progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(progress_file, 'w') as f:
            json.dump({
                **progress_data,
                "timestamp": time.time()
            }, f, indent=2)
    
    def load_progress(
        self,
        video_id: str,
        output_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Load previous progress if exists"""
        progress_file = output_dir / "embeddings" / f"{video_id}_progress.json"
        
        if progress_file.exists():
            try:
                import json
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load progress: {e}")
        
        return None