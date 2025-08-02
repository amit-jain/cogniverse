#!/usr/bin/env python3
"""
Embedding Generator Implementation - Backend-agnostic, produces raw embeddings.

This module provides the main implementation of the embedding generator that supports
multiple video processing profiles:

1. Frame-based Processing (e.g., ColPali):
   - Processes pre-extracted video frames
   - Generates embeddings for individual frames
   - Suitable for models that work on static images
   
2. Direct Video Processing (e.g., ColQwen, VideoPrism):
   - Processes video segments directly without frame extraction
   - Handles temporal information natively
   - More efficient for video-native models

The generator is backend-agnostic and delegates format conversion to the backend client.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time
import json
import numpy as np
import torch
from PIL import Image

from .embedding_generator import (
    EmbeddingGenerator, EmbeddingResult, ProcessingConfig,
    Document, MediaType, TemporalInfo, SegmentInfo
)
from src.models import get_or_load_model


class EmbeddingGeneratorImpl(EmbeddingGenerator):
    """
    Concrete implementation of the embedding generator.
    
    This class handles the core logic of generating embeddings from videos using
    different processing profiles. It supports both frame-based and direct video
    processing approaches.
    
    Attributes:
        profile_config: Configuration dict containing profile-specific settings
        process_type: Type of processing ("frame_based" or "direct_video*")
        model_name: Name of the embedding model to use
        media_type: MediaType enum indicating the type of documents created
        backend_client: Backend client for storing documents
        
    The generator follows this flow:
    1. Load model based on process_type (only for direct video)
    2. Process video/frames to generate raw numpy embeddings
    3. Create Document objects with metadata
    4. Pass Documents to backend for format conversion and storage
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        profile_config: Dict[str, Any] = None,
        backend_client: Any = None
    ):
        super().__init__(config, logger)
        
        self.profile_config = profile_config or {}
        # Get process type from embedding_type in profile config
        embedding_type = self.profile_config.get("embedding_type", "frame_based")
        self.process_type = embedding_type
        self.model_name = self.profile_config.get("embedding_model", "vidore/colsmol-500m")
        self.backend_client = backend_client
        
        # Determine media type based on process type
        if self.process_type.startswith("direct_video"):
            self.media_type = MediaType.VIDEO
        else:
            self.media_type = MediaType.VIDEO_FRAME
        
        # Model and processor
        self.model = None
        self.processor = None
        self.videoprism_loader = None
        
        # Load model if needed
        if self._should_load_model():
            self._load_model()
    
    def _should_load_model(self) -> bool:
        """
        Check if model should be loaded during initialization.
        
        Frame-based processing doesn't load models during init because frames
        are processed later. Direct video processing loads models immediately
        to process video segments.
        
        Returns:
            bool: True if model should be loaded, False otherwise
        """
        return self.process_type.startswith("direct_video")
    
    def _load_model(self):
        """Load the appropriate model"""
        try:
            if "videoprism" in self.model_name.lower():
                self.videoprism_loader, _ = get_or_load_model(
                    self.model_name,
                    self.config,
                    self.logger
                )
            else:
                self.model, self.processor = get_or_load_model(
                    self.model_name,
                    self.config,
                    self.logger
                )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embeddings(
        self,
        video_data: Dict[str, Any],
        output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for a video"""
        start_time = time.time()
        video_id = video_data.get('video_id', 'unknown')
        
        self.logger.info(f"Starting embedding generation for video: {video_id}")
        self.logger.info(f"Process type: {self.process_type}")
        self.logger.info(f"Model: {self.model_name}")
        
        try:
            if self.process_type.startswith("direct_video"):
                result = self._generate_direct_video_embeddings(video_data, output_dir)
            elif self.process_type == "frame_based":
                result = self._generate_frame_based_embeddings(video_data, output_dir)
            else:
                raise ValueError(f"Unknown process type: {self.process_type}")
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self.logger.info(
                f"Completed embedding generation for {video_id} in {processing_time:.2f}s - "
                f"{result.documents_processed} processed, {result.documents_fed} fed"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return EmbeddingResult(
                video_id=video_id,
                total_documents=0,
                documents_processed=0,
                documents_fed=0,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                metadata={}
            )
        finally:
            if self.backend_client:
                self.backend_client.close()
    
    def _process_video_segment(
        self,
        video_path: Path,
        video_id: str,
        segment_idx: int,
        start_time: float,
        end_time: float,
        num_segments: int
    ) -> Optional[Document]:
        """
        Process a video segment and create a Document.
        
        This method orchestrates the segment processing:
        1. Generate raw embeddings using the model
        2. Create a Document with appropriate metadata
        3. Return Document for backend processing
        
        Args:
            video_path: Path to the video file
            video_id: Unique identifier for the video
            segment_idx: Index of this segment (0-based)
            start_time: Start time of segment in seconds
            end_time: End time of segment in seconds
            num_segments: Total number of segments in video
            
        Returns:
            Optional[Document]: Document object ready for backend processing,
                              or None if embedding generation fails
        """
        self.logger.info(
            f"Processing segment {segment_idx + 1}/{num_segments}: "
            f"{start_time:.1f}s - {end_time:.1f}s"
        )
        
        # Generate RAW embeddings
        raw_embeddings = self._generate_raw_embeddings(
            video_path, start_time, end_time
        )
        
        if raw_embeddings is None:
            self.logger.warning(f"No raw embeddings generated for segment {segment_idx}")
            return None
        
        self.logger.info(f"Generated raw embeddings for segment {segment_idx}: shape={raw_embeddings.shape if hasattr(raw_embeddings, 'shape') else 'unknown'}")
        
        # Create Document with all necessary information
        doc_id = f"{video_id}_{segment_idx}_{int(start_time)}"
        
        # Create metadata including source_id
        metadata = {
            'source_id': video_id,
            "video_title": video_id,
            "video_path": str(video_path)
        }
        
        # Create EmbeddingResult
        embedding_result = EmbeddingResult(
            embeddings=raw_embeddings,
            metadata=metadata
        )
        
        return Document(
            doc_id=doc_id,
            media_type=MediaType.VIDEO,
            embeddings=embedding_result,
            temporal_info=TemporalInfo(
                start_time=start_time,
                end_time=end_time
            ),
            segment_info=SegmentInfo(
                segment_idx=segment_idx,
                total_segments=num_segments
            ),
            metadata=metadata
        )
    
    def _generate_raw_embeddings(
        self,
        video_path: Path,
        start_time: float,
        end_time: float
    ) -> Optional[np.ndarray]:
        """
        Generate raw embeddings for a video segment.
        
        This method handles the actual model inference to produce embeddings.
        Different models have different approaches:
        - VideoPrism: Returns dict with 'embeddings_np' key
        - ColPali/ColQwen: Process video frames and return embeddings
        
        Args:
            video_path: Path to the video file
            start_time: Start time of the segment in seconds
            end_time: End time of the segment in seconds
            
        Returns:
            Optional[np.ndarray]: Raw numpy embeddings or None if generation fails
                                 Shape depends on model (patches, embedding_dim)
        """
        try:
            if self.videoprism_loader:
                self.logger.info(f"Using VideoPrism loader for segment {start_time}s-{end_time}s")
                # VideoPrism returns dict with embeddings
                result = self.videoprism_loader.process_video_segment(
                    video_path,
                    start_time,
                    end_time
                )
                self.logger.info(f"VideoPrism result: {result is not None}")
                if result:
                    self.logger.info(f"VideoPrism result keys: {list(result.keys())}")
                    # VideoPrism returns 'embeddings' not 'embeddings_np'
                    if "embeddings" in result:
                        embeddings = result["embeddings"]
                        self.logger.info(f"VideoPrism embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'no shape'}")
                        return embeddings
                    elif "embeddings_np" in result:
                        embeddings = result["embeddings_np"]
                        self.logger.info(f"VideoPrism embeddings_np shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'no shape'}")
                        return embeddings
                    else:
                        self.logger.warning(f"VideoPrism returned no embeddings. Keys: {list(result.keys())}")
                        return None
                else:
                    self.logger.warning(f"VideoPrism returned None")
                    return None
            else:
                # Other models - generate embeddings
                import tempfile
                import subprocess
                import os
                
                # Extract segment to temp file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    # Extract video segment
                    cmd = [
                        'ffmpeg', '-i', str(video_path),
                        '-ss', str(start_time),
                        '-t', str(end_time - start_time),
                        '-c', 'copy',
                        '-y', tmp_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Process with model - ColQwen needs special handling
                    if "colqwen" in self.model_name.lower():
                        # ColQwen expects PIL images, not video paths
                        import cv2
                        cap = cv2.VideoCapture(tmp_path)
                        frames = []
                        
                        # Extract frames at regular intervals
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        target_fps = self.profile_config.get('fps', 1.0)
                        
                        # Calculate frame indices to extract
                        interval = int(fps / target_fps) if fps > target_fps else 1
                        frame_indices = list(range(0, total_frames, interval))
                        
                        for idx in frame_indices[:10]:  # Limit to 10 frames per segment
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                # Convert BGR to RGB
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                pil_image = Image.fromarray(frame_rgb)
                                frames.append(pil_image)
                        
                        cap.release()
                        
                        if not frames:
                            self.logger.error("No frames extracted from video segment")
                            return None
                        
                        # Process frames with ColQwen
                        batch_inputs = self.processor.process_images(frames).to(self.model.device)
                        
                        # Generate embeddings
                        with torch.no_grad():
                            embeddings = self.model(**batch_inputs)
                        
                        # Convert to numpy and average across frames
                        embeddings_np = embeddings.cpu().numpy()
                        
                        # Average embeddings across frames to get segment embedding
                        # Shape: (num_frames, patches, dim) -> (patches, dim)
                        return embeddings_np.mean(axis=0)
                    
                    elif hasattr(self.processor, 'process_videos'):
                        batch_inputs = self.processor.process_videos([tmp_path]).to(self.model.device)
                    else:
                        batch_inputs = self.processor.process_images([tmp_path]).to(self.model.device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        embeddings = self.model(**batch_inputs)
                    
                    # Convert to numpy
                    return embeddings.cpu().numpy()
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _generate_frame_embeddings(self, frame_path: Path) -> Optional[np.ndarray]:
        """Generate embeddings for a single frame using ColPali model"""
        try:
            if not self.model or not self.processor:
                self.logger.error("Model or processor not loaded")
                return None
            
            # Load and process image
            image = Image.open(frame_path).convert("RGB")
            
            # Process image with model
            batch_inputs = self.processor.process_images([image]).to(self.model.device)
            
            with torch.no_grad():
                embeddings = self.model(**batch_inputs)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().to(torch.float32).numpy()
            
            # Handle different output shapes
            if len(embeddings_np.shape) == 3:
                # (batch, patches, dim) - squeeze batch dimension
                embeddings_np = embeddings_np.squeeze(0)
            
            return embeddings_np
            
        except Exception as e:
            self.logger.error(f"Failed to generate frame embeddings: {e}")
            return None
    
    # Implement abstract methods
    def process_segment(self, segment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass
    
    def create_document(self, segment_data: Dict[str, Any], embeddings: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def _feed_single_document(self, document: Document) -> bool:
        if self.backend_client:
            success_count, failed_ids = self.backend_client.feed(document)
            self.logger.debug(f"Fed document {document.document_id}: success={success_count}, failed={failed_ids}")
            return success_count > 0
        self.logger.warning("No backend client available")
        return False
    
    def _generate_direct_video_embeddings(
        self,
        video_data: Dict[str, Any],
        output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for direct video processing"""
        video_path = Path(video_data['video_path'])
        video_id = video_data['video_id']
        
        # Calculate segments
        duration = video_data.get('duration', 0)
        segment_duration = self.profile_config.get('segment_duration', 30.0)
        num_segments = max(1, int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0))
        
        documents_processed = 0
        documents_fed = 0
        errors = []
        
        # Process each segment
        for segment_idx in range(num_segments):
            start_time = segment_idx * segment_duration
            end_time = min((segment_idx + 1) * segment_duration, duration)
            
            try:
                # Process segment and get Document
                doc = self._process_video_segment(
                    video_path, video_id, segment_idx,
                    start_time, end_time, num_segments
                )
                
                if doc:
                    documents_processed += 1
                    
                    # Feed immediately
                    if self._feed_single_document(doc):
                        documents_fed += 1
                        self.logger.info(f"Fed segment {segment_idx} successfully")
                    else:
                        errors.append(f"Failed to feed segment {segment_idx}")
                        
            except Exception as e:
                self.logger.error(f"Error processing segment {segment_idx}: {e}")
                errors.append(f"Segment {segment_idx}: {str(e)}")
        
        return EmbeddingResult(
            video_id=video_id,
            total_documents=num_segments,
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0.0,  # Will be set by caller
            errors=errors,
            metadata={'num_segments': num_segments}
        )
    
    def _generate_frame_based_embeddings(
        self,
        video_data: Dict[str, Any],
        output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for frame-based processing"""
        video_id = video_data['video_id']
        
        # Load data from files like the old implementation
        output_dir = Path(video_data.get("output_dir", output_dir))
        
        # Load keyframes metadata
        keyframes_file = output_dir / "metadata" / f"{video_id}_keyframes.json"
        if not keyframes_file.exists():
            return EmbeddingResult(
                video_id=video_id,
                total_documents=0,
                documents_processed=0,
                documents_fed=0,
                processing_time=0.0,
                errors=["Keyframes file not found"],
                metadata={}
            )
        
        with open(keyframes_file, 'r') as f:
            metadata = json.load(f)
        
        keyframes = metadata.get("keyframes", [])
        
        # Load descriptions
        descriptions_file = output_dir / "descriptions" / f"{video_id}.json"
        descriptions = {}
        if descriptions_file.exists():
            with open(descriptions_file, 'r') as f:
                descriptions = json.load(f)
        
        # Load transcript
        transcript_file = output_dir / "transcripts" / f"{video_id}.json"
        transcript_data = {}
        if transcript_file.exists():
            with open(transcript_file, 'r') as f:
                transcript_data = json.load(f)
        
        # Extract audio transcript
        audio_transcript = ""
        if transcript_data:
            if isinstance(transcript_data, list):
                audio_transcript = " ".join([segment.get("text", "") for segment in transcript_data])
            elif isinstance(transcript_data, dict) and "segments" in transcript_data:
                audio_transcript = " ".join([segment.get("text", "") for segment in transcript_data["segments"]])
        
        documents_processed = 0
        documents_fed = 0
        errors = []
        
        # Load model if not already loaded
        if not self.model and not self.processor:
            self._load_model()
        
        # Process frames in batches
        batch_size = self.profile_config.get('batch_size', 32)
        
        for i in range(0, len(keyframes), batch_size):
            batch_keyframes = keyframes[i:i + batch_size]
            
            try:
                for keyframe in batch_keyframes:
                    frame_id = str(keyframe["frame_id"])
                    frame_path = Path(keyframe["path"])
                    timestamp = keyframe.get("timestamp", 0.0)
                    description = descriptions.get(frame_id, "")
                    
                    if not frame_path.exists():
                        self.logger.warning(f"Frame not found: {frame_path}")
                        continue
                    
                    # Generate real embeddings using the model
                    raw_embeddings = self._generate_frame_embeddings(frame_path)
                    
                    if raw_embeddings is None:
                        continue
                    
                    # Create Document for frame
                    metadata = {
                        "source_id": video_id,
                        "video_title": video_id,
                        "frame_id": int(frame_id),
                        "frame_path": str(frame_path),
                        "frame_description": description,
                        "audio_transcript": audio_transcript
                    }
                    
                    embedding_result = EmbeddingResult(
                        embeddings=raw_embeddings,
                        metadata=metadata
                    )
                    
                    doc = Document(
                        doc_id=f"{video_id}_frame_{frame_id}",
                        media_type=MediaType.VIDEO_FRAME,
                        embeddings=embedding_result,
                        temporal_info=TemporalInfo(
                            start_time=timestamp,
                            end_time=timestamp + 1.0
                        ),
                        metadata=metadata
                    )
                    
                    documents_processed += 1
                    
                    if self._feed_single_document(doc):
                        documents_fed += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing batch starting at {i}: {e}")
                errors.append(f"Batch {i}: {str(e)}")
        
        return EmbeddingResult(
            video_id=video_id,
            total_documents=len(keyframes),
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0.0,
            errors=errors,
            metadata={'num_frames': len(keyframes)}
        )