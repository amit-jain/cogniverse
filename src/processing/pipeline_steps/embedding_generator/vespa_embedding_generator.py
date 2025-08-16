#!/usr/bin/env python3
"""
Vespa Embedding Generator - Implementation for Vespa backend
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
import logging
import json
import time
import numpy as np

from .embedding_generator import BaseEmbeddingGenerator, EmbeddingResult, ProcessingConfig
from src.models import get_or_load_model
from .document_builders import DocumentBuilderFactory, DocumentMetadata
from .embedding_processors import EmbeddingProcessor
from .vespa_pyvespa_client import VespaPyClient


class VespaEmbeddingGenerator(BaseEmbeddingGenerator):
    """Embedding generator for Vespa backend"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        embedding_type: str = "frame_based"
    ):
        super().__init__(config, logger)
        
        self.embedding_type = embedding_type
        self.schema_name = config.get("schema_name")
        if not self.schema_name:
            raise ValueError("schema_name is required in config")
        self.model_name = self._get_model_name()
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor(logger)
        self.document_builder = DocumentBuilderFactory.create_builder(self.schema_name)
        self.backend_client = None
        
        # Model and processor
        self.model = None
        self.processor = None
        self.videoprism_loader = None
        
        # Load model if needed
        if self._should_load_model():
            self._load_model()
    
    def _get_model_name(self) -> str:
        """Get the model name from config"""
        # Check active profile first
        active_profile = self.config.get("active_profile")
        if active_profile:
            profiles = self.config.get("video_processing_profiles", {})
            if active_profile in profiles:
                return profiles[active_profile].get("embedding_model", "vidore/colsmol-500m")
        
        # Fallback to colpali_model
        return self.config.get("colpali_model", "vidore/colsmol-500m")
    
    def _should_load_model(self) -> bool:
        """Check if model should be loaded during init"""
        return self.embedding_type in ["direct_video", "direct_video_segment", "direct_video_frame"]
    
    def _load_model(self):
        """Load the appropriate model"""
        try:
            if "videoprism" in self.model_name.lower():
                # VideoPrism returns loader as model
                self.videoprism_loader, _ = get_or_load_model(
                    self.model_name,
                    self.config,
                    self.logger
                )
            else:
                # Other models return model and processor
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
        self.logger.info(f"Embedding type: {self.embedding_type}")
        
        # Initialize Vespa client using pyvespa
        self.backend_client = VespaPyClient(
            config=self.config,
            schema_name=self.schema_name,
            logger=self.logger
        )
        
        try:
            if self.embedding_type in ["direct_video", "direct_video_segment", "direct_video_frame"]:
                result = self._generate_direct_video_embeddings(video_data, output_dir)
            else:
                result = self._generate_frame_based_embeddings(video_data, output_dir)
            
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
    
    def _generate_frame_based_embeddings(
        self,
        video_data: Dict[str, Any],
        output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for frame-based processing"""
        video_id = video_data.get('video_id', 'unknown')
        
        # Load data files
        keyframes = self._load_keyframes(video_id, output_dir)
        descriptions = self._load_descriptions(video_id, output_dir)
        transcript = self._load_transcript(video_id, output_dir)
        
        if not keyframes:
            return EmbeddingResult(
                video_id=video_id,
                total_documents=0,
                documents_processed=0,
                documents_fed=0,
                processing_time=0,
                errors=["No keyframes found"],
                metadata={}
            )
        
        # Check progress
        progress = self.load_progress(video_id, output_dir)
        start_idx = progress.get("last_processed_idx", 0) if progress else 0
        
        # Process frames
        total_frames = len(keyframes)
        documents_processed = start_idx
        documents_fed = 0
        errors = []
        
        # Process in batches
        batch_size = self.processing_config.batch_size
        
        for batch_start in range(start_idx, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frames = keyframes[batch_start:batch_end]
            
            self.logger.info(
                f"Processing batch: frames {batch_start+1}-{batch_end} of {total_frames}"
            )
            
            # Process batch
            batch_docs = []
            for i, frame in enumerate(batch_frames):
                frame_idx = batch_start + i
                
                try:
                    # Process frame
                    doc = self._process_frame(
                        frame,
                        frame_idx,
                        video_id,
                        descriptions,
                        transcript
                    )
                    
                    if doc:
                        batch_docs.append(doc)
                        documents_processed += 1
                        
                        # Feed immediately
                        if self.backend_client.feed_document(doc):
                            documents_fed += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx}: {e}")
                    errors.append(f"Frame {frame_idx}: {str(e)}")
            
            # Save progress
            if documents_processed % self.processing_config.save_progress_interval == 0:
                self.save_progress(video_id, output_dir, {
                    "last_processed_idx": batch_end,
                    "documents_processed": documents_processed,
                    "documents_fed": documents_fed
                })
        
        return EmbeddingResult(
            video_id=video_id,
            total_documents=total_frames,
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0,  # Will be set by caller
            errors=errors,
            metadata={
                "keyframes_count": total_frames,
                "embedding_type": self.embedding_type
            }
        )
    
    def _generate_direct_video_embeddings(
        self,
        video_data: Dict[str, Any],
        output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for direct video processing"""
        video_id = video_data.get('video_id', 'unknown')
        
        # Get video path
        video_path = self._get_video_path(video_data)
        if not video_path:
            return EmbeddingResult(
                video_id=video_id,
                total_documents=0,
                documents_processed=0,
                documents_fed=0,
                processing_time=0,
                errors=["Video file not found"],
                metadata={}
            )
        
        # Get video info
        video_info = self._get_video_info(video_path)
        duration = video_info["duration"]
        
        # Calculate segments
        segment_duration = self.config.get("model_specific", {}).get("segment_duration", 30.0)
        num_segments = max(1, int(np.ceil(duration / segment_duration)))
        
        self.logger.info(
            f"Processing video with {num_segments} segments of {segment_duration}s"
        )
        
        # Process segments
        documents_processed = 0
        documents_fed = 0
        errors = []
        
        for segment_idx in range(num_segments):
            start_time = segment_idx * segment_duration
            end_time = min((segment_idx + 1) * segment_duration, duration)
            
            try:
                # Process segment
                doc = self._process_video_segment(
                    video_path,
                    video_id,
                    segment_idx,
                    start_time,
                    end_time,
                    num_segments
                )
                
                if doc:
                    documents_processed += 1
                    
                    # Feed immediately
                    if self.backend_client.feed_document(doc):
                        documents_fed += 1
                        self.logger.info(f"✅ Successfully fed segment {segment_idx}")
                    else:
                        self.logger.warning(f"⚠️ Failed to feed segment {segment_idx}")
                
            except Exception as e:
                self.logger.error(f"Error processing segment {segment_idx}: {e}")
                errors.append(f"Segment {segment_idx}: {str(e)}")
        
        return EmbeddingResult(
            video_id=video_id,
            total_documents=num_segments,
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0,  # Will be set by caller
            errors=errors,
            metadata={
                "video_duration": duration,
                "segment_duration": segment_duration,
                "num_segments": num_segments,
                "embedding_type": self.embedding_type
            }
        )
    
    def _process_frame(
        self,
        frame: Dict[str, Any],
        frame_idx: int,
        video_id: str,
        descriptions: Dict[str, str],
        transcript: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single frame"""
        frame_id = str(frame["frame_id"])
        frame_path = Path(frame["path"])
        
        if not frame_path.exists():
            self.logger.warning(f"Frame not found: {frame_path}")
            return None
        
        # Generate embeddings
        embeddings_np = self.embedding_processor.generate_embeddings_from_image(
            frame_path,
            self.model,
            self.processor
        )
        
        if embeddings_np is None:
            return None
        
        # Convert embeddings
        float_embeddings = self.embedding_processor.convert_to_float_embeddings(embeddings_np)
        binary_embeddings = self.embedding_processor.convert_to_binary_embeddings(embeddings_np)
        
        # Create document
        metadata = DocumentMetadata(
            video_id=video_id,
            video_title=video_id,
            segment_idx=int(frame_id),
            start_time=float(frame.get("timestamp", 0)),
            end_time=float(frame.get("timestamp", 0) + 1.0)
        )
        
        additional_fields = {
            "frame_description": descriptions.get(frame_id, ""),
            "audio_transcript": transcript
        }
        
        return self.document_builder.build_document(
            metadata,
            {
                "float_embeddings": float_embeddings,
                "binary_embeddings": binary_embeddings
            },
            additional_fields
        )
    
    def _process_video_segment(
        self,
        video_path: Path,
        video_id: str,
        segment_idx: int,
        start_time: float,
        end_time: float,
        num_segments: int
    ) -> Optional[Dict[str, Any]]:
        """Process a video segment"""
        self.logger.info(
            f"Processing segment {segment_idx + 1}/{num_segments}: "
            f"{start_time:.1f}s - {end_time:.1f}s"
        )
        
        # Generate embeddings based on model type
        if self.videoprism_loader:
            # Use VideoPrism
            result = self.embedding_processor.process_videoprism_segment(
                video_path,
                start_time,
                end_time,
                self.videoprism_loader
            )
            
            if not result:
                return None
            
            float_embeddings = result.get("float_embeddings", {})
            binary_embeddings = result.get("binary_embeddings", {})
            
        else:
            # Use ColQwen or other models
            embeddings_np = self.embedding_processor.generate_embeddings_from_video_segment(
                video_path,
                start_time,
                end_time,
                self.model,
                self.processor
            )
            
            if embeddings_np is None:
                return None
            
            # Convert embeddings
            float_embeddings = self.embedding_processor.convert_to_float_embeddings(embeddings_np)
            binary_embeddings = self.embedding_processor.convert_to_binary_embeddings(embeddings_np)
        
        # Create document
        metadata = DocumentMetadata(
            video_id=video_id,
            video_title=video_id,
            segment_idx=segment_idx,
            start_time=start_time,
            end_time=end_time
        )
        
        additional_fields = {
            "segment_id": segment_idx,
            "total_segments": num_segments,
            "segment_duration": end_time - start_time
        }
        
        return self.document_builder.build_document(
            metadata,
            {
                "float_embeddings": float_embeddings,
                "binary_embeddings": binary_embeddings
            },
            additional_fields
        )
    
    def process_segment(self, segment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single segment - implements abstract method"""
        # This is handled by specific methods above
        pass
    
    def create_document(
        self,
        segment_data: Dict[str, Any],
        embeddings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create document - implements abstract method"""
        # This is handled by document builder
        pass
    
    def _feed_single_document(self, document: Dict[str, Any]) -> bool:
        """Feed single document - implements abstract method"""
        if self.backend_client:
            return self.backend_client.feed_document(document)
        return False
    
    # Helper methods
    
    def _load_keyframes(self, video_id: str, output_dir: Path) -> List[Dict[str, Any]]:
        """Load keyframes from file"""
        keyframes_file = output_dir / "metadata" / f"{video_id}_keyframes.json"
        if keyframes_file.exists():
            with open(keyframes_file, 'r') as f:
                data = json.load(f)
                return data.get("keyframes", [])
        return []
    
    def _load_descriptions(self, video_id: str, output_dir: Path) -> Dict[str, str]:
        """Load descriptions from file"""
        descriptions_file = output_dir / "descriptions" / f"{video_id}.json"
        if descriptions_file.exists():
            with open(descriptions_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_transcript(self, video_id: str, output_dir: Path) -> str:
        """Load transcript from file"""
        transcript_file = output_dir / "transcripts" / f"{video_id}.json"
        if transcript_file.exists():
            with open(transcript_file, 'r') as f:
                data = json.load(f)
                # Handle different transcript formats
                if isinstance(data, list):
                    return " ".join([seg.get("text", "") for seg in data])
                elif isinstance(data, dict) and "segments" in data:
                    return " ".join([seg.get("text", "") for seg in data["segments"]])
        return ""
    
    def _get_video_path(self, video_data: Dict[str, Any]) -> Optional[Path]:
        """Get video file path"""
        video_id = video_data.get("video_id", "")
        video_path = Path(video_data.get("video_path", ""))
        
        if video_path.exists():
            return video_path
        
        # Try to find in video directory
        video_dir = Path(self.config.get("video_data_dir", "data/videos"))
        for pattern in [f"{video_id}.*", f"*/{video_id}.*"]:
            matches = list(video_dir.glob(pattern))
            if matches:
                return matches[0]
        
        return None
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information"""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            return {
                "fps": fps,
                "total_frames": total_frames,
                "duration": duration
            }
        finally:
            cap.release()