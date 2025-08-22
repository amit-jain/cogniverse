#!/usr/bin/env python3
"""
Generic Embedding Generator - Backend-agnostic implementation
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.common.models import get_or_load_model
from .document_builders import DocumentBuilderFactory, DocumentMetadata
from .embedding_processors import EmbeddingProcessor


@dataclass
class EmbeddingResult:
    video_id: str
    total_documents: int
    documents_processed: int
    documents_fed: int
    processing_time: float
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class ProcessingConfig:
    process_type: str
    model_name: str
    backend: str


class BaseEmbeddingGenerator(ABC):
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def generate_embeddings(
        self, video_data: Dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        pass


class EmbeddingGenerator(BaseEmbeddingGenerator):
    """Backend-agnostic embedding generator that passes raw embeddings"""

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        profile_config: Dict[str, Any] = None,
        backend_client: Any = None,
    ):
        super().__init__(config, logger)

        self.profile_config = profile_config or {}
        self.process_type = self.profile_config.get("process_type", "frame_based")
        self.model_name = self.profile_config.get(
            "embedding_model", "vidore/colsmol-500m"
        )
        self.backend_client = backend_client

        # Get schema name from backend client (it knows its own schema)
        self.schema_name = (
            backend_client.schema_name if backend_client else config.get("schema_name")
        )

        # Initialize components
        self.embedding_processor = EmbeddingProcessor(logger)
        self.document_builder = DocumentBuilderFactory.create_builder(self.schema_name)

        # Model and processor
        self.model = None
        self.processor = None
        self.videoprism_loader = None

        # Load model if needed
        if self._should_load_model():
            self._load_model()

    def _should_load_model(self) -> bool:
        """Check if model should be loaded during init"""
        # Load model for any direct video processing or video chunks
        return (
            self.process_type.startswith("direct_video")
            or self.process_type == "video_chunks"
        )

    def _load_model(self):
        """Load the appropriate model"""
        try:
            if "videoprism" in self.model_name.lower():
                self.videoprism_loader, _ = get_or_load_model(
                    self.model_name, self.config, self.logger
                )
            else:
                self.model, self.processor = get_or_load_model(
                    self.model_name, self.config, self.logger
                )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings(
        self, video_data: Dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for a video"""
        start_time = time.time()
        video_id = video_data.get("video_id", "unknown")

        self.logger.info(f"Starting embedding generation for video: {video_id}")
        self.logger.info(f"Process type: {self.process_type}")
        self.logger.info(f"Model: {self.model_name}")

        try:
            # Use explicit processing_type from video_data (set by pipeline)
            processing_type = video_data.get("processing_type", self.process_type)

            # Registry of processing methods - no if/elif chains
            processing_methods = {
                "single_vector": self._generate_single_vector_embeddings,
                "video_chunks": self._generate_video_chunks_embeddings,
                "direct_video": self._generate_direct_video_embeddings,
                "frame_based": self._generate_frame_based_embeddings,
            }

            # Handle direct_video variants
            if processing_type.startswith("direct_video"):
                processing_type = "direct_video"

            # Get the processing method
            process_method = processing_methods.get(processing_type)
            if not process_method:
                raise ValueError(f"Unknown processing type: {processing_type}")

            # Call the method - no if/elif needed
            result = process_method(video_data, output_dir)

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
                metadata={},
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
        num_segments: int,
    ) -> Optional[Dict[str, Any]]:
        """Process a video segment - returns raw embeddings"""
        self.logger.info(
            f"Processing segment {segment_idx + 1}/{num_segments}: "
            f"{start_time:.1f}s - {end_time:.1f}s"
        )

        # Generate embeddings based on model type
        if self.videoprism_loader:
            # Use VideoPrism (returns dict with numpy arrays)
            result = self.embedding_processor.process_videoprism_segment(
                video_path, start_time, end_time, self.videoprism_loader
            )

            if not result:
                return None

            # Extract raw numpy arrays
            embeddings = result.get("embeddings_np")

        else:
            # Use ColQwen or other models
            embeddings = (
                self.embedding_processor.generate_embeddings_from_video_segment(
                    video_path, start_time, end_time, self.model, self.processor
                )
            )

            if embeddings is None:
                return None

        # Create document with RAW embeddings
        metadata = DocumentMetadata(
            video_id=video_id,
            video_title=video_id,
            segment_idx=segment_idx,
            start_time=start_time,
            end_time=end_time,
        )

        additional_fields = {
            "segment_id": segment_idx,
            "total_segments": num_segments,
            "segment_duration": end_time - start_time,
        }

        # Pass raw numpy arrays - let backend handle conversion
        return self.document_builder.build_document(
            metadata,
            {
                "embeddings": embeddings,  # Raw numpy array
            },
            additional_fields,
        )

    def _generate_direct_video_embeddings(
        self, video_data: Dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for direct video processing"""
        video_id = video_data.get("video_id", "unknown")

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
                metadata={},
            )

        # Get video info
        video_info = self._get_video_info(video_path)
        duration = video_info["duration"]

        # Calculate segments
        segment_duration = self.profile_config.get("model_specific", {}).get(
            "segment_duration", 30.0
        )
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
                    num_segments,
                )

                if doc:
                    documents_processed += 1

                    # Feed immediately
                    if self.backend_client and self.backend_client.feed_document(doc):
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
                "process_type": self.process_type,
                "model": self.model_name,
            },
        )

    def _generate_single_vector_embeddings(
        self, video_data: Dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for single vector processing (pre-segmented data)"""
        video_id = video_data.get("video_id", "unknown")
        segments = video_data.get("segments", [])
        document_structure = video_data.get("document_structure", {})

        self.logger.info(
            f"Processing {len(segments)} segments with {document_structure.get('type', 'unknown')} structure"
        )

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
                metadata={},
            )

        # Process all segments and collect embeddings
        embeddings = []
        errors = []

        for segment in segments:
            try:
                # Generate embeddings based on model type
                if self.videoprism_loader:
                    # Use VideoPrism
                    result = self.embedding_processor.process_videoprism_segment(
                        video_path,
                        segment.start_time,
                        segment.end_time,
                        self.videoprism_loader,
                    )

                    if result:
                        embeddings.append(result.get("embeddings_np"))
                    else:
                        embeddings.append(None)
                        errors.append(f"Failed to process segment {segment.segment_id}")
                else:
                    # Use ColQwen or other models
                    embedding = (
                        self.embedding_processor.generate_embeddings_from_video_segment(
                            video_path,
                            segment.start_time,
                            segment.end_time,
                            self.model,
                            self.processor,
                        )
                    )
                    embeddings.append(embedding)

            except Exception as e:
                self.logger.error(f"Error processing segment {segment.segment_id}: {e}")
                embeddings.append(None)
                errors.append(f"Segment {segment.segment_id}: {str(e)}")

        # Filter out None embeddings
        valid_embeddings = [e for e in embeddings if e is not None]

        # Create documents using SingleVectorDocumentBuilder
        from .single_vector_document_builder import SingleVectorDocumentBuilder

        # Use storage mode from video_data (set by pipeline) or fallback to document structure
        storage_mode = video_data.get("storage_mode")
        if not storage_mode:
            storage_mode = (
                "single_doc"
                if document_structure.get("type") == "single_document"
                else "multi_doc"
            )

        builder = SingleVectorDocumentBuilder(
            schema_name=self.schema_name, storage_mode=storage_mode
        )

        try:
            documents = builder.build_documents(
                video_data=video_data,
                embeddings=valid_embeddings,
                additional_metadata={
                    "title": video_data.get("video_title", video_id),
                    "keywords": video_data.get("keywords", ""),
                    "url": video_data.get("url", ""),
                },
            )

            # Feed documents
            documents_fed = 0
            for doc in documents:
                if self.backend_client and self.backend_client.feed_document(doc):
                    documents_fed += 1
                    self.logger.info(f"✅ Successfully fed document for {video_id}")
                else:
                    self.logger.warning(f"⚠️ Failed to feed document for {video_id}")

            return EmbeddingResult(
                video_id=video_id,
                total_documents=len(documents),
                documents_processed=len(valid_embeddings),
                documents_fed=documents_fed,
                processing_time=0,  # Will be set by caller
                errors=errors,
                metadata={
                    "process_type": "single_vector",
                    "storage_mode": storage_mode,
                    "num_segments": len(segments),
                    "model": self.model_name,
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating documents: {e}")
            errors.append(f"Document creation: {str(e)}")
            return EmbeddingResult(
                video_id=video_id,
                total_documents=0,
                documents_processed=len(valid_embeddings),
                documents_fed=0,
                processing_time=0,
                errors=errors,
                metadata={},
            )

    def _generate_video_chunks_embeddings(
        self, video_data: Dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for video chunks - all segments in one document"""
        video_id = video_data.get("video_id", "unknown")

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
                metadata={},
            )

        # Get video info
        video_info = self._get_video_info(video_path)
        duration = video_info["duration"]

        # Calculate segments
        segment_duration = self.profile_config.get("model_specific", {}).get(
            "segment_duration", 30.0
        )
        num_segments = max(1, int(np.ceil(duration / segment_duration)))

        self.logger.info(
            f"Processing video with {num_segments} segments of {segment_duration}s for chunks document"
        )

        # Process all segments and collect embeddings
        segments = []
        embeddings_list = []
        errors = []

        for segment_idx in range(num_segments):
            start_time = segment_idx * segment_duration
            end_time = min((segment_idx + 1) * segment_duration, duration)

            segments.append(
                {
                    "segment_id": segment_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

            try:
                # Process segment to get embeddings
                if self.videoprism_loader:
                    result = self.embedding_processor.process_videoprism_segment(
                        video_path, start_time, end_time, self.videoprism_loader
                    )
                    if result:
                        embeddings_list.append(result.get("embeddings_np"))
                    else:
                        embeddings_list.append(None)
                else:
                    embeddings = (
                        self.embedding_processor.generate_embeddings_from_video_segment(
                            video_path, start_time, end_time, self.model, self.processor
                        )
                    )
                    embeddings_list.append(embeddings)

            except Exception as e:
                self.logger.error(f"Error processing segment {segment_idx}: {e}")
                embeddings_list.append(None)
                errors.append(f"Segment {segment_idx}: {str(e)}")

        # Filter out None embeddings
        valid_embeddings = [e for e in embeddings_list if e is not None]

        # Build individual documents for each segment (all schemas now use per-document structure)
        documents = []
        for segment_idx, (segment, embeddings) in enumerate(
            zip(segments, embeddings_list)
        ):
            if embeddings is None:
                continue

            # Create metadata for this segment
            from .document_builders import DocumentMetadata

            metadata = DocumentMetadata(
                video_id=video_id,
                video_title=video_data.get("video_title", video_id),
                segment_idx=segment_idx,
                start_time=segment["start_time"],
                end_time=segment["end_time"],
            )

            # Prepare additional fields
            additional_fields = {}
            if "audio_transcript" in segment:
                additional_fields["audio_transcript"] = segment["audio_transcript"]
            if "segment_description" in segment:
                additional_fields["segment_description"] = segment[
                    "segment_description"
                ]
            additional_fields["total_segments"] = num_segments
            additional_fields["segment_duration"] = (
                segment["end_time"] - segment["start_time"]
            )

            # Build document
            document = self.document_builder.build_document(
                metadata=metadata,
                embeddings=embeddings,
                additional_fields=additional_fields,
            )
            documents.append(document)

        # Feed all documents
        documents_fed = 0
        if self.backend_client:
            for document in documents:
                if self.backend_client.feed_document(document):
                    documents_fed += 1

        self.logger.info(
            f"✅ Successfully fed {documents_fed}/{len(documents)} documents for {video_id}"
        )

        return EmbeddingResult(
            video_id=video_id,
            total_documents=len(documents),
            documents_processed=len(documents),
            documents_fed=documents_fed,
            processing_time=0,  # Will be set by caller
            errors=errors,
            metadata={
                "video_duration": duration,
                "segment_duration": segment_duration,
                "num_segments": num_segments,
                "valid_segments": len(valid_embeddings),
                "model": self.model_name,
            },
        )

    def _generate_frame_based_embeddings(
        self, video_data: Dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        """Generate embeddings for frame-based processing"""
        # Similar implementation but for frames
        # Not shown for brevity - would follow same pattern
        pass

    def process_segment(self, segment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single segment - implements abstract method"""
        # Handled by specific methods above
        pass

    def create_document(
        self, segment_data: Dict[str, Any], embeddings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create document - implements abstract method"""
        # Handled by document builder
        pass

    def _feed_single_document(self, document: Dict[str, Any]) -> bool:
        """Feed single document - implements abstract method"""
        if self.backend_client:
            return self.backend_client.feed_document(document)
        return False

    # Helper methods (same as before)
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

            return {"fps": fps, "total_frames": total_frames, "duration": duration}
        finally:
            cap.release()
