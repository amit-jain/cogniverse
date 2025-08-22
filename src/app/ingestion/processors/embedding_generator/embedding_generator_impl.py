#!/usr/bin/env python3
"""
Generic Embedding Generator - Unified processing for all segment types.

This module provides a simplified, strategy-driven embedding generator that handles
all video processing profiles through a single generic method.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time
import numpy as np
import torch
from PIL import Image

from .embedding_generator import EmbeddingGenerator, EmbeddingResult
from src.common.document import Document, ContentType, ProcessingStatus


class EmbeddingGeneratorImpl(EmbeddingGenerator):
    """
    Embedding generator that processes all segment types uniformly.

    Key principle: All processing follows the same pattern:
    1. Iterate over segments (frames/chunks/windows)
    2. Generate embeddings for each segment
    3. Create Document objects
    4. Feed to backend

    The only differences are:
    - How segments are defined (frames vs chunks)
    - Whether to create one doc per segment or one doc for all
    - What metadata to include
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        backend_client: Any = None,
    ):
        super().__init__(config, logger)

        self.profile_config = config
        self.model_name = self.profile_config.get(
            "embedding_model", "vidore/colsmol-500m"
        )
        self.backend_client = backend_client
        self.schema_name = self.profile_config.get("schema_name")

        # Storage mode determines if we create one doc per segment or one doc total
        self.storage_mode = self.profile_config.get("storage_mode", "multi_doc")

        # Model and processor
        self.model = None
        self.processor = None
        self.videoprism_loader = None

        # Load model if needed
        if self._should_load_model():
            self._load_model()

    def _should_load_model(self) -> bool:
        """Check if model should be loaded during initialization."""
        # Frame-based processing loads model later when processing frames
        processing_type = self.profile_config.get("embedding_type", "frame_based")
        return processing_type != "frame_based"

    def _load_model(self):
        """Load the embedding model based on configuration."""
        from src.common.models import get_or_load_model

        try:
            if "videoprism" in self.model_name.lower():
                self.videoprism_loader, _ = get_or_load_model(
                    self.model_name, self.profile_config, self.logger
                )
            else:
                self.model, self.processor = get_or_load_model(
                    self.model_name, self.profile_config, self.logger
                )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings(
        self, video_data: Dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        """
        Generate embeddings using a unified approach for all segment types.

        The video_data should contain:
        - segments: List of segments to process (frames/chunks/windows)
        - segment_type: Type of segments (frame/chunk/window)
        - storage_mode: How to store (multi_doc/single_doc)
        - metadata: Additional metadata for documents
        """
        start_time = time.time()
        video_id = video_data.get("video_id", "unknown")

        # Extract segments - they could be under different keys
        segments = self._extract_segments(video_data)
        if not segments:
            return EmbeddingResult(
                video_id=video_id,
                total_documents=0,
                documents_processed=0,
                documents_fed=0,
                processing_time=time.time() - start_time,
                errors=["No segments found in video_data"],
                metadata={},
            )

        self.logger.info(
            f"📊 Processing {len(segments)} segments for {video_id} (schema: {self.schema_name})"
        )

        # Determine storage mode
        storage_mode = video_data.get("storage_mode", self.storage_mode)

        if storage_mode == "single_doc":
            # Process all segments and create one document
            result = self._process_single_document(video_data, segments)
        else:
            # Process each segment as a separate document
            result = self._process_multi_documents(video_data, segments)

        result.processing_time = time.time() - start_time
        self.logger.info(
            f"Completed {video_id}: {result.documents_processed} processed, "
            f"{result.documents_fed} fed in {result.processing_time:.2f}s"
        )

        return result

    def _extract_segments(self, video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract segments from video_data, handling different key names.

        Segments could be under:
        - 'segments' (generic)
        - 'keyframes' (frame-based)
        - 'frames' (frame-based alternative)
        - 'chunks' (chunk-based)
        - 'video_chunks' (chunk-based alternative)
        """
        # Try different keys in order of preference
        for key in ["segments", "keyframes", "frames", "chunks", "video_chunks"]:
            if key in video_data:
                segments = video_data[key]
                # Ensure it's a list
                if isinstance(segments, dict):
                    # Extract the actual segment list from dict structure
                    if "keyframes" in segments:
                        return segments["keyframes"]
                    elif "chunks" in segments:
                        return segments["chunks"]
                    elif "segments" in segments:
                        return segments["segments"]
                return segments if isinstance(segments, list) else []

        # Check for single_vector_processing structure
        if "single_vector_processing" in video_data:
            sv_data = video_data["single_vector_processing"]
            if "segments" in sv_data:
                return sv_data["segments"]

        return []

    def _process_multi_documents(
        self, video_data: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> EmbeddingResult:
        """Process segments as individual documents."""
        video_id = video_data["video_id"]
        video_path = Path(video_data.get("video_path", ""))

        documents_processed = 0
        documents_fed = 0
        errors = []

        # Get additional data
        transcript_data = video_data.get("transcript", {})
        descriptions = video_data.get("descriptions", {})

        # Extract transcript text
        transcript_text = self._extract_transcript_text(transcript_data)

        for idx, segment in enumerate(segments):
            try:
                # Generate embeddings for this segment
                self.logger.debug(
                    f"  Processing segment {idx}/{len(segments)}: {segment.get('start_time', 0):.1f}s - {segment.get('end_time', 0):.1f}s"
                )
                embeddings = self._generate_segment_embeddings(
                    segment, video_path, video_data
                )

                if embeddings is None:
                    self.logger.debug(
                        f"    ⚠️ No embeddings generated for segment {idx}"
                    )
                    continue
                else:
                    self.logger.debug(
                        f"    ✅ Generated embeddings shape: {embeddings.shape}"
                    )

                # Create document for this segment
                doc = self._create_segment_document(
                    video_id=video_id,
                    segment=segment,
                    segment_idx=idx,
                    total_segments=len(segments),
                    embeddings=embeddings,
                    transcript=transcript_text,
                    description=descriptions.get(str(idx), ""),
                )

                documents_processed += 1

                # Feed to backend
                if self._feed_document(doc):
                    documents_fed += 1

            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                errors.append(f"Segment {idx}: {str(e)}")

        return EmbeddingResult(
            video_id=video_id,
            total_documents=len(segments),
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0,  # Set by caller
            errors=errors,
            metadata={"num_segments": len(segments)},
        )

    def _process_single_document(
        self, video_data: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> EmbeddingResult:
        """Process all segments and create a single document."""
        video_id = video_data["video_id"]
        video_path = Path(video_data.get("video_path", ""))

        # Collect embeddings from all segments
        all_embeddings = []
        errors = []

        for idx, segment in enumerate(segments):
            try:
                self.logger.debug(
                    f"  Processing segment {idx}/{len(segments)} for single doc"
                )
                embeddings = self._generate_segment_embeddings(
                    segment, video_path, video_data
                )
                if embeddings is not None:
                    all_embeddings.append(embeddings)
                    self.logger.debug(
                        f"    ✅ Added embeddings shape: {embeddings.shape}"
                    )
            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                errors.append(f"Segment {idx}: {str(e)}")

        if not all_embeddings:
            return EmbeddingResult(
                video_id=video_id,
                total_documents=1,
                documents_processed=0,
                documents_fed=0,
                processing_time=0,
                errors=["No embeddings generated"],
                metadata={},
            )

        # Stack embeddings
        combined_embeddings = (
            np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        )

        # Create single document
        doc = self._create_combined_document(
            video_id=video_id,
            embeddings=combined_embeddings,
            segments=segments,
            video_data=video_data,
        )

        documents_fed = 1 if self._feed_document(doc) else 0

        return EmbeddingResult(
            video_id=video_id,
            total_documents=1,
            documents_processed=1,
            documents_fed=documents_fed,
            processing_time=0,
            errors=errors,
            metadata={"num_segments": len(segments)},
        )

    def _generate_segment_embeddings(
        self, segment: Dict[str, Any], video_path: Path, video_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Generate embeddings for a single segment.

        Handles different segment types:
        - Frame: Load image and process
        - Chunk: Process video file or segment
        - Window: Process time range
        """
        # All segments are now dictionaries
        # Check for chunk segments first (video files)
        if "chunk_path" in segment:
            # Pre-extracted chunk
            chunk_path = Path(segment["chunk_path"])
            return self._generate_chunk_embeddings(chunk_path)
        elif "path" in segment:
            # Check if path points to video chunk or image frame
            path = Path(segment["path"])
            if path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                # Video chunk
                return self._generate_chunk_embeddings(path)
            else:
                # Image frame
                return self._generate_frame_embeddings(path)
        elif "frame_path" in segment:
            # Frame segment
            frame_path = Path(segment["frame_path"])
            return self._generate_frame_embeddings(frame_path)
        elif "start_time" in segment and "end_time" in segment:
            # Time-based segment (including single-vector segments)
            return self._generate_time_segment_embeddings(
                video_path, segment["start_time"], segment["end_time"]
            )
        else:
            self.logger.warning(f"Unknown segment type: {segment.keys()}")
            return None

    def _generate_frame_embeddings(self, frame_path: Path) -> Optional[np.ndarray]:
        """Generate embeddings for a single frame."""
        if not self.model:
            self._load_model()

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

            self.logger.info(
                f"    🖼️ Generated embeddings for frame {frame_path.name}: shape={embeddings_np.shape}"
            )
            return embeddings_np

        except Exception as e:
            self.logger.error(f"Error generating frame embeddings: {e}")
            return None

    def _generate_chunk_embeddings(self, chunk_path: Path) -> Optional[np.ndarray]:
        """Generate embeddings for a video chunk."""
        try:
            if "colqwen" in self.model_name.lower():
                # ColQwen processes video chunks
                import cv2

                cap = cv2.VideoCapture(str(chunk_path))
                frames = []

                # Extract frames at regular intervals
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                target_fps = self.profile_config.get("fps", 1.0)

                # Calculate frame indices to extract
                interval = int(fps / target_fps) if fps > target_fps else 1
                frame_indices = list(range(0, total_frames, interval))[
                    :10
                ]  # Limit to 10 frames

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        frames.append(pil_image)

                cap.release()

                if not frames:
                    self.logger.error("No frames extracted from chunk")
                    return None

                # Process frames with ColQwen
                batch_inputs = self.processor.process_images(frames).to(
                    self.model.device
                )

                with torch.no_grad():
                    embeddings = self.model(**batch_inputs)

                # Convert to numpy and average across frames
                embeddings_np = embeddings.cpu().numpy()
                return embeddings_np.mean(axis=0)  # Average across frames

            elif self.videoprism_loader:
                # VideoPrism processing
                import subprocess

                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(chunk_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                duration = (
                    float(result.stdout.strip()) if result.returncode == 0 else 30.0
                )

                result = self.videoprism_loader.process_video_segment(
                    chunk_path, 0, duration
                )

                if result:
                    return result.get("embeddings_np", result.get("embeddings"))
                return None

            else:
                # Other models
                if hasattr(self.processor, "process_videos"):
                    batch_inputs = self.processor.process_videos([str(chunk_path)]).to(
                        self.model.device
                    )
                else:
                    batch_inputs = self.processor.process_images([str(chunk_path)]).to(
                        self.model.device
                    )

                with torch.no_grad():
                    embeddings = self.model(**batch_inputs)

                return embeddings.cpu().numpy()

        except Exception as e:
            self.logger.error(f"Error generating chunk embeddings: {e}")
            return None

    def _generate_time_segment_embeddings(
        self, video_path: Path, start_time: float, end_time: float
    ) -> Optional[np.ndarray]:
        """Generate embeddings for a time segment."""
        try:
            if self.videoprism_loader:
                # Use VideoPrism for time-based segments
                self.logger.debug(
                    f"    Time segment: start={start_time:.1f}s, end={end_time:.1f}s"
                )
                result = self.videoprism_loader.process_video_segment(
                    video_path, start_time, end_time
                )
                if result:
                    embeddings = result.get("embeddings_np", result.get("embeddings"))
                    if embeddings is not None:
                        self.logger.debug(
                            f"    Generated time segment embeddings: shape={embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}"
                        )
                    return embeddings
            else:
                # Extract frames from time segment for other models
                import cv2

                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Set to start time
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                frames = []
                target_fps = self.profile_config.get("fps", 1.0)
                interval = int(fps / target_fps) if fps > target_fps else 1

                for frame_idx in range(
                    start_frame, min(end_frame, start_frame + 10 * interval), interval
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))

                cap.release()

                if not frames:
                    return None

                # Process frames
                batch_inputs = self.processor.process_images(frames).to(
                    self.model.device
                )
                with torch.no_grad():
                    embeddings = self.model(**batch_inputs)

                embeddings_np = embeddings.cpu().numpy()
                return embeddings_np.mean(axis=0) if len(frames) > 1 else embeddings_np

        except Exception as e:
            self.logger.error(f"Error generating time segment embeddings: {e}")
            return None

    def _create_segment_document(
        self,
        video_id: str,
        segment: Dict[str, Any],
        segment_idx: int,
        total_segments: int,
        embeddings: np.ndarray,
        transcript: str = "",
        description: str = "",
    ) -> Document:
        """Create a Document for a single segment."""
        # All segments use generic Document now - no MediaType needed
        # Extract timing info
        start_time = segment.get("timestamp", segment.get("start_time", 0.0))
        end_time = segment.get("end_time", start_time + 1.0)

        # Create document
        doc_id = f"{video_id}_seg_{segment_idx}"

        document = Document(
            id=doc_id,
            content_type=ContentType.VIDEO,
            content_id=video_id,
            status=ProcessingStatus.COMPLETED,
        )

        # Add embeddings
        document.add_embedding("embedding", embeddings, {"type": "float", "raw": True})

        # Add temporal metadata
        document.add_metadata("start_time", start_time)
        document.add_metadata("end_time", end_time)
        document.add_metadata("segment_index", segment_idx)
        document.add_metadata("total_segments", total_segments)

        # Add transcript if available
        if transcript:
            document.add_metadata("audio_transcript", transcript)

        # Add basic metadata
        document.add_metadata("video_id", video_id)
        document.add_metadata("video_title", video_id)

        # Add description if available
        if description:
            document.add_metadata("description", description)

        return document

    def _create_combined_document(
        self,
        video_id: str,
        embeddings: np.ndarray,
        segments: List[Dict[str, Any]],
        video_data: Dict[str, Any],
    ) -> Document:
        """Create a single Document containing all segments."""
        # Extract timing info from segments (all are dicts now)
        start_times = [s.get("start_time", 0.0) for s in segments]
        end_times = [s.get("end_time", 0.0) for s in segments]

        # Get transcript
        transcript = self._extract_transcript_text(video_data.get("transcript", {}))

        document = Document(
            id=video_id,
            content_type=ContentType.VIDEO,
            content_id=video_id,
            status=ProcessingStatus.COMPLETED,
        )

        # Add embeddings
        document.add_embedding(
            "embedding",
            embeddings,
            {"type": "float", "raw": True, "storage_mode": "combined"},
        )

        # Add temporal metadata
        if start_times and end_times:
            document.add_metadata("start_time", min(start_times))
            document.add_metadata("end_time", max(end_times))

        document.add_metadata("total_segments", len(segments))

        # Add transcript if available
        if transcript:
            document.add_metadata("audio_transcript", transcript)

        # Add basic metadata
        document.add_metadata("video_id", video_id)
        document.add_metadata("video_title", video_id)

        return document

    def _extract_transcript_text(self, transcript_data: Any) -> str:
        """Extract text from various transcript formats."""
        if not transcript_data:
            return ""

        if isinstance(transcript_data, str):
            return transcript_data
        elif isinstance(transcript_data, list):
            return " ".join([s.get("text", "") for s in transcript_data])
        elif isinstance(transcript_data, dict):
            if "segments" in transcript_data:
                return " ".join(
                    [s.get("text", "") for s in transcript_data["segments"]]
                )
            elif "full_text" in transcript_data:
                return transcript_data["full_text"]
            elif "text" in transcript_data:
                return transcript_data["text"]

        return ""

    def _feed_document(self, document: Document) -> bool:
        """Feed document to backend."""
        if self.backend_client:
            result = self.backend_client.ingest_documents([document], self.schema_name)
            return result.get("success_count", 0) > 0
        return False
