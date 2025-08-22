#!/usr/bin/env python3
"""
New strategy implementations extending BaseStrategy.

These strategies work with the pluggable ProcessorManager.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .processor_base import BaseStrategy


class FrameSegmentationStrategy(BaseStrategy):
    """Extract frames from video (e.g., for ColPali)."""

    def __init__(
        self, fps: float = 1.0, threshold: float = 0.999, max_frames: int = 3000
    ):
        self.fps = fps
        self.threshold = threshold
        self.max_frames = max_frames

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Frame segmentation requires keyframe processor."""
        return {
            "keyframe": {
                "fps": self.fps,
                "threshold": self.threshold,
                "max_frames": self.max_frames,
            }
        }


class ChunkSegmentationStrategy(BaseStrategy):
    """Extract video chunks (e.g., for ColQwen, VideoPrism)."""

    def __init__(
        self,
        chunk_duration: float = 30.0,
        chunk_overlap: float = 0.0,
        cache_chunks: bool = True,
    ):
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.cache_chunks = cache_chunks

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Chunk segmentation requires chunk processor."""
        return {
            "chunk": {
                "chunk_duration": self.chunk_duration,
                "chunk_overlap": self.chunk_overlap,
                "cache_chunks": self.cache_chunks,
            }
        }


class SingleVectorSegmentationStrategy(BaseStrategy):
    """Process video for single-vector embeddings (e.g., VideoPrism LVT)."""

    def __init__(
        self,
        strategy: str = "sliding_window",
        segment_duration: float = 6.0,
        segment_overlap: float = 1.0,
        sampling_fps: float = 2.0,
        max_frames_per_segment: int = 12,
        store_as_single_doc: bool = False,
    ):
        self.strategy = strategy
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.sampling_fps = sampling_fps
        self.max_frames_per_segment = max_frames_per_segment
        self.store_as_single_doc = store_as_single_doc

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Single-vector segmentation requires single-vector processor."""
        return {
            "single_vector": {
                "strategy": self.strategy,
                "segment_duration": self.segment_duration,
                "segment_overlap": self.segment_overlap,
                "sampling_fps": self.sampling_fps,
                "max_frames_per_segment": self.max_frames_per_segment,
                "store_as_single_doc": self.store_as_single_doc,
            }
        }

    async def segment(
        self,
        video_path: Path,
        pipeline_context: Any,
        transcript_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Process video with single-vector processor."""
        # In the new pluggable architecture, we need to check for the processor differently
        if hasattr(pipeline_context, "processor_manager"):
            # Try to get single_vector processor from processor manager
            single_vector_processor = pipeline_context.processor_manager.get_processor(
                "single_vector"
            )

            if single_vector_processor:
                processed_data = single_vector_processor.process_video(
                    video_path=video_path, transcript_data=transcript_data
                )
                # Convert VideoSegment objects to dictionaries for consistency
                processed_data_serializable = processed_data.copy()
                processed_data_serializable["segments"] = [
                    seg.to_dict() for seg in processed_data["segments"]
                ]
                return {"single_vector_processing": processed_data_serializable}

            # Fallback: check for old-style processor
            elif (
                hasattr(pipeline_context, "single_vector_processor")
                and pipeline_context.single_vector_processor
            ):
                processed_data = pipeline_context.single_vector_processor.process_video(
                    video_path=video_path, transcript_data=transcript_data
                )
                # Convert VideoSegment objects to dictionaries for consistency
                processed_data_serializable = processed_data.copy()
                processed_data_serializable["segments"] = [
                    seg.to_dict() for seg in processed_data["segments"]
                ]
                return {"single_vector_processing": processed_data_serializable}

        return {"error": "No single-vector processor configured"}


class AudioTranscriptionStrategy(BaseStrategy):
    """Transcribe audio from video."""

    def __init__(self, model: str = "whisper-large-v3", language: str = "auto"):
        self.model = model
        self.language = language

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Audio transcription requires audio processor."""
        return {"audio": {"model": self.model, "language": self.language}}


class VLMDescriptionStrategy(BaseStrategy):
    """Generate descriptions using VLM."""

    def __init__(self, model_name: str = "gpt-4-vision", batch_size: int = 10):
        self.model_name = model_name
        self.batch_size = batch_size

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """VLM description requires VLM processor."""
        return {"vlm": {"model_name": self.model_name, "batch_size": self.batch_size}}

    async def generate_descriptions(
        self,
        keyframes_data: Dict[str, Any],
        video_path: Path,
        pipeline_context: Any,
        cached_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate descriptions (for backward compatibility)."""
        # Implementation would use VLM processor through pipeline_context
        # For now, return None to maintain compatibility
        return None


class NoDescriptionStrategy(BaseStrategy):
    """No descriptions needed."""

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """No processors required."""
        return {}

    async def generate_descriptions(
        self,
        keyframes_data: Dict[str, Any],
        video_path: Path,
        pipeline_context: Any,
        cached_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """No descriptions generated."""
        return None


class MultiVectorEmbeddingStrategy(BaseStrategy):
    """Generate multi-vector embeddings."""

    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        self.model_name = model_name

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Multi-vector embedding uses the generic embedding generator."""
        return {"embedding": {"type": "multi_vector", "model_name": self.model_name}}

    async def generate_embeddings_with_processor(
        self, results: Dict[str, Any], pipeline_context: Any, processor_manager: Any
    ) -> Dict[str, Any]:
        """Generate embeddings (for backward compatibility)."""
        # Prepare data for embedding generation
        wrapped_results = {
            "video_id": (
                pipeline_context.video_path.stem
                if hasattr(pipeline_context, "video_path")
                else "unknown"
            ),
            "video_path": (
                str(pipeline_context.video_path)
                if hasattr(pipeline_context, "video_path")
                else ""
            ),
            "results": results,
        }

        return await pipeline_context.generate_embeddings(wrapped_results)


class SingleVectorEmbeddingStrategy(BaseStrategy):
    """Generate single-vector embeddings."""

    def __init__(self, model_name: str = "google/videoprism-base"):
        self.model_name = model_name

    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """Single-vector embedding uses the generic embedding generator."""
        return {"embedding": {"type": "single_vector", "model_name": self.model_name}}

    async def generate_embeddings_with_processor(
        self, results: Dict[str, Any], pipeline_context: Any, processor_manager: Any
    ) -> Dict[str, Any]:
        """Generate embeddings (for backward compatibility)."""
        # Prepare data for embedding generation
        wrapped_results = {
            "video_id": (
                pipeline_context.video_path.stem
                if hasattr(pipeline_context, "video_path")
                else "unknown"
            ),
            "video_path": (
                str(pipeline_context.video_path)
                if hasattr(pipeline_context, "video_path")
                else ""
            ),
            "results": {},
        }

        # Map single-vector results
        if "single_vector_processing" in results:
            wrapped_results["results"]["single_vector_processing"] = results[
                "single_vector_processing"
            ]
        if "transcript" in results:
            wrapped_results["results"]["transcript"] = results["transcript"]

        return await pipeline_context.generate_embeddings(wrapped_results)
