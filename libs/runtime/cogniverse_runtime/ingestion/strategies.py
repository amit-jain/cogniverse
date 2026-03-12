#!/usr/bin/env python3
"""
Strategy implementations extending BaseStrategy.

These strategies work with the pluggable ProcessorManager.
Supports video, image, audio, and document content types.
"""

from pathlib import Path
from typing import Any

from .processor_base import BaseStrategy

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".rtf"}


class FrameSegmentationStrategy(BaseStrategy):
    """Extract frames from video (e.g., for ColPali)."""

    def __init__(
        self, fps: float = 1.0, threshold: float = 0.999, max_frames: int = 3000
    ):
        self.fps = fps
        self.threshold = threshold
        self.max_frames = max_frames

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
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

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
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

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
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
        transcript_data: dict | None = None,
    ) -> dict[str, Any]:
        """Process video with single-vector processor."""
        if not hasattr(pipeline_context, "processor_manager"):
            raise ValueError("Pipeline context missing processor_manager")

        single_vector_processor = pipeline_context.processor_manager.get_processor(
            "single_vector"
        )

        if not single_vector_processor:
            raise ValueError(
                "No single-vector processor available in processor_manager"
            )

        processed_data = single_vector_processor.process_video(
            video_path=video_path, transcript_data=transcript_data
        )

        processed_data_serializable = processed_data.copy()
        processed_data_serializable["segments"] = [
            seg.to_dict() for seg in processed_data["segments"]
        ]

        return {"single_vector_processing": processed_data_serializable}


class AudioTranscriptionStrategy(BaseStrategy):
    """Transcribe audio from video."""

    def __init__(self, model: str = "whisper-large-v3", language: str = "auto"):
        self.model = model
        self.language = language

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """Audio transcription requires audio processor."""
        return {"audio": {"model": self.model, "language": self.language}}


class VLMDescriptionStrategy(BaseStrategy):
    """Generate descriptions using VLM."""

    def __init__(self, model_name: str = "gpt-4-vision", batch_size: int = 10):
        self.model_name = model_name
        self.batch_size = batch_size

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """VLM description requires VLM processor."""
        return {"vlm": {"model_name": self.model_name, "batch_size": self.batch_size}}

    async def generate_descriptions(
        self,
        segments: Any,
        video_path: Path,
        pipeline_context: Any,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate frame descriptions via the VLM processor."""
        if not hasattr(pipeline_context, "processor_manager"):
            raise ValueError("Pipeline context missing processor_manager")
        processor = pipeline_context.processor_manager.get_processor("vlm")
        if processor is None:
            raise ValueError(
                f"VLMDescriptionStrategy requires a 'vlm' processor but none was "
                f"initialised in ProcessorManager for profile "
                f"{getattr(pipeline_context, 'schema_name', 'unknown')!r}."
            )
        return processor.generate_descriptions(segments)


class NoDescriptionStrategy(BaseStrategy):
    """No descriptions needed."""

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """No processors required."""
        return {}


class NoTranscriptionStrategy(BaseStrategy):
    """No transcription needed (for non-video content like images)."""

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {}


class ImageSegmentationStrategy(BaseStrategy):
    """Load images from a directory and present them as keyframes for ColPali embedding.

    Each image becomes one "keyframe" in the same format that FrameSegmentationStrategy
    produces, so the downstream MultiVectorEmbeddingStrategy works unchanged.
    """

    def __init__(self, max_images: int = 10000, **kwargs):
        self.max_images = max_images

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """Image segmentation uses a special 'image' processor key so
        ProcessingStrategySet dispatches to _process_segmentation correctly."""
        return {"image": {"max_images": self.max_images}}


class AudioFileSegmentationStrategy(BaseStrategy):
    """Discover audio files in a directory for audio ingestion.

    Each audio file becomes one item for transcription and embedding.
    Analogous to ImageSegmentationStrategy for images.
    """

    def __init__(self, max_files: int = 10000):
        self.max_files = max_files

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {"audio_file": {"max_files": self.max_files}}


class AudioEmbeddingStrategy(BaseStrategy):
    """Generate acoustic (CLAP 512-dim) and ColBERT semantic (128-dim multi-vector) embeddings for audio."""

    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-unfused",
        colbert_model: str = "lightonai/GTE-ModernColBERT-v1",
    ):
        self.clap_model = clap_model
        self.colbert_model = colbert_model

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {
            "embedding": {
                "type": "audio",
                "clap_model": self.clap_model,
                "colbert_model": self.colbert_model,
            }
        }

    async def generate_embeddings_with_processor(
        self, results: dict[str, Any], pipeline_context: Any, processor_manager: Any
    ) -> dict[str, Any]:
        """Generate acoustic (CLAP) + ColBERT semantic embeddings for audio."""
        from .processors.audio_embedding_generator import AudioEmbeddingGenerator

        generator = AudioEmbeddingGenerator(
            clap_model=self.clap_model,
        )

        audio_files = results.get("audio_files", [])
        transcript_data = results.get("transcript", {})
        if not isinstance(transcript_data, dict):
            raise TypeError(
                f"Expected transcript_data to be a dict, got {type(transcript_data).__name__!r}. "
                "AudioEmbeddingStrategy requires the transcription step to produce a dict."
            )
        transcript_text = transcript_data.get("full_text", "")

        from pylate import models as pylate_models

        colbert = pylate_models.ColBERT(self.colbert_model)

        embedded_docs = []
        for audio_file_info in audio_files:
            audio_path = Path(audio_file_info["path"])
            acoustic_emb = generator.generate_acoustic_embedding(audio_path=audio_path)

            if not transcript_text.strip():
                raise ValueError(
                    f"Audio file {audio_path.name!r} has no transcript text. "
                    "AudioEmbeddingStrategy requires transcription to produce non-empty text "
                    "before generating semantic embeddings."
                )
            semantic_emb = colbert.encode([transcript_text], is_query=False)[0]

            embedded_docs.append({
                "audio_id": audio_file_info.get("audio_id", audio_path.stem),
                "audio_path": str(audio_path),
                "acoustic_embedding": acoustic_emb.tolist(),
                "semantic_embedding": [tok.tolist() for tok in semantic_emb],
            })

        if not hasattr(pipeline_context, "video_path"):
            raise AttributeError(
                "AudioEmbeddingStrategy requires pipeline_context.video_path to be set. "
                "Ensure the pipeline context carries a 'video_path' attribute pointing to the content path."
            )
        content_path = pipeline_context.video_path
        wrapped_results = {
            "video_id": content_path.stem,
            "video_path": str(content_path),
            "results": results,
            "audio_embeddings": embedded_docs,
        }

        return await pipeline_context.generate_embeddings(wrapped_results)


class DocumentSegmentationStrategy(BaseStrategy):
    """Discover document files in a directory for text-based ingestion.

    Each document file is discovered and its text extracted for embedding.
    Supports PDF (via PyPDF2), plain text, and markdown files.
    """

    def __init__(self, max_files: int = 10000):
        self.max_files = max_files

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {"document_file": {"max_files": self.max_files}}


class DocumentTextEmbeddingStrategy(BaseStrategy):
    """Generate ColBERT multi-vector embeddings (128-dim per token) for document text."""

    def __init__(
        self,
        colbert_model: str = "lightonai/GTE-ModernColBERT-v1",
    ):
        self.colbert_model = colbert_model

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {
            "embedding": {
                "type": "document_text",
                "colbert_model": self.colbert_model,
            }
        }

    async def generate_embeddings_with_processor(
        self, results: dict[str, Any], pipeline_context: Any, processor_manager: Any
    ) -> dict[str, Any]:
        """Generate ColBERT token-level embeddings for documents."""
        from pylate import models as pylate_models

        model = pylate_models.ColBERT(self.colbert_model)

        document_files = results.get("document_files", [])

        embedded_docs = []
        for doc_info in document_files:
            text = doc_info.get("extracted_text", "")
            if not text.strip():
                raise ValueError(
                    f"Document {doc_info['filename']!r} has no extracted text. "
                    "Cannot generate embeddings for empty documents."
                )

            token_embeddings = model.encode([text[:8192]], is_query=False)[0]
            embedded_docs.append({
                "document_id": doc_info.get("document_id", Path(doc_info["path"]).stem),
                "document_path": doc_info["path"],
                "embedding": [tok.tolist() for tok in token_embeddings],
                "text_length": len(text),
            })

        if not hasattr(pipeline_context, "video_path"):
            raise AttributeError(
                "DocumentTextEmbeddingStrategy requires pipeline_context.video_path to be set. "
                "Ensure the pipeline context carries a 'video_path' attribute pointing to the content path."
            )
        content_path = pipeline_context.video_path
        wrapped_results = {
            "video_id": content_path.stem,
            "video_path": str(content_path),
            "results": results,
            "document_embeddings": embedded_docs,
        }

        return await pipeline_context.generate_embeddings(wrapped_results)


class MultiVectorEmbeddingStrategy(BaseStrategy):
    """Generate multi-vector embeddings."""

    def __init__(self, model_name: str = "vidore/colsmol-500m", **kwargs):
        self.model_name = model_name

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """Multi-vector embedding uses the generic embedding generator."""
        return {"embedding": {"type": "multi_vector", "model_name": self.model_name}}

    async def generate_embeddings_with_processor(
        self, results: dict[str, Any], pipeline_context: Any, processor_manager: Any
    ) -> dict[str, Any]:
        """Generate embeddings using pipeline context."""
        if not hasattr(pipeline_context, "video_path"):
            raise AttributeError(
                "MultiVectorEmbeddingStrategy requires pipeline_context.video_path to be set. "
                "Ensure the pipeline context carries a 'video_path' attribute pointing to the content path."
            )
        wrapped_results = {
            "video_id": pipeline_context.video_path.stem,
            "video_path": str(pipeline_context.video_path),
            "results": results,
        }

        return await pipeline_context.generate_embeddings(wrapped_results)


class SingleVectorEmbeddingStrategy(BaseStrategy):
    """Generate single-vector embeddings."""

    def __init__(self, model_name: str = "google/videoprism-base", **kwargs):
        self.model_name = model_name

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """Single-vector embedding uses the generic embedding generator."""
        return {"embedding": {"type": "single_vector", "model_name": self.model_name}}

    async def generate_embeddings_with_processor(
        self, results: dict[str, Any], pipeline_context: Any, processor_manager: Any
    ) -> dict[str, Any]:
        """Generate embeddings using pipeline context."""
        if not hasattr(pipeline_context, "video_path"):
            raise AttributeError(
                "SingleVectorEmbeddingStrategy requires pipeline_context.video_path to be set. "
                "Ensure the pipeline context carries a 'video_path' attribute pointing to the content path."
            )
        wrapped_results = {
            "video_id": pipeline_context.video_path.stem,
            "video_path": str(pipeline_context.video_path),
            "results": {},
        }

        if "single_vector_processing" in results:
            wrapped_results["results"]["single_vector_processing"] = results[
                "single_vector_processing"
            ]
        if "transcript" in results:
            wrapped_results["results"]["transcript"] = results["transcript"]

        return await pipeline_context.generate_embeddings(wrapped_results)
