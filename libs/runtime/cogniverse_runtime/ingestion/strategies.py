#!/usr/bin/env python3
"""
Strategy implementations extending BaseStrategy.

These strategies work with the pluggable ProcessorManager.
Supports video, image, audio, document, and code content types.
"""

from pathlib import Path
from typing import Any

from .processor_base import BaseStrategy

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".rtf"}
CODE_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".jsx", ".mjs"},
    "typescript": {".ts", ".tsx"},
    "go": {".go"},
}


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

    def __init__(self, model: str = "base", language: str = "auto"):
        # Default to the "base" Whisper model (~150MB) to keep the ingestion
        # pod's memory footprint bounded. Profiles that genuinely need larger
        # accuracy should set "model": "large-v3" (3GB) or another tier
        # explicitly in the profile's strategy params.
        self.model = model
        self.language = language

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """Audio transcription requires audio processor."""
        return {"audio": {"model": self.model, "language": self.language}}


class VLMDescriptionStrategy(BaseStrategy):
    """Generate descriptions using VLM via Modal service."""

    def __init__(
        self,
        vlm_endpoint: str,
        batch_size: int = 500,
        timeout: int = 10800,
        auto_start: bool = True,
    ):
        self.vlm_endpoint = vlm_endpoint
        self.batch_size = batch_size
        self.timeout = timeout
        self.auto_start = auto_start

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        """VLM description requires VLM processor."""
        return {
            "vlm": {
                "vlm_endpoint": self.vlm_endpoint,
                "batch_size": self.batch_size,
                "timeout": self.timeout,
                "auto_start": self.auto_start,
            }
        }

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
    """Generate acoustic (CLAP 512-dim) and ColBERT semantic (128-dim multi-vector) embeddings for audio.

    Models are loaded through the model loader infrastructure:
    - CLAP: Loaded by AudioEmbeddingGenerator (lazy, via transformers)
    - ColBERT: Loaded by ModelLoaderFactory → ColBERTModelLoader in EmbeddingGeneratorImpl

    This strategy wraps the results and delegates to the pipeline's embedding generator.
    """

    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-unfused",
        colbert_model: str = "lightonai/LateOn",
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
        """Delegate to pipeline_context.generate_embeddings() which routes through EmbeddingGeneratorImpl."""
        if not hasattr(pipeline_context, "video_path"):
            raise AttributeError(
                "AudioEmbeddingStrategy requires pipeline_context.video_path to be set. "
                "Ensure the pipeline context carries a 'video_path' attribute pointing to the content path."
            )
        wrapped_results = {
            "video_id": pipeline_context.video_path.stem,
            "video_path": str(pipeline_context.video_path),
            "results": results,
            "audio_files": results.get("audio_files", []),
            "transcript": results.get("transcript", {}),
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


class CodeSegmentationStrategy(BaseStrategy):
    """Parse source code files into AST-aware chunks using tree-sitter.

    Each source file is parsed into function, method, class, and top-level
    block segments. The structured text per chunk includes: name + signature +
    docstring + body — the same representation colgrep uses.

    Walks directories, respects .gitignore, filters by language extensions.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        max_files: int = 50000,
    ):
        self.languages = languages or ["python", "typescript", "go", "javascript"]
        self.max_files = max_files

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {
            "code_file": {
                "languages": self.languages,
                "max_files": self.max_files,
            }
        }

    def get_supported_extensions(self) -> set[str]:
        """Return the union of file extensions for configured languages."""
        exts: set[str] = set()
        for lang in self.languages:
            exts.update(CODE_EXTENSIONS.get(lang, set()))
        return exts

    @staticmethod
    def _get_language(lang_name: str):
        """Load a tree-sitter Language object for the given language name."""
        import tree_sitter

        lang_modules = {
            "python": "tree_sitter_python",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "go": "tree_sitter_go",
        }
        module_name = lang_modules.get(lang_name)
        if not module_name:
            raise ValueError(f"Unsupported language: {lang_name}")

        import importlib

        mod = importlib.import_module(module_name)

        if lang_name == "typescript":
            return tree_sitter.Language(mod.language_typescript())
        return tree_sitter.Language(mod.language())

    @staticmethod
    def _lang_for_extension(ext: str) -> str | None:
        """Return the language name for a file extension."""
        for lang, exts in CODE_EXTENSIONS.items():
            if ext in exts:
                return lang
        return None

    def parse_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse a single source file into AST chunks.

        Returns a list of segment dicts with keys:
            content, metadata (file, type, name, signature, line_start, line_end)
        """
        import tree_sitter

        ext = file_path.suffix.lower()
        lang_name = self._lang_for_extension(ext)
        if not lang_name or lang_name not in self.languages:
            return []

        source_bytes = file_path.read_bytes()
        if not source_bytes.strip():
            return []

        language = self._get_language(lang_name)
        parser = tree_sitter.Parser(language)
        tree = parser.parse(source_bytes)

        segments = []
        self._extract_segments(tree.root_node, source_bytes, file_path, segments)

        if not segments:
            # If no functions/classes found, treat entire file as one segment
            text = source_bytes.decode("utf-8", errors="replace")
            segments.append(
                {
                    "content": text,
                    "metadata": {
                        "file": str(file_path),
                        "type": "module",
                        "name": file_path.stem,
                        "signature": "",
                        "line_start": 1,
                        "line_end": text.count("\n") + 1,
                        "language": lang_name,
                    },
                }
            )

        return segments

    def _extract_segments(
        self,
        node,
        source: bytes,
        file_path: Path,
        segments: list[dict[str, Any]],
    ) -> None:
        """Recursively extract function/class/method segments from AST."""
        lang_name = self._lang_for_extension(file_path.suffix.lower()) or "unknown"

        extractable_types = {
            "function_definition",  # Python
            "class_definition",  # Python
            "function_declaration",  # JS/TS/Go
            "method_definition",  # JS/TS
            "class_declaration",  # JS/TS
            "arrow_function",  # JS/TS (only named, via parent)
            "type_declaration",  # Go
            "method_declaration",  # Go
        }

        if node.type in extractable_types:
            text = source[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )
            name = self._extract_name(node)
            signature = self._extract_signature(node, source)
            seg_type = "class" if "class" in node.type else "function"

            segments.append(
                {
                    "content": text,
                    "metadata": {
                        "file": str(file_path),
                        "type": seg_type,
                        "name": name,
                        "signature": signature,
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "language": lang_name,
                    },
                }
            )

            # For classes, also extract child methods
            if "class" in node.type:
                for child in node.children:
                    if child.type in (
                        "block",
                        "class_body",
                        "declaration_list",
                    ):
                        self._extract_segments(child, source, file_path, segments)
                return

        for child in node.children:
            self._extract_segments(child, source, file_path, segments)

    @staticmethod
    def _extract_name(node) -> str:
        """Extract the name identifier from an AST node."""
        for child in node.children:
            if child.type in (
                "identifier",
                "name",
                "property_identifier",
                "field_identifier",
                "type_identifier",
            ):
                return child.text.decode("utf-8", errors="replace")
            if child.type == "type_spec":
                for gc in child.children:
                    if gc.type == "type_identifier":
                        return gc.text.decode("utf-8", errors="replace")
        return "<anonymous>"

    @staticmethod
    def _extract_signature(node, source: bytes) -> str:
        """Extract the function/method signature (first line up to body)."""
        start = node.start_byte
        # Find the body child to know where signature ends
        for child in node.children:
            if child.type in (
                "block",
                "statement_block",
                "class_body",
                "declaration_list",
                "function_body",
            ):
                sig = (
                    source[start : child.start_byte]
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                return sig

        # Fallback: first line
        text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        return text.split("\n")[0].strip()


class DocumentTextEmbeddingStrategy(BaseStrategy):
    """Generate ColBERT multi-vector embeddings (128-dim per token) for document text.

    The ColBERT model is loaded through the model loader infrastructure
    (ModelLoaderFactory → ColBERTModelLoader) in EmbeddingGeneratorImpl.
    This strategy only wraps the results and delegates embedding generation
    to the pipeline's embedding generator.
    """

    def __init__(
        self,
        colbert_model: str = "lightonai/LateOn",
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
        """Delegate to pipeline_context.generate_embeddings() which routes through EmbeddingGeneratorImpl."""
        if not hasattr(pipeline_context, "video_path"):
            raise AttributeError(
                "DocumentTextEmbeddingStrategy requires pipeline_context.video_path to be set. "
                "Ensure the pipeline context carries a 'video_path' attribute pointing to the content path."
            )
        wrapped_results = {
            "video_id": pipeline_context.video_path.stem,
            "video_path": str(pipeline_context.video_path),
            "results": results,
            "document_files": results.get("document_files", []),
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


class CodeTextEmbeddingStrategy(BaseStrategy):
    """Generate ColBERT multi-vector embeddings (128-dim per token) for source code.

    Uses LateOn-Code-edge (or another ColBERT model) loaded through
    ModelLoaderFactory → ColBERTModelLoader in EmbeddingGeneratorImpl.
    Wraps code segments and delegates to pipeline_context.generate_embeddings().
    """

    def __init__(
        self,
        colbert_model: str = "lightonai/LateOn-Code-edge",
    ):
        self.colbert_model = colbert_model

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {
            "embedding": {
                "type": "code_text",
                "colbert_model": self.colbert_model,
            }
        }

    async def generate_embeddings_with_processor(
        self, results: dict[str, Any], pipeline_context: Any, processor_manager: Any
    ) -> dict[str, Any]:
        """Delegate to pipeline_context.generate_embeddings()."""
        if not hasattr(pipeline_context, "video_path"):
            raise AttributeError(
                "CodeTextEmbeddingStrategy requires pipeline_context.video_path to be set. "
                "Ensure the pipeline context carries a 'video_path' attribute pointing to the content path."
            )
        wrapped_results = {
            "video_id": pipeline_context.video_path.stem,
            "video_path": str(pipeline_context.video_path),
            "results": results,
            "code_files": results.get("code_files", []),
        }
        return await pipeline_context.generate_embeddings(wrapped_results)
