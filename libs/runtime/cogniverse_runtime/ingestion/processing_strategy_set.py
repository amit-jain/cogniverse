#!/usr/bin/env python3
"""
Dynamic ProcessingStrategySet - No hardcoded strategies.

Container for processing strategies that works with any number and type of strategies.
"""

import asyncio
from pathlib import Path
from typing import Any

from .processor_base import BaseStrategy
from .strategies import DOCUMENT_EXTENSIONS


class ProcessingStrategySet:
    """Dynamic container for processing strategies - no hardcoded strategy types."""

    def __init__(self, **strategies):
        """
        Initialize with any number of strategies.

        Args:
            **strategies: Named strategies (e.g., segmentation=FrameStrategy(), transcription=AudioStrategy())
        """
        self._strategies: dict[str, BaseStrategy] = {}

        for name, strategy in strategies.items():
            if isinstance(strategy, BaseStrategy):
                self._strategies[name] = strategy
            else:
                raise ValueError(f"Strategy '{name}' must extend BaseStrategy")

    def get_all_strategies(self) -> list[BaseStrategy]:
        """Get all strategies - simple and explicit."""
        return list(self._strategies.values())

    def get_strategy(self, name: str) -> BaseStrategy:
        """Get a strategy by name."""
        return self._strategies.get(name)

    def has_strategy(self, name: str) -> bool:
        """Check if strategy exists."""
        return name in self._strategies

    def list_strategy_names(self) -> list[str]:
        """List all strategy names."""
        return list(self._strategies.keys())

    def get_all_required_processors(self) -> dict[str, dict[str, Any]]:
        """Get all required processors from all strategies."""
        all_requirements = {}
        for strategy in self._strategies.values():
            requirements = strategy.get_required_processors()
            all_requirements.update(requirements)
        return all_requirements

    @property
    def segmentation(self):
        return self.get_strategy("segmentation")

    @property
    def transcription(self):
        return self.get_strategy("transcription")

    @property
    def description(self):
        return self.get_strategy("description")

    @property
    def embedding(self):
        return self.get_strategy("embedding")

    async def process(
        self, video_path: Path, processor_manager, pipeline_context
    ) -> dict[str, Any]:
        """
        Process through all strategies dynamically.

        Args:
            video_path: Path to video file
            processor_manager: Manager containing initialized processors
            pipeline_context: Pipeline instance for context

        Returns:
            Processing results from all strategies
        """
        schema_name = getattr(pipeline_context, "schema_name", "unknown")
        pipeline_context.logger.info(
            f"🎯 ProcessingStrategySet.process() starting for {video_path.name} [Schema: {schema_name}]"
        )

        results = {}

        strategy_order = ["segmentation", "transcription", "description", "embedding"]

        for strategy_name in strategy_order:
            strategy = self.get_strategy(strategy_name)
            if strategy:
                pipeline_context.logger.info(
                    f"Processing {strategy_name} strategy: {type(strategy).__name__}"
                )

                strategy_result = await self._process_strategy(
                    strategy_name,
                    strategy,
                    video_path,
                    processor_manager,
                    pipeline_context,
                    results,
                )

                if isinstance(strategy_result, dict):
                    results.update(strategy_result)
                    pipeline_context.logger.info(
                        f"  ✅ {strategy_name} completed: {list(strategy_result.keys())}"
                    )

        pipeline_context.logger.info(
            f"🏁 ProcessingStrategySet completed with: {list(results.keys())}"
        )
        return results

    async def _process_strategy(
        self,
        strategy_name: str,
        strategy: BaseStrategy,
        video_path: Path,
        processor_manager,
        pipeline_context,
        accumulated_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a single strategy with proper method dispatch."""
        if strategy_name == "segmentation":
            return await self._process_segmentation(
                strategy, video_path, processor_manager, pipeline_context
            )
        elif strategy_name == "transcription":
            return await self._process_transcription(
                strategy,
                video_path,
                processor_manager,
                pipeline_context,
                accumulated_results,
            )
        elif strategy_name == "description":
            return await self._process_description(
                strategy,
                video_path,
                processor_manager,
                pipeline_context,
                accumulated_results,
            )
        elif strategy_name == "embedding":
            return await self._process_embedding(
                strategy,
                video_path,
                processor_manager,
                pipeline_context,
                accumulated_results,
            )
        else:
            raise ValueError(
                f"Unknown strategy type '{strategy_name}'. "
                f"Supported types: segmentation, transcription, description, embedding."
            )

    async def _process_segmentation(
        self, strategy, video_path: Path, processor_manager, pipeline_context
    ) -> dict[str, Any]:
        """Process segmentation strategy."""
        requirements = strategy.get_required_processors()

        if "keyframe" in requirements:
            processor = processor_manager.get_processor("keyframe")
            if processor:
                result = processor.extract_keyframes(
                    video_path, pipeline_context.profile_output_dir
                )
                num_frames = len(result.get("keyframes", [])) if result else 0
                pipeline_context.logger.info(f"  🖼️ Extracted {num_frames} keyframes")
                return {"keyframes": result}

        elif "chunk" in requirements:
            processor = processor_manager.get_processor("chunk")
            if processor:
                result = processor.extract_chunks(
                    video_path, pipeline_context.profile_output_dir
                )
                num_chunks = len(result.get("chunks", [])) if result else 0
                pipeline_context.logger.info(
                    f"  🎬 Extracted {num_chunks} video chunks"
                )
                return {"video_chunks": result}

        elif "single_vector" in requirements:
            result = await strategy.segment(video_path, pipeline_context, None)
            num_segments = (
                len(result.get("single_vector_processing", {}).get("segments", []))
                if result
                else 0
            )
            pipeline_context.logger.info(
                f"  📦 Processed {num_segments} single-vector segments"
            )
            return result

        elif "document_file" in requirements:
            content_path = video_path

            if content_path.is_dir():
                doc_files = sorted(
                    f
                    for f in content_path.iterdir()
                    if f.suffix.lower() in DOCUMENT_EXTENSIONS
                )
            elif content_path.suffix.lower() in DOCUMENT_EXTENSIONS:
                doc_files = [content_path]
            else:
                raise ValueError(
                    f"Expected document file or directory, got: {content_path}"
                )

            max_files = requirements["document_file"].get("max_files", 10000)
            doc_files = doc_files[:max_files]

            if not doc_files:
                raise ValueError(f"No document files found at {content_path}")

            document_file_list = []
            for idx, doc_path in enumerate(doc_files):
                extracted_text = self._extract_document_text(doc_path)
                document_file_list.append(
                    {
                        "document_id": doc_path.stem,
                        "file_index": idx,
                        "path": str(doc_path),
                        "filename": doc_path.name,
                        "document_type": doc_path.suffix.lstrip("."),
                        "extracted_text": extracted_text,
                        "text_length": len(extracted_text),
                    }
                )

            pipeline_context.logger.info(
                f"  Discovered {len(document_file_list)} document files"
            )
            return {"document_files": document_file_list}

        elif "audio_file" in requirements:
            content_path = video_path
            audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

            if content_path.is_dir():
                audio_files = sorted(
                    f
                    for f in content_path.iterdir()
                    if f.suffix.lower() in audio_extensions
                )
            elif content_path.suffix.lower() in audio_extensions:
                audio_files = [content_path]
            else:
                raise ValueError(
                    f"Expected audio file or directory, got: {content_path}"
                )

            max_files = requirements["audio_file"].get("max_files", 10000)
            audio_files = audio_files[:max_files]

            if not audio_files:
                raise ValueError(f"No audio files found at {content_path}")

            audio_file_list = []
            for idx, audio_path in enumerate(audio_files):
                audio_file_list.append(
                    {
                        "audio_id": audio_path.stem,
                        "file_index": idx,
                        "path": str(audio_path),
                        "filename": audio_path.name,
                    }
                )

            pipeline_context.logger.info(
                f"  Discovered {len(audio_file_list)} audio files"
            )
            return {"audio_files": audio_file_list}

        elif "image" in requirements:
            import shutil
            import time as _time

            content_path = video_path
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

            if content_path.is_dir():
                image_files = sorted(
                    f
                    for f in content_path.iterdir()
                    if f.suffix.lower() in image_extensions
                )
            elif content_path.suffix.lower() in image_extensions:
                image_files = [content_path]
            else:
                raise ValueError(
                    f"Expected image file or directory, got: {content_path}"
                )

            max_images = requirements["image"].get("max_images", 10000)
            image_files = image_files[:max_images]

            if not image_files:
                raise ValueError(f"No image files found at {content_path}")

            content_id = content_path.stem
            output_dir = pipeline_context.profile_output_dir
            keyframes_dir = output_dir / "keyframes" / content_id
            keyframes_dir.mkdir(parents=True, exist_ok=True)

            keyframes = []
            for idx, img_path in enumerate(image_files):
                dest = keyframes_dir / f"frame_{idx:04d}{img_path.suffix}"
                shutil.copy2(img_path, dest)
                keyframes.append(
                    {
                        "frame_id": idx,
                        "original_frame_number": idx,
                        "timestamp": 0.0,
                        "path": str(dest),
                        "filename": dest.name,
                        "source_image": str(img_path),
                    }
                )

            result = {
                "video_id": content_id,
                "video_path": str(content_path),
                "keyframes": keyframes,
                "stats": {
                    "total_keyframes": len(keyframes),
                    "total_frames": len(keyframes),
                    "extraction_method": "image_load",
                },
                "created_at": _time.time(),
            }
            pipeline_context.logger.info(
                f"  Loaded {len(keyframes)} images as keyframes"
            )
            return {"keyframes": result}

        elif "code_file" in requirements:
            from .strategies import CodeSegmentationStrategy

            if not isinstance(strategy, CodeSegmentationStrategy):
                raise TypeError(
                    f"'code_file' processor requires CodeSegmentationStrategy, "
                    f"got {type(strategy).__name__}"
                )

            content_path = video_path
            supported_exts = strategy.get_supported_extensions()

            if content_path.is_dir():
                code_files = sorted(
                    f
                    for f in content_path.rglob("*")
                    if f.is_file()
                    and f.suffix.lower() in supported_exts
                    and ".git" not in f.parts
                    and "__pycache__" not in f.parts
                    and "node_modules" not in f.parts
                    and ".venv" not in f.parts
                )
            elif content_path.suffix.lower() in supported_exts:
                code_files = [content_path]
            else:
                raise ValueError(
                    f"Expected code file or directory, got: {content_path}"
                )

            max_files = requirements["code_file"].get("max_files", 50000)
            code_files = code_files[:max_files]

            if not code_files:
                raise ValueError(f"No code files found at {content_path}")

            code_file_list = []
            for idx, code_path in enumerate(code_files):
                segments = strategy.parse_file(code_path)
                for seg in segments:
                    code_file_list.append(
                        {
                            "document_id": f"{code_path.stem}_{seg['metadata']['name']}_{seg['metadata']['line_start']}",
                            "file_index": idx,
                            "path": str(code_path),
                            "filename": code_path.name,
                            "document_type": code_path.suffix.lstrip("."),
                            "extracted_text": seg["content"],
                            "text_length": len(seg["content"]),
                            "chunk_type": seg["metadata"]["type"],
                            "chunk_name": seg["metadata"]["name"],
                            "signature": seg["metadata"]["signature"],
                            "line_start": seg["metadata"]["line_start"],
                            "line_end": seg["metadata"]["line_end"],
                            "language": seg["metadata"].get("language", "unknown"),
                        }
                    )

            pipeline_context.logger.info(
                f"  Parsed {len(code_file_list)} code segments from {len(code_files)} files"
            )
            return {"code_files": code_file_list}

        else:
            processor_keys = list(requirements.keys())
            raise ValueError(
                f"Segmentation strategy {type(strategy).__name__!r} requires unknown "
                f"processor(s) {processor_keys}. Supported: keyframe, chunk, single_vector, "
                f"document_file, audio_file, image, code_file."
            )

    async def _process_transcription(
        self,
        strategy,
        video_path: Path,
        processor_manager,
        pipeline_context,
        accumulated_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Process transcription strategy."""
        if not pipeline_context.config.transcribe_audio:
            return {}

        requirements = strategy.get_required_processors()

        if "audio" in requirements:
            processor = processor_manager.get_processor("audio")
            if processor:
                # Don't pass async cache to sync transcribe_audio — the cache
                # methods are async coroutines and can't be called from sync code.
                result = await asyncio.to_thread(
                    processor.transcribe_audio,
                    video_path,
                    pipeline_context.profile_output_dir,
                    None,
                )
                return {"transcript": result}

        return {}

    async def _process_description(
        self,
        strategy,
        video_path: Path,
        processor_manager,
        pipeline_context,
        accumulated_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Process description strategy."""
        if not pipeline_context.config.generate_descriptions:
            return {}

        requirements = strategy.get_required_processors()

        if "vlm" in requirements:
            if not hasattr(strategy, "generate_descriptions"):
                raise AttributeError(
                    f"Strategy {type(strategy).__name__!r} requires 'vlm' processor "
                    "but does not implement generate_descriptions()."
                )
            segments = accumulated_results.get("segments", {})
            result = await strategy.generate_descriptions(
                segments, video_path, pipeline_context, {}
            )
            return {"descriptions": result} if result else {}

        return {}

    async def _process_embedding(
        self,
        strategy,
        video_path: Path,
        processor_manager,
        pipeline_context,
        accumulated_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Process embedding strategy."""
        if not pipeline_context.config.generate_embeddings:
            return {}

        pipeline_context.logger.info(
            f"🧬 Generating embeddings with strategy: {type(strategy).__name__}"
        )
        pipeline_context.logger.info(
            f"  Data available: {list(accumulated_results.keys())}"
        )

        if not hasattr(strategy, "generate_embeddings_with_processor"):
            raise AttributeError(
                f"Strategy {type(strategy).__name__!r} does not implement "
                "generate_embeddings_with_processor()."
            )
        embeddings = await strategy.generate_embeddings_with_processor(
            accumulated_results, pipeline_context, processor_manager
        )
        if isinstance(embeddings, dict):
            docs_fed = embeddings.get("documents_fed", 0)
            pipeline_context.logger.info(
                f"  ✅ Embeddings generated: {docs_fed} documents fed to backend"
            )
        return {"embeddings": embeddings}

    @staticmethod
    def _extract_document_text(doc_path: Path) -> str:
        """Extract text content from a document file.

        Supports PDF (via PyPDF2), plain text (.txt, .md, .rtf), and raises
        ValueError for unsupported formats.
        """
        suffix = doc_path.suffix.lower()

        if suffix == ".pdf":
            import PyPDF2

            text = ""
            with open(doc_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()

        if suffix in {".txt", ".md", ".rtf"}:
            return doc_path.read_text(encoding="utf-8").strip()

        if suffix in {".docx", ".doc"}:
            raise ValueError(
                f"Document type {suffix!r} requires python-docx. "
                f"Install it with: pip install python-docx"
            )

        raise ValueError(f"Unsupported document type: {suffix!r} for {doc_path}")
