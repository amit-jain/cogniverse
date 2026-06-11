#!/usr/bin/env python3
"""
Generic Embedding Generator - Unified processing for all segment types.

This module provides a simplified, strategy-driven embedding generator that handles
all video processing profiles through a single generic method.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cogniverse_sdk.document import ContentType, Document, ProcessingStatus

from .embedding_generator import BaseEmbeddingGenerator, EmbeddingResult


class EmbeddingGeneratorImpl(BaseEmbeddingGenerator):
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
        config: dict[str, Any],
        logger: logging.Logger | None = None,
        backend_client: Any = None,
    ):
        super().__init__(config, logger)

        self.profile_config = config
        self.model_name = self.profile_config.get("embedding_model")
        if not self.model_name:
            raise ValueError(
                f"Profile config missing 'embedding_model'. "
                f"Got keys: {sorted(self.profile_config.keys())}"
            )
        self.backend_client = backend_client
        self.schema_name = self.profile_config.get("schema_name")

        # Storage mode determines if we create one doc per segment or one doc total
        self.storage_mode = self.profile_config.get("storage_mode", "multi_doc")

        # Model and processor
        self.model = None
        self.processor = None
        self.videoprism_loader = None
        self.colbert_model = None

        # Load model if needed
        if self._should_load_model():
            self._load_model()

    def _should_load_model(self) -> bool:
        """Check if model should be loaded during initialization.

        ColPali/ColQwen models are loaded lazily when the first frame/chunk
        is processed. All other models (ColBERT, VideoPrism) load at init.
        """
        model_loader = self.profile_config.get("model_loader")
        if not model_loader:
            raise ValueError(
                f"Profile config missing 'model_loader'. "
                f"Got keys: {sorted(self.profile_config.keys())}"
            )
        return model_loader not in ("colpali", "colqwen")

    def _load_model(self):
        """Load the embedding model based on model_loader configuration."""
        from cogniverse_core.common.models import get_or_load_model

        model_loader = self.profile_config.get("model_loader")
        if not model_loader:
            raise ValueError(
                f"Profile config missing 'model_loader'. "
                f"Got keys: {sorted(self.profile_config.keys())}"
            )

        try:
            if model_loader == "colbert":
                model_name = self.profile_config.get("semantic_model", self.model_name)
                self.colbert_model, _ = get_or_load_model(
                    model_name, self.profile_config, self.logger
                )
            elif model_loader == "videoprism":
                self.videoprism_loader, _ = get_or_load_model(
                    self.model_name, self.profile_config, self.logger
                )
            elif model_loader in ("colpali", "colqwen"):
                self.model, self.processor = get_or_load_model(
                    self.model_name, self.profile_config, self.logger
                )
            else:
                raise ValueError(
                    f"Unknown model_loader={model_loader!r} in profile config."
                )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings(
        self, video_data: dict[str, Any], output_dir: Path
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

        if "document_pages" in video_data:
            result = self._process_document_visual_segments(video_data, segments)
        elif "document_files" in video_data:
            result = self._process_document_segments(video_data, segments)
        elif "code_files" in video_data:
            result = self._process_code_segments(video_data, segments)
        elif "audio_files" in video_data:
            result = self._process_audio_segments(video_data, segments)
        else:
            storage_mode = video_data.get("storage_mode", self.storage_mode)

            if storage_mode == "single_doc":
                result = self._process_single_document(video_data, segments)
            else:
                result = self._process_multi_documents(video_data, segments)

        result.processing_time = time.time() - start_time
        self.logger.info(
            f"Completed {video_id}: {result.documents_processed} processed, "
            f"{result.documents_fed} fed in {result.processing_time:.2f}s"
        )

        return result

    def _extract_segments(self, video_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract segments from video_data, handling different key names.

        Segments could be under:
        - 'segments' (generic)
        - 'keyframes' (frame-based)
        - 'frames' (frame-based alternative)
        - 'chunks' (chunk-based)
        - 'video_chunks' (chunk-based alternative)
        - 'document_files' (document content)
        - 'code_files' (source-code chunks)
        - 'audio_files' (audio content)
        """
        for key in ["document_pages", "document_files", "code_files", "audio_files"]:
            if key in video_data:
                return video_data[key]

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

    @staticmethod
    def _frame_description_map(descriptions: Any) -> dict[str, str]:
        """Per-frame text map from the VLM wrapper
        ``{"video_id", "descriptions": {<frame_ref>: text}, ...}``, keyed by
        frame_number/frame_id."""
        if not isinstance(descriptions, dict):
            return {}
        inner = descriptions.get("descriptions")
        if isinstance(inner, dict):
            return inner
        return descriptions

    @staticmethod
    def _segment_frame_ref(segment: dict[str, Any]) -> str | None:
        """The keyframe's stable frame reference (matches the VLM map keys)."""
        ref = segment.get("frame_id")
        if ref is None:
            ref = segment.get("frame_number")
        return str(ref) if ref is not None else None

    def _process_multi_documents(
        self, video_data: dict[str, Any], segments: list[dict[str, Any]]
    ) -> EmbeddingResult:
        """Process segments as individual documents."""
        import gc

        video_id = video_data["video_id"]
        video_path = Path(video_data.get("video_path", ""))

        documents_processed = 0
        documents_fed = 0
        errors = []

        # Accumulate documents and feed in batches: one feed per ~50 segments
        # instead of one Vespa round-trip (+ index-wait) per segment.
        _FEED_BATCH_SIZE = 50
        batch: list[Document] = []

        def _flush_batch() -> None:
            nonlocal documents_fed
            if batch:
                documents_fed += self._feed_documents(batch)
                batch.clear()

        # Get additional data
        transcript_data = video_data.get("transcript", {})
        frame_descriptions = self._frame_description_map(
            video_data.get("descriptions", {})
        )

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

                frame_ref = self._segment_frame_ref(segment)
                description = (
                    frame_descriptions.get(frame_ref, "")
                    if frame_ref is not None
                    else ""
                )

                # Create document for this segment
                doc = self._create_segment_document(
                    video_id=video_id,
                    segment=segment,
                    segment_idx=idx,
                    total_segments=len(segments),
                    embeddings=embeddings,
                    transcript=transcript_text,
                    description=description,
                    source_url=video_data.get("source_url", ""),
                )

                documents_processed += 1

                # Queue for the next batch feed; flush at the batch boundary so
                # at most _FEED_BATCH_SIZE documents are held at once.
                batch.append(doc)

                # Drop the per-segment embedding now — the document keeps the
                # copy it needs; ColPali multi-vector output is ~370 KB/frame.
                del embeddings

                if len(batch) >= _FEED_BATCH_SIZE:
                    _flush_batch()
                    gc.collect()
                elif (idx + 1) % 20 == 0:
                    # Force a cycle collection so generational GC reclaims any
                    # PyTorch wrapper objects and the CPU allocator can return
                    # unused slabs.
                    gc.collect()

            except Exception as e:
                self.logger.error(f"Error processing segment {idx}: {e}")
                errors.append(f"Segment {idx}: {str(e)}")

        # Flush any remaining documents from the final partial batch.
        _flush_batch()

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
        self, video_data: dict[str, Any], segments: list[dict[str, Any]]
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

        # Stack embeddings. vstack copies the inputs into a new contiguous
        # array, so the per-segment arrays are immediately redundant —
        # drop them to reclaim ~370 KB/frame × segment_count.
        combined_embeddings = (
            np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        )
        all_embeddings.clear()

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

    def _process_document_segments(
        self, video_data: dict[str, Any], segments: list[dict[str, Any]]
    ) -> EmbeddingResult:
        """Process document files: encode text with ColBERT, create Documents, feed to backend."""
        content_id = video_data.get("video_id", "unknown")

        if not self.colbert_model:
            self._load_model()
        if not self.colbert_model:
            raise RuntimeError(
                f"ColBERT model not loaded for document embedding. "
                f"Expected embedding_model containing 'colbert', got: {self.model_name!r}"
            )

        documents_processed = 0
        documents_fed = 0
        errors = []

        for idx, doc_info in enumerate(segments):
            try:
                text = doc_info.get("extracted_text", "")
                if not text.strip():
                    raise ValueError(
                        f"Document {doc_info.get('filename', idx)!r} has no extracted text."
                    )

                token_embeddings = self.colbert_model.encode(
                    [text[:8192]], is_query=False
                )[0]
                embeddings_np = np.array(token_embeddings, dtype=np.float32)

                self.logger.info(
                    f"  📄 Document {doc_info.get('filename', idx)}: "
                    f"embeddings shape={embeddings_np.shape}"
                )

                doc = Document(
                    id=f"{content_id}_{doc_info.get('document_id', idx)}",
                    content_type=ContentType.DOCUMENT,
                    content_id=content_id,
                    status=ProcessingStatus.COMPLETED,
                )
                doc.add_embedding(
                    "embedding", embeddings_np, {"type": "float", "raw": True}
                )
                doc.add_metadata("document_id", doc_info.get("document_id", ""))
                doc.add_metadata("document_title", doc_info.get("filename", ""))
                doc.add_metadata("document_type", doc_info.get("document_type", ""))
                doc.add_metadata("document_path", doc_info.get("path", ""))
                doc.add_metadata("full_text", text)
                doc.add_metadata("page_count", doc_info.get("page_count", 1))
                if video_data.get("source_url"):
                    doc.add_metadata("source_url", video_data["source_url"])

                documents_processed += 1
                if self._feed_document(doc):
                    documents_fed += 1

            except Exception as e:
                self.logger.error(f"Error processing document {idx}: {e}")
                errors.append(f"Document {idx}: {str(e)}")

        return EmbeddingResult(
            video_id=content_id,
            total_documents=len(segments),
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0,
            errors=errors,
            metadata={"num_documents": len(segments)},
        )

    def _process_document_visual_segments(
        self, video_data: dict[str, Any], segments: list[dict[str, Any]]
    ) -> EmbeddingResult:
        """Process PDF page images: ColPali-embed each page, build document_visual
        Documents (one per page), feed to backend.

        Each ``segment`` is a rendered page from DocumentVisualSegmentationStrategy
        carrying ``path`` (page PNG) plus document_id/page_number/page_count/
        document_path/document_title — mapped onto the document_visual schema
        fields. The page image is embedded with ColPali via the shared
        ``_generate_frame_embeddings`` (same multi-vector path as video frames).
        """
        content_id = video_data.get("video_id", "unknown")

        documents_processed = 0
        documents_fed = 0
        errors = []

        for idx, page in enumerate(segments):
            try:
                page_path = Path(page.get("path", ""))
                if not page_path.exists():
                    raise ValueError(
                        f"Page image missing for "
                        f"{page.get('document_id', idx)!r}: {page_path}"
                    )

                embeddings_np = self._generate_frame_embeddings(page_path)
                if embeddings_np is None:
                    raise ValueError(
                        f"ColPali embedding failed for page {page_path.name}"
                    )
                embeddings_np = np.asarray(embeddings_np, dtype=np.float32)

                page_number = page.get("page_number", idx + 1)
                self.logger.info(
                    f"  🖼️ Page {page.get('document_id', idx)} p{page_number}: "
                    f"embeddings shape={embeddings_np.shape}"
                )

                doc = Document(
                    id=f"{content_id}_{page.get('document_id', idx)}_p{page_number}",
                    content_type=ContentType.DOCUMENT,
                    content_id=content_id,
                    status=ProcessingStatus.COMPLETED,
                )
                doc.add_embedding(
                    "embedding", embeddings_np, {"type": "float", "raw": True}
                )
                doc.add_metadata("document_id", page.get("document_id", ""))
                doc.add_metadata("document_title", page.get("document_title", ""))
                doc.add_metadata("document_type", page.get("document_type", "pdf"))
                doc.add_metadata("document_path", page.get("document_path", ""))
                doc.add_metadata("page_number", page_number)
                doc.add_metadata("page_count", page.get("page_count", len(segments)))

                documents_processed += 1
                if self._feed_document(doc):
                    documents_fed += 1

            except Exception as e:
                self.logger.error(f"Error processing page {idx}: {e}")
                errors.append(f"Page {idx}: {str(e)}")

        return EmbeddingResult(
            video_id=content_id,
            total_documents=len(segments),
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0,
            errors=errors,
            metadata={"num_pages": len(segments)},
        )

    def _process_code_segments(
        self, video_data: dict[str, Any], segments: list[dict[str, Any]]
    ) -> EmbeddingResult:
        """Process source-code chunks: encode the chunk text with ColBERT, build
        Documents whose metadata matches the code schema fields, feed to backend.

        Each ``segment`` is an AST chunk from CodeSegmentationStrategy carrying
        ``extracted_text`` (the chunk source) plus ``chunk_name``/``chunk_type``/
        ``signature``/``line_start``/``line_end``/``language`` — mapped onto the
        ``code_*`` schema fields (code_id, file_path, chunk_name, chunk_type,
        language, signature, line_start, line_end, source_code).
        """
        content_id = video_data.get("video_id", "unknown")

        if not self.colbert_model:
            self._load_model()
        if not self.colbert_model:
            raise RuntimeError(
                f"ColBERT model not loaded for code embedding. "
                f"Expected embedding_model containing 'colbert', got: {self.model_name!r}"
            )

        documents_processed = 0
        documents_fed = 0
        errors = []

        for idx, seg in enumerate(segments):
            try:
                text = seg.get("extracted_text", "")
                if not text.strip():
                    raise ValueError(
                        f"Code chunk {seg.get('document_id', idx)!r} has no source text."
                    )

                token_embeddings = self.colbert_model.encode(
                    [text[:8192]], is_query=False
                )[0]
                embeddings_np = np.array(token_embeddings, dtype=np.float32)

                self.logger.info(
                    f"  💻 Code chunk {seg.get('chunk_name', idx)}: "
                    f"embeddings shape={embeddings_np.shape}"
                )

                doc = Document(
                    id=f"{content_id}_{seg.get('document_id', idx)}",
                    content_type=ContentType.TEXT,
                    content_id=content_id,
                    status=ProcessingStatus.COMPLETED,
                )
                doc.add_embedding(
                    "embedding", embeddings_np, {"type": "float", "raw": True}
                )
                doc.add_metadata("code_id", seg.get("document_id", ""))
                doc.add_metadata("file_path", seg.get("path", ""))
                doc.add_metadata("chunk_name", seg.get("chunk_name", ""))
                doc.add_metadata("chunk_type", seg.get("chunk_type", ""))
                doc.add_metadata("language", seg.get("language", ""))
                doc.add_metadata("signature", seg.get("signature", ""))
                doc.add_metadata("line_start", int(seg.get("line_start", 0)))
                doc.add_metadata("line_end", int(seg.get("line_end", 0)))
                doc.add_metadata("source_code", text)

                documents_processed += 1
                if self._feed_document(doc):
                    documents_fed += 1

            except Exception as e:
                self.logger.error(f"Error processing code chunk {idx}: {e}")
                errors.append(f"Code chunk {idx}: {str(e)}")

        return EmbeddingResult(
            video_id=content_id,
            total_documents=len(segments),
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0,
            errors=errors,
            metadata={"num_documents": len(segments)},
        )

    def _process_audio_segments(
        self, video_data: dict[str, Any], segments: list[dict[str, Any]]
    ) -> EmbeddingResult:
        """Process audio files: CLAP acoustic + ColBERT semantic embeddings."""
        from ..audio_embedding_generator import AudioEmbeddingGenerator

        content_id = video_data.get("video_id", "unknown")

        if not self.colbert_model:
            self._load_model()
        if not self.colbert_model:
            raise RuntimeError(
                "ColBERT model not loaded for audio semantic embedding. "
                "Ensure 'semantic_model' is set in the audio profile config."
            )

        clap_model_name = self.profile_config.get(
            "embedding_model", "laion/clap-htsat-unfused"
        )
        audio_generator = AudioEmbeddingGenerator(clap_model=clap_model_name)

        transcript_data = video_data.get("transcript", {})
        if not isinstance(transcript_data, dict):
            raise TypeError(
                f"Expected transcript_data to be a dict, got {type(transcript_data).__name__!r}."
            )
        transcript_text = transcript_data.get("full_text", "")

        documents_processed = 0
        documents_fed = 0
        errors = []

        for idx, audio_info in enumerate(segments):
            try:
                audio_path = Path(audio_info["path"])

                # CLAP acoustic embedding (512-dim float list). Best-effort:
                # it needs torch+CLAP in-process, which the deployed runtime
                # image doesn't ship — an unavailable acoustic vector must
                # not discard the semantic/transcript chunk.
                try:
                    acoustic_emb = audio_generator.generate_acoustic_embedding(
                        audio_path=audio_path
                    )
                except Exception as exc:
                    self.logger.warning(
                        f"Acoustic embedding unavailable for "
                        f"{audio_info.get('filename', idx)}: {exc} — feeding "
                        f"semantic-only audio chunk"
                    )
                    acoustic_emb = None

                if not transcript_text.strip():
                    raise ValueError(
                        f"Audio file {audio_path.name!r} has no transcript text. "
                        "Transcription must run before audio embedding."
                    )
                semantic_tokens = self.colbert_model.encode(
                    [transcript_text[:8192]], is_query=False
                )[0]
                semantic_np = np.array(semantic_tokens, dtype=np.float32)

                acoustic_shape = None if acoustic_emb is None else acoustic_emb.shape
                self.logger.info(
                    f"  🔊 Audio {audio_info.get('filename', idx)}: "
                    f"acoustic={acoustic_shape}, semantic={semantic_np.shape}"
                )

                # Create Document with dual embeddings
                doc = Document(
                    id=f"{content_id}_{audio_info.get('audio_id', idx)}",
                    content_type=ContentType.AUDIO,
                    content_id=content_id,
                    status=ProcessingStatus.COMPLETED,
                )

                # Semantic ColBERT embedding goes through standard embedding path
                doc.add_embedding(
                    "embedding", semantic_np, {"type": "float", "raw": True}
                )

                # Acoustic CLAP embedding stored as pre-formatted Vespa field in metadata.
                # The ingestion client's generic metadata→field mapping picks this up
                # because 'acoustic_embedding' matches the schema field name.
                if acoustic_emb is not None:
                    doc.add_metadata("acoustic_embedding", acoustic_emb.tolist())

                doc.add_metadata(
                    "audio_id", audio_info.get("audio_id", audio_path.stem)
                )
                doc.add_metadata("audio_title", audio_path.stem)
                doc.add_metadata("audio_path", str(audio_path))
                doc.add_metadata("audio_transcript", transcript_text)
                if video_data.get("source_url"):
                    doc.add_metadata("source_url", video_data["source_url"])

                documents_processed += 1
                if self._feed_document(doc):
                    documents_fed += 1

            except Exception as e:
                self.logger.error(f"Error processing audio {idx}: {e}")
                errors.append(f"Audio {idx}: {str(e)}")

        return EmbeddingResult(
            video_id=content_id,
            total_documents=len(segments),
            documents_processed=documents_processed,
            documents_fed=documents_fed,
            processing_time=0,
            errors=errors,
            metadata={"num_audio_files": len(segments)},
        )

    def _generate_segment_embeddings(
        self, segment: dict[str, Any], video_path: Path, video_data: dict[str, Any]
    ) -> np.ndarray | None:
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

    def _generate_frame_embeddings(self, frame_path: Path) -> np.ndarray | None:
        """Generate embeddings for a single frame."""
        if not self.model:
            self._load_model()

        try:
            if not self.model or not self.processor:
                self.logger.error("Model or processor not loaded")
                return None

            # Remote ColPali path: RemoteColPaliLoader returns the same
            # RemoteInferenceClient as both ``model`` and ``processor``.
            # ``process_images`` ships base64 PNGs to the remote pod and
            # returns a dict of {"embeddings": np.ndarray}. The local
            # PyTorch path below would crash on it (``dict`` has no
            # ``.to()`` and the client has no ``.device``).
            from cogniverse_core.common.models.model_loaders import (
                RemoteInferenceClient,
            )

            if isinstance(self.processor, RemoteInferenceClient):
                with Image.open(frame_path) as image:
                    image = image.convert("RGB")
                    result = self.processor.process_images(
                        [image], model_name=self.model_name
                    )
                embeddings_arr = np.asarray(result.get("embeddings", []))
                if embeddings_arr.ndim == 0 or embeddings_arr.size == 0:
                    self.logger.error(
                        f"Remote inference returned empty embeddings for {frame_path.name}"
                    )
                    return None
                # Server returns one multi-vector per image: shape [B, T, D].
                # The local path below feeds a single image and returns
                # shape [T, D], so unwrap the batch dim to match.
                if embeddings_arr.ndim == 3 and embeddings_arr.shape[0] == 1:
                    embeddings_arr = embeddings_arr[0]
                self.logger.info(
                    f"    🖼️ Generated embeddings for frame {frame_path.name} "
                    f"(remote): shape={embeddings_arr.shape}"
                )
                return embeddings_arr.astype(np.float32, copy=False)

            # Context manager releases the decoded PIL buffer (~6 MB at 1080p)
            # as soon as process_images copies pixel_values into a tensor.
            # Without this, the PIL Image refcount keeps the decoded RGB bytes
            # alive for the full segment loop and bloats the process by
            # 6 MB × N_frames over an ingestion run.
            import torch

            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                batch_inputs = self.processor.process_images([image]).to(
                    self.model.device
                )

            with torch.no_grad():
                embeddings = self.model(**batch_inputs)

            embeddings_np = embeddings.cpu().to(torch.float32).numpy()

            # Drop the processor/model intermediate tensors before the next
            # frame's forward pass so PyTorch's CPU caching allocator can
            # reuse the slab. Without this the allocator holds per-frame
            # activations (hundreds of MB for ColPali-family models) across
            # the whole segment loop, accumulating into GB-scale bloat.
            del batch_inputs, embeddings

            self.logger.info(
                f"    🖼️ Generated embeddings for frame {frame_path.name}: shape={embeddings_np.shape}"
            )
            return embeddings_np

        except Exception as e:
            self.logger.error(f"Error generating frame embeddings: {e}")
            return None

    def _generate_chunk_embeddings(self, chunk_path: Path) -> np.ndarray | None:
        """Generate embeddings for a video chunk."""
        try:
            model_loader = self.profile_config.get("model_loader")
            if model_loader in ("colpali", "colqwen"):
                # ColQwen/video-chunk model processes video chunks
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

                # Remote ColPali/ColQwen path — see _generate_frame_embeddings
                # for the rationale (RemoteInferenceClient.process_images
                # returns a dict, not a tensor container).
                from cogniverse_core.common.models.model_loaders import (
                    RemoteInferenceClient,
                )

                if isinstance(self.processor, RemoteInferenceClient):
                    result = self.processor.process_images(
                        frames, model_name=self.model_name
                    )
                    frames.clear()
                    embeddings_arr = np.asarray(result.get("embeddings", []))
                    if embeddings_arr.size == 0:
                        self.logger.error(
                            "Remote inference returned empty chunk embeddings"
                        )
                        return None
                    # Remote returns shape [N_frames, T, D]; collapse to a
                    # chunk-level vector by mean-pooling over the frame dim,
                    # matching the local path (which does
                    # ``embeddings_np.mean(axis=0)``).
                    return (
                        embeddings_arr.mean(axis=0)
                        if embeddings_arr.ndim >= 2
                        else embeddings_arr
                    ).astype(np.float32, copy=False)

                import torch

                batch_inputs = self.processor.process_images(frames).to(
                    self.model.device
                )
                # Release decoded RGB buffers (up to 10 × ~6 MB) now that
                # process_images has copied them into pixel_values tensors.
                frames.clear()

                with torch.no_grad():
                    embeddings = self.model(**batch_inputs)

                embeddings_np = embeddings.cpu().numpy()
                result = embeddings_np.mean(axis=0)

                # Drop tensors so the CPU allocator can reclaim activations
                # before the caller moves on to the next chunk.
                del batch_inputs, embeddings, embeddings_np
                return result

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
                import torch

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
    ) -> np.ndarray | None:
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

                # Remote ColPali/ColQwen path — same dict-vs-tensor split as
                # _generate_frame_embeddings.
                from cogniverse_core.common.models.model_loaders import (
                    RemoteInferenceClient,
                )

                if isinstance(self.processor, RemoteInferenceClient):
                    result = self.processor.process_images(
                        frames, model_name=self.model_name
                    )
                    embeddings_arr = np.asarray(result.get("embeddings", []))
                    if embeddings_arr.size == 0:
                        return None
                    if len(frames) > 1 and embeddings_arr.ndim >= 2:
                        embeddings_arr = embeddings_arr.mean(axis=0)
                    return embeddings_arr.astype(np.float32, copy=False)

                import torch

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
        segment: dict[str, Any],
        segment_idx: int,
        total_segments: int,
        embeddings: np.ndarray,
        transcript: str = "",
        description: str = "",
        source_url: str = "",
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

        # Canonical source URI — declared in the schema; consumers resolve bytes
        # (visual evaluators / frame extraction) from it.
        if source_url:
            document.add_metadata("source_url", source_url)

        # Add description if available
        if description:
            document.add_metadata("description", description)

        return document

    def _create_combined_document(
        self,
        video_id: str,
        embeddings: np.ndarray,
        segments: list[dict[str, Any]],
        video_data: dict[str, Any],
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

        # Canonical source URI — declared in the schema; consumers resolve bytes
        # (visual evaluators / frame extraction) from it.
        source_url = video_data.get("source_url", "")
        if source_url:
            document.add_metadata("source_url", source_url)

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

    def _feed_documents(self, documents: list[Document]) -> int:
        """Feed a batch of documents in one backend call; return fed count."""
        if self.backend_client and documents:
            result = self.backend_client.ingest_documents(documents, self.schema_name)
            return int(result.get("success_count", 0))
        return 0
