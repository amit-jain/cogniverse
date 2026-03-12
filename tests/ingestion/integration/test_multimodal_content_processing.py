"""
Integration tests for multi-modal content processing pipelines.

Tests the full ProcessingStrategySet.process() round-trip for:
- Image ingestion: ImageSegmentationStrategy → MultiVectorEmbeddingStrategy
- Audio ingestion: AudioFileSegmentationStrategy → AudioTranscriptionStrategy → AudioEmbeddingStrategy
- Document ingestion: DocumentSegmentationStrategy → DocumentTextEmbeddingStrategy

Each test constructs its own test files, exercises the real dispatch path in
_process_segmentation, and mocks only external ML models (ColBERT, CLAP).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.strategies import (
    AudioEmbeddingStrategy,
    AudioFileSegmentationStrategy,
    AudioTranscriptionStrategy,
    DocumentSegmentationStrategy,
    DocumentTextEmbeddingStrategy,
    ImageSegmentationStrategy,
    MultiVectorEmbeddingStrategy,
    NoDescriptionStrategy,
    NoTranscriptionStrategy,
)


@pytest.fixture
def image_dir(tmp_path):
    """Create a directory with minimal PNG images (valid 1x1 pixel)."""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()

    # Minimal valid 8x8 PNG — smallest valid PNG structure
    # PNG signature + IHDR + IDAT (uncompressed 1x1 red pixel) + IEND
    import struct
    import zlib

    def make_png(width=1, height=1, r=255, g=0, b=0):
        """Create a minimal valid PNG file in memory."""
        signature = b"\x89PNG\r\n\x1a\n"

        # IHDR chunk
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

        # IDAT chunk — raw pixel data: filter byte (0) + RGB per pixel per row
        raw_data = b""
        for _ in range(height):
            raw_data += b"\x00"  # filter byte
            for _ in range(width):
                raw_data += bytes([r, g, b])
        compressed = zlib.compress(raw_data)
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)

        # IEND chunk
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

        return signature + ihdr + idat + iend

    (img_dir / "photo_001.png").write_bytes(make_png(r=255, g=0, b=0))
    (img_dir / "photo_002.jpg").write_bytes(make_png(r=0, g=255, b=0))  # .jpg ext, valid PNG content
    (img_dir / "photo_003.png").write_bytes(make_png(r=0, g=0, b=255))

    return img_dir


@pytest.fixture
def audio_dir(tmp_path):
    """Create a directory with dummy audio files."""
    a_dir = tmp_path / "test_audio"
    a_dir.mkdir()
    for i, ext in enumerate([".mp3", ".wav", ".flac"]):
        (a_dir / f"recording_{i:03d}{ext}").write_bytes(b"\x00" * 128)
    return a_dir


@pytest.fixture
def document_dir(tmp_path):
    """Create a directory with real document files containing extractable text."""
    doc_dir = tmp_path / "test_docs"
    doc_dir.mkdir()

    (doc_dir / "report.txt").write_text(
        "Quarterly revenue grew 15% year-over-year driven by enterprise adoption.",
        encoding="utf-8",
    )
    (doc_dir / "design.md").write_text(
        "# System Design\n\nThe architecture uses event-driven microservices.",
        encoding="utf-8",
    )

    # Minimal valid PDF with extractable text
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 72 720 Td (Hello PDF) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000362 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n436\n%%EOF"
    )
    (doc_dir / "whitepaper.pdf").write_bytes(pdf_content)

    return doc_dir



def make_pipeline_context(
    content_path: Path,
    output_dir: Path,
    *,
    transcribe_audio: bool = False,
    generate_descriptions: bool = False,
    generate_embeddings: bool = True,
):
    """Build a mock pipeline context with the fields ProcessingStrategySet.process() needs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimpleNamespace(
        transcribe_audio=transcribe_audio,
        generate_descriptions=generate_descriptions,
        generate_embeddings=generate_embeddings,
    )

    logger = MagicMock()

    ctx = SimpleNamespace(
        config=config,
        logger=logger,
        profile_output_dir=output_dir,
        video_path=content_path,
        schema_name="test_schema",
        cache=None,
        generate_embeddings=AsyncMock(return_value={"documents_fed": 3}),
        processor_manager=MagicMock(),
    )
    return ctx



@pytest.mark.integration
class TestImageIngestionPipeline:
    """Full round-trip: image dir → segmentation → embedding."""

    @pytest.mark.asyncio
    async def test_image_segmentation_produces_keyframes_for_embedding(self, image_dir, tmp_path):
        """Segmentation discovers images and formats them as keyframes that the
        embedding strategy can consume via pipeline_context.generate_embeddings."""
        strategy_set = ProcessingStrategySet(
            segmentation=ImageSegmentationStrategy(max_images=100),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=MultiVectorEmbeddingStrategy(model_name="test/model"),
        )

        output_dir = tmp_path / "output"
        ctx = make_pipeline_context(
            image_dir, output_dir, generate_embeddings=True,
        )

        # Mock the embedding strategy's generate_embeddings_with_processor so we
        # don't need a real ColPali model, but still verify it receives correct data.
        embedding_call_args = {}

        async def capture_embedding_call(results, pipeline_context, processor_manager):
            embedding_call_args["results"] = results
            return {"documents_fed": 3}

        strategy_set.embedding.generate_embeddings_with_processor = capture_embedding_call

        processor_manager = MagicMock()
        results = await strategy_set.process(image_dir, processor_manager, ctx)

        # Verify segmentation produced keyframes
        assert "keyframes" in results
        keyframes_data = results["keyframes"]
        assert keyframes_data["video_id"] == image_dir.stem
        assert len(keyframes_data["keyframes"]) == 3
        assert keyframes_data["stats"]["extraction_method"] == "image_load"

        # Verify each keyframe has required fields
        for kf in keyframes_data["keyframes"]:
            assert "frame_id" in kf
            assert "path" in kf
            assert "filename" in kf
            assert "source_image" in kf
            assert Path(kf["path"]).exists()

        # Verify embedding strategy received the segmentation output
        assert "keyframes" in embedding_call_args["results"]

        # Verify embeddings result was captured
        assert "embeddings" in results
        assert results["embeddings"]["documents_fed"] == 3

    @pytest.mark.asyncio
    async def test_image_segmentation_respects_max_images(self, image_dir, tmp_path):
        """max_images parameter limits how many images are processed."""
        strategy_set = ProcessingStrategySet(
            segmentation=ImageSegmentationStrategy(max_images=2),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=MultiVectorEmbeddingStrategy(),
        )

        output_dir = tmp_path / "output"
        ctx = make_pipeline_context(image_dir, output_dir, generate_embeddings=False)

        processor_manager = MagicMock()
        results = await strategy_set.process(image_dir, processor_manager, ctx)

        assert len(results["keyframes"]["keyframes"]) == 2

    @pytest.mark.asyncio
    async def test_image_single_file_processing(self, image_dir, tmp_path):
        """A single image file path works directly."""
        single_image = image_dir / "photo_001.png"

        strategy_set = ProcessingStrategySet(
            segmentation=ImageSegmentationStrategy(),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=MultiVectorEmbeddingStrategy(),
        )

        output_dir = tmp_path / "output"
        ctx = make_pipeline_context(single_image, output_dir, generate_embeddings=False)

        processor_manager = MagicMock()
        results = await strategy_set.process(single_image, processor_manager, ctx)

        assert len(results["keyframes"]["keyframes"]) == 1
        assert results["keyframes"]["keyframes"][0]["source_image"] == str(single_image)

    @pytest.mark.asyncio
    async def test_image_non_image_file_raises(self, tmp_path):
        """Passing a non-image file raises ValueError."""
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b,c")

        strategy_set = ProcessingStrategySet(
            segmentation=ImageSegmentationStrategy(),
        )

        ctx = make_pipeline_context(bad_file, tmp_path / "out", generate_embeddings=False)

        with pytest.raises(ValueError, match="Expected image file or directory"):
            await strategy_set.process(bad_file, MagicMock(), ctx)

    @pytest.mark.asyncio
    async def test_image_empty_dir_raises(self, tmp_path):
        """Empty directory raises ValueError."""
        empty = tmp_path / "empty"
        empty.mkdir()

        strategy_set = ProcessingStrategySet(
            segmentation=ImageSegmentationStrategy(),
        )

        ctx = make_pipeline_context(empty, tmp_path / "out", generate_embeddings=False)

        with pytest.raises(ValueError, match="No image files found"):
            await strategy_set.process(empty, MagicMock(), ctx)



@pytest.mark.integration
class TestAudioIngestionPipeline:
    """Full round-trip: audio dir → segmentation → transcription → embedding."""

    @pytest.mark.asyncio
    async def test_audio_pipeline_data_flow(self, audio_dir, tmp_path):
        """Segmentation discovers audio files, transcription processes them,
        and embedding receives both segmentation + transcription results."""
        strategy_set = ProcessingStrategySet(
            segmentation=AudioFileSegmentationStrategy(max_files=100),
            transcription=AudioTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=AudioEmbeddingStrategy(),
        )

        output_dir = tmp_path / "output"
        ctx = make_pipeline_context(
            audio_dir, output_dir,
            transcribe_audio=True,
            generate_embeddings=True,
        )

        # Mock audio processor for transcription
        mock_audio_processor = MagicMock()
        mock_audio_processor.transcribe_audio.return_value = {
            "full_text": "This is a test transcription of the audio files.",
            "segments": [{"start": 0.0, "end": 5.0, "text": "This is a test"}],
            "language": "en",
        }

        processor_manager = MagicMock()
        processor_manager.get_processor.return_value = mock_audio_processor

        # Capture what embedding receives
        embedding_call_args = {}

        async def capture_embedding(results, pipeline_context, processor_manager):
            embedding_call_args.update(results)
            return {"documents_fed": 3}

        strategy_set.embedding.generate_embeddings_with_processor = capture_embedding

        results = await strategy_set.process(audio_dir, processor_manager, ctx)

        # Verify segmentation discovered all audio files
        assert "audio_files" in results
        assert len(results["audio_files"]) == 3
        for af in results["audio_files"]:
            assert "audio_id" in af
            assert "path" in af
            assert "filename" in af
            assert Path(af["path"]).exists()

        # Verify transcription flowed through
        assert "transcript" in results

        # Verify embedding received both audio_files and transcript
        assert "audio_files" in embedding_call_args
        assert "transcript" in embedding_call_args

    @pytest.mark.asyncio
    async def test_audio_segmentation_only(self, audio_dir, tmp_path):
        """Segmentation alone produces correct audio file list."""
        strategy_set = ProcessingStrategySet(
            segmentation=AudioFileSegmentationStrategy(),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=AudioEmbeddingStrategy(),
        )

        output_dir = tmp_path / "output"
        ctx = make_pipeline_context(
            audio_dir, output_dir,
            transcribe_audio=False,
            generate_embeddings=False,
        )

        results = await strategy_set.process(audio_dir, MagicMock(), ctx)

        assert "audio_files" in results
        filenames = {af["filename"] for af in results["audio_files"]}
        assert filenames == {"recording_000.mp3", "recording_001.wav", "recording_002.flac"}

    @pytest.mark.asyncio
    async def test_audio_single_file_processing(self, audio_dir, tmp_path):
        """Single audio file works directly."""
        single = audio_dir / "recording_000.mp3"

        strategy_set = ProcessingStrategySet(
            segmentation=AudioFileSegmentationStrategy(),
        )

        output_dir = tmp_path / "output"
        ctx = make_pipeline_context(single, output_dir, generate_embeddings=False)

        results = await strategy_set.process(single, MagicMock(), ctx)

        assert len(results["audio_files"]) == 1
        assert results["audio_files"][0]["filename"] == "recording_000.mp3"

    @pytest.mark.asyncio
    async def test_audio_respects_max_files(self, audio_dir, tmp_path):
        """max_files limits discovery."""
        strategy_set = ProcessingStrategySet(
            segmentation=AudioFileSegmentationStrategy(max_files=1),
        )

        ctx = make_pipeline_context(audio_dir, tmp_path / "out", generate_embeddings=False)

        results = await strategy_set.process(audio_dir, MagicMock(), ctx)
        assert len(results["audio_files"]) == 1

    @pytest.mark.asyncio
    async def test_audio_non_audio_file_raises(self, tmp_path):
        """Non-audio file raises ValueError."""
        bad = tmp_path / "image.png"
        bad.write_bytes(b"\x89PNG" + b"\x00" * 64)

        strategy_set = ProcessingStrategySet(
            segmentation=AudioFileSegmentationStrategy(),
        )

        ctx = make_pipeline_context(bad, tmp_path / "out", generate_embeddings=False)

        with pytest.raises(ValueError, match="Expected audio file or directory"):
            await strategy_set.process(bad, MagicMock(), ctx)

    @pytest.mark.asyncio
    async def test_audio_empty_dir_raises(self, tmp_path):
        """Empty directory raises ValueError."""
        empty = tmp_path / "empty"
        empty.mkdir()

        strategy_set = ProcessingStrategySet(
            segmentation=AudioFileSegmentationStrategy(),
        )

        ctx = make_pipeline_context(empty, tmp_path / "out", generate_embeddings=False)

        with pytest.raises(ValueError, match="No audio files found"):
            await strategy_set.process(empty, MagicMock(), ctx)



@pytest.mark.integration
class TestDocumentIngestionPipeline:
    """Full round-trip: document dir → segmentation (with text extraction) → embedding."""

    @pytest.mark.asyncio
    async def test_document_pipeline_data_flow(self, document_dir, tmp_path):
        """Segmentation discovers documents, extracts text, and embedding
        receives the extracted text for ColBERT tokenization."""
        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(max_files=100),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=DocumentTextEmbeddingStrategy(),
        )

        output_dir = tmp_path / "output"
        ctx = make_pipeline_context(
            document_dir, output_dir, generate_embeddings=True,
        )

        # Capture what embedding receives
        embedding_call_args = {}

        async def capture_embedding(results, pipeline_context, processor_manager):
            embedding_call_args.update(results)
            return {"documents_fed": 3}

        strategy_set.embedding.generate_embeddings_with_processor = capture_embedding

        results = await strategy_set.process(document_dir, MagicMock(), ctx)

        # Verify segmentation discovered all documents
        assert "document_files" in results
        assert len(results["document_files"]) == 3

        # Verify text was extracted from each document
        doc_by_name = {df["filename"]: df for df in results["document_files"]}

        assert "Quarterly revenue" in doc_by_name["report.txt"]["extracted_text"]
        assert doc_by_name["report.txt"]["document_type"] == "txt"

        assert "# System Design" in doc_by_name["design.md"]["extracted_text"]
        assert doc_by_name["design.md"]["document_type"] == "md"

        assert doc_by_name["whitepaper.pdf"]["document_type"] == "pdf"
        assert len(doc_by_name["whitepaper.pdf"]["extracted_text"]) > 0

        # Verify each document has all required fields
        for df in results["document_files"]:
            assert "document_id" in df
            assert "path" in df
            assert "filename" in df
            assert "extracted_text" in df
            assert "text_length" in df
            assert df["text_length"] > 0

        # Verify embedding received the document_files data
        assert "document_files" in embedding_call_args

        # Verify embeddings result was captured
        assert "embeddings" in results

    @pytest.mark.asyncio
    async def test_document_segmentation_only(self, document_dir, tmp_path):
        """Segmentation produces correct document file list with extracted text."""
        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=DocumentTextEmbeddingStrategy(),
        )

        ctx = make_pipeline_context(
            document_dir, tmp_path / "out", generate_embeddings=False,
        )

        results = await strategy_set.process(document_dir, MagicMock(), ctx)

        assert "document_files" in results
        filenames = {df["filename"] for df in results["document_files"]}
        assert filenames == {"report.txt", "design.md", "whitepaper.pdf"}

    @pytest.mark.asyncio
    async def test_document_single_file(self, document_dir, tmp_path):
        """Single document file works directly."""
        single = document_dir / "report.txt"

        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(),
        )

        ctx = make_pipeline_context(single, tmp_path / "out", generate_embeddings=False)

        results = await strategy_set.process(single, MagicMock(), ctx)

        assert len(results["document_files"]) == 1
        assert results["document_files"][0]["filename"] == "report.txt"
        assert "Quarterly revenue" in results["document_files"][0]["extracted_text"]

    @pytest.mark.asyncio
    async def test_document_respects_max_files(self, document_dir, tmp_path):
        """max_files limits discovery."""
        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(max_files=2),
        )

        ctx = make_pipeline_context(document_dir, tmp_path / "out", generate_embeddings=False)

        results = await strategy_set.process(document_dir, MagicMock(), ctx)
        assert len(results["document_files"]) == 2

    @pytest.mark.asyncio
    async def test_document_non_document_file_raises(self, tmp_path):
        """Non-document file raises ValueError."""
        bad = tmp_path / "video.mp4"
        bad.write_bytes(b"\x00" * 64)

        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(),
        )

        ctx = make_pipeline_context(bad, tmp_path / "out", generate_embeddings=False)

        with pytest.raises(ValueError, match="Expected document file or directory"):
            await strategy_set.process(bad, MagicMock(), ctx)

    @pytest.mark.asyncio
    async def test_document_empty_dir_raises(self, tmp_path):
        """Empty directory raises ValueError."""
        empty = tmp_path / "empty"
        empty.mkdir()

        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(),
        )

        ctx = make_pipeline_context(empty, tmp_path / "out", generate_embeddings=False)

        with pytest.raises(ValueError, match="No document files found"):
            await strategy_set.process(empty, MagicMock(), ctx)

    @pytest.mark.asyncio
    async def test_pdf_text_extraction_round_trip(self, document_dir, tmp_path):
        """PDF text extraction produces non-empty text that flows to embedding."""
        pdf_file = document_dir / "whitepaper.pdf"

        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=DocumentTextEmbeddingStrategy(),
        )

        embedding_received = {}

        async def capture(results, pipeline_context, processor_manager):
            embedding_received.update(results)
            return {"documents_fed": 1}

        strategy_set.embedding.generate_embeddings_with_processor = capture

        ctx = make_pipeline_context(pdf_file, tmp_path / "out", generate_embeddings=True)

        await strategy_set.process(pdf_file, MagicMock(), ctx)

        # Verify PDF text was extracted and reached embedding
        doc = embedding_received["document_files"][0]
        assert doc["document_type"] == "pdf"
        assert "Hello PDF" in doc["extracted_text"]



@pytest.mark.integration
class TestStrategyConfigurationPropagation:
    """Verify that strategy parameters propagate correctly through ProcessorManager."""

    def test_image_strategy_config_reaches_processor_manager(self):
        """ImageSegmentationStrategy config is visible to ProcessorManager."""
        strategy_set = ProcessingStrategySet(
            segmentation=ImageSegmentationStrategy(max_images=42),
            embedding=MultiVectorEmbeddingStrategy(model_name="test/colpali"),
        )

        all_reqs = strategy_set.get_all_required_processors()
        assert all_reqs["image"]["max_images"] == 42
        assert all_reqs["embedding"]["model_name"] == "test/colpali"

    def test_audio_strategy_config_reaches_processor_manager(self):
        """AudioFileSegmentationStrategy + AudioEmbeddingStrategy configs propagate."""
        strategy_set = ProcessingStrategySet(
            segmentation=AudioFileSegmentationStrategy(max_files=77),
            transcription=AudioTranscriptionStrategy(model="whisper-tiny"),
            description=NoDescriptionStrategy(),
            embedding=AudioEmbeddingStrategy(
                clap_model="custom/clap",
                colbert_model="custom/colbert",
            ),
        )

        all_reqs = strategy_set.get_all_required_processors()
        assert all_reqs["audio_file"]["max_files"] == 77
        assert all_reqs["audio"]["model"] == "whisper-tiny"
        assert all_reqs["embedding"]["clap_model"] == "custom/clap"
        assert all_reqs["embedding"]["colbert_model"] == "custom/colbert"

    def test_document_strategy_config_reaches_processor_manager(self):
        """DocumentSegmentationStrategy + DocumentTextEmbeddingStrategy configs propagate."""
        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(max_files=500),
            embedding=DocumentTextEmbeddingStrategy(colbert_model="custom/colbert"),
        )

        all_reqs = strategy_set.get_all_required_processors()
        assert all_reqs["document_file"]["max_files"] == 500
        assert all_reqs["embedding"]["colbert_model"] == "custom/colbert"

    def test_processor_manager_rejects_unknown_processor_types(self):
        """ProcessorManager raises ValueError for processor types without registered classes.

        Document/audio/image segmentation is handled inline in _process_segmentation,
        NOT via ProcessorManager. So ProcessorManager correctly rejects these unknown types.
        """
        strategy_set = ProcessingStrategySet(
            segmentation=DocumentSegmentationStrategy(),
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=DocumentTextEmbeddingStrategy(),
        )

        with patch(
            "cogniverse_runtime.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []
            logger = MagicMock()
            manager = ProcessorManager(logger)
            with pytest.raises(ValueError, match="Unknown processor type: document_file"):
                manager.initialize_from_strategies(strategy_set)
