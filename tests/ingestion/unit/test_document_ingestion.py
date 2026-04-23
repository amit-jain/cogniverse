"""
Tests for document ingestion pipeline components.

Validates:
1. DocumentSegmentationStrategy produces document file list with extracted text
2. DocumentTextEmbeddingStrategy processor requirements
3. Document profile loads correctly from config
4. StrategyFactory creates document strategy set
5. ProcessingStrategySet handles document_file segmentation dispatch
6. Text extraction for PDF, TXT, and MD files
"""

import json
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.strategies import (
    DocumentSegmentationStrategy,
    DocumentTextEmbeddingStrategy,
    NoDescriptionStrategy,
    NoTranscriptionStrategy,
)
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory


@pytest.fixture
def document_dir(tmp_path):
    """Create a temporary directory with test document files."""
    doc_dir = tmp_path / "test_docs"
    doc_dir.mkdir()

    (doc_dir / "readme.txt").write_text(
        "This is a plain text document.", encoding="utf-8"
    )
    (doc_dir / "notes.md").write_text(
        "# Notes\n\nSome markdown content.", encoding="utf-8"
    )
    (doc_dir / "report.rtf").write_text("RTF content here.", encoding="utf-8")

    return doc_dir


@pytest.fixture
def pdf_document(tmp_path):
    """Create a minimal PDF file with extractable text using raw PDF syntax."""
    pdf_path = tmp_path / "test.pdf"
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 57>>stream\n"
        b"BT /F1 12 Tf 72 720 Td (test PDF content for ingestion) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000375 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n449\n%%EOF"
    )
    pdf_path.write_bytes(pdf_content)
    return pdf_path


class TestDocumentSegmentationStrategy:
    def test_get_required_processors(self):
        strategy = DocumentSegmentationStrategy(max_files=50)
        processors = strategy.get_required_processors()
        assert "document_file" in processors
        assert processors["document_file"]["max_files"] == 50

    def test_default_max_files(self):
        strategy = DocumentSegmentationStrategy()
        processors = strategy.get_required_processors()
        assert processors["document_file"]["max_files"] == 10000


class TestDocumentTextEmbeddingStrategy:
    def test_get_required_processors(self):
        strategy = DocumentTextEmbeddingStrategy()
        processors = strategy.get_required_processors()
        assert "embedding" in processors
        assert processors["embedding"]["type"] == "document_text"
        assert processors["embedding"]["colbert_model"] == "lightonai/LateOn"

    def test_custom_model(self):
        strategy = DocumentTextEmbeddingStrategy(colbert_model="custom/colbert")
        processors = strategy.get_required_processors()
        assert processors["embedding"]["colbert_model"] == "custom/colbert"


class TestDocumentProfileConfig:
    def test_document_profile_exists_in_config(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profiles = config["backend"]["profiles"]
        assert "document_text_semantic" in profiles

    def test_document_profile_type_is_document(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profile = config["backend"]["profiles"]["document_text_semantic"]
        assert profile["type"] == "document"

    def test_document_profile_strategies(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profile = config["backend"]["profiles"]["document_text_semantic"]
        strategies = profile["strategies"]
        assert strategies["segmentation"]["class"] == "DocumentSegmentationStrategy"
        assert strategies["transcription"]["class"] == "NoTranscriptionStrategy"
        assert strategies["description"]["class"] == "NoDescriptionStrategy"
        assert strategies["embedding"]["class"] == "DocumentTextEmbeddingStrategy"

    def test_document_profile_disables_audio(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        pipeline_config = config["backend"]["profiles"]["document_text_semantic"][
            "pipeline_config"
        ]
        assert pipeline_config["transcribe_audio"] is False
        assert pipeline_config["generate_descriptions"] is False
        assert pipeline_config["generate_embeddings"] is True

    def test_document_profile_schema_config(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        schema_config = config["backend"]["profiles"]["document_text_semantic"][
            "schema_config"
        ]
        assert schema_config["embedding_dim"] == 128
        assert schema_config["binary_dim"] == 16


class TestDocumentSchemaFile:
    def test_document_schema_exists(self):
        schema_path = Path("configs/schemas/document_text_schema.json")
        assert schema_path.exists()

    def test_document_schema_fields(self):
        with open("configs/schemas/document_text_schema.json") as f:
            schema = json.load(f)
        field_names = [field["name"] for field in schema["document"]["fields"]]
        assert "document_id" in field_names
        assert "document_title" in field_names
        assert "document_type" in field_names
        assert "document_path" in field_names
        assert "page_count" in field_names
        assert "full_text" in field_names
        assert "section_headings" in field_names
        assert "embedding" in field_names
        assert "embedding_binary" in field_names

    def test_document_schema_embedding_dimensions(self):
        with open("configs/schemas/document_text_schema.json") as f:
            schema = json.load(f)
        fields = {f["name"]: f for f in schema["document"]["fields"]}
        assert fields["embedding"]["type"] == "tensor<bfloat16>(token{}, v[128])"
        assert fields["embedding_binary"]["type"] == "tensor<int8>(token{}, v[16])"

    def test_document_schema_colbert_index(self):
        with open("configs/schemas/document_text_schema.json") as f:
            schema = json.load(f)
        fields = {f["name"]: f for f in schema["document"]["fields"]}
        assert "index" in fields["embedding_binary"]["indexing"]
        assert fields["embedding"]["indexing"] == ["attribute"]

    def test_document_schema_bm25_fields(self):
        with open("configs/schemas/document_text_schema.json") as f:
            schema = json.load(f)
        fields = {f["name"]: f for f in schema["document"]["fields"]}
        assert fields["full_text"]["index"] == "enable-bm25"
        assert fields["document_title"]["index"] == "enable-bm25"
        assert fields["section_headings"]["index"] == "enable-bm25"

    def test_document_schema_rank_profiles(self):
        with open("configs/schemas/document_text_schema.json") as f:
            schema = json.load(f)
        profile_names = [rp["name"] for rp in schema["rank_profiles"]]
        assert "default" in profile_names
        assert "bm25_only" in profile_names
        assert "float_float" in profile_names
        assert "binary_binary" in profile_names
        assert "float_binary" in profile_names
        assert "phased" in profile_names
        assert "hybrid_float_bm25" in profile_names
        assert "hybrid_binary_bm25" in profile_names


class TestStrategyFactoryDocumentProfile:
    def test_factory_creates_document_strategy_set(self):
        profile_config = {
            "strategies": {
                "segmentation": {
                    "class": "DocumentSegmentationStrategy",
                    "params": {"max_files": 100},
                },
                "transcription": {"class": "NoTranscriptionStrategy", "params": {}},
                "description": {"class": "NoDescriptionStrategy", "params": {}},
                "embedding": {"class": "DocumentTextEmbeddingStrategy", "params": {}},
            }
        }
        strategy_set = StrategyFactory.create_from_profile_config(profile_config)
        assert isinstance(strategy_set.segmentation, DocumentSegmentationStrategy)
        assert isinstance(strategy_set.transcription, NoTranscriptionStrategy)
        assert isinstance(strategy_set.description, NoDescriptionStrategy)
        assert isinstance(strategy_set.embedding, DocumentTextEmbeddingStrategy)


class TestDocumentSegmentationDispatch:
    """Test that ProcessingStrategySet correctly dispatches document_file segmentation."""

    @pytest.mark.asyncio
    async def test_document_segmentation_produces_file_list(self, document_dir):
        strategy = DocumentSegmentationStrategy(max_files=100)

        class MockContext:
            profile_output_dir = document_dir.parent / "output"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                    "warning": staticmethod(lambda msg: None),
                    "error": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)

        strategy_set = ProcessingStrategySet(
            segmentation=strategy,
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=DocumentTextEmbeddingStrategy(),
        )

        result = await strategy_set._process_segmentation(
            strategy, document_dir, None, MockContext()
        )

        assert "document_files" in result
        doc_files = result["document_files"]
        assert len(doc_files) == 3

        for df in doc_files:
            assert "document_id" in df
            assert "path" in df
            assert "filename" in df
            assert "extracted_text" in df
            assert "document_type" in df
            assert "text_length" in df
            assert len(df["extracted_text"]) > 0

    @pytest.mark.asyncio
    async def test_document_segmentation_extracts_text(self, document_dir):
        strategy = DocumentSegmentationStrategy()

        class MockContext:
            profile_output_dir = document_dir.parent / "output2"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        result = await strategy_set._process_segmentation(
            strategy, document_dir, None, MockContext()
        )

        doc_by_name = {df["filename"]: df for df in result["document_files"]}
        assert (
            "This is a plain text document."
            in doc_by_name["readme.txt"]["extracted_text"]
        )
        assert "# Notes" in doc_by_name["notes.md"]["extracted_text"]
        assert doc_by_name["readme.txt"]["document_type"] == "txt"
        assert doc_by_name["notes.md"]["document_type"] == "md"

    @pytest.mark.asyncio
    async def test_document_segmentation_respects_max_files(self, document_dir):
        strategy = DocumentSegmentationStrategy(max_files=1)

        class MockContext:
            profile_output_dir = document_dir.parent / "output3"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        result = await strategy_set._process_segmentation(
            strategy, document_dir, None, MockContext()
        )

        assert len(result["document_files"]) == 1

    @pytest.mark.asyncio
    async def test_document_segmentation_single_file(self, document_dir):
        strategy = DocumentSegmentationStrategy()
        single_file = document_dir / "readme.txt"

        class MockContext:
            profile_output_dir = document_dir.parent / "output4"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        result = await strategy_set._process_segmentation(
            strategy, single_file, None, MockContext()
        )

        assert len(result["document_files"]) == 1
        assert result["document_files"][0]["filename"] == "readme.txt"
        assert "plain text" in result["document_files"][0]["extracted_text"]

    @pytest.mark.asyncio
    async def test_document_segmentation_empty_dir_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        strategy = DocumentSegmentationStrategy()

        class MockContext:
            profile_output_dir = tmp_path / "output5"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        with pytest.raises(ValueError, match="No document files found"):
            await strategy_set._process_segmentation(
                strategy, empty_dir, None, MockContext()
            )

    @pytest.mark.asyncio
    async def test_document_segmentation_non_document_file_raises(self, tmp_path):
        bad_file = tmp_path / "audio.mp3"
        bad_file.write_bytes(b"\x00" * 64)
        strategy = DocumentSegmentationStrategy()

        class MockContext:
            profile_output_dir = tmp_path / "output6"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        with pytest.raises(ValueError, match="Expected document file or directory"):
            await strategy_set._process_segmentation(
                strategy, bad_file, None, MockContext()
            )

    @pytest.mark.asyncio
    async def test_pdf_text_extraction(self, pdf_document):
        strategy = DocumentSegmentationStrategy()

        class MockContext:
            profile_output_dir = pdf_document.parent / "output_pdf"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        result = await strategy_set._process_segmentation(
            strategy, pdf_document, None, MockContext()
        )

        assert len(result["document_files"]) == 1
        doc = result["document_files"][0]
        assert doc["document_type"] == "pdf"
        assert "test PDF content" in doc["extracted_text"]


class TestDocumentTextExtraction:
    """Test the _extract_document_text static method directly."""

    def test_extract_txt(self, document_dir):
        text = ProcessingStrategySet._extract_document_text(document_dir / "readme.txt")
        assert text == "This is a plain text document."

    def test_extract_md(self, document_dir):
        text = ProcessingStrategySet._extract_document_text(document_dir / "notes.md")
        assert "# Notes" in text
        assert "markdown content" in text

    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported document type"):
            ProcessingStrategySet._extract_document_text(bad)

    def test_docx_not_installed_raises(self, tmp_path):
        docx_file = tmp_path / "file.docx"
        docx_file.write_bytes(b"data")
        with pytest.raises(ValueError, match="python-docx"):
            ProcessingStrategySet._extract_document_text(docx_file)
