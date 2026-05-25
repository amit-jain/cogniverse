"""
Integration tests for schema-driven evaluation pipeline.

These tests verify that the evaluation system actually works with
different schema types without hardcoded assumptions.
"""

from unittest.mock import Mock

import pytest

from cogniverse_evaluation.core.ground_truth import SchemaAwareGroundTruthStrategy
from cogniverse_evaluation.core.schema_analyzer import (
    DefaultSchemaAnalyzer,
    get_schema_analyzer,
    register_analyzer,
)
from cogniverse_evaluation.plugins.document_analyzer import (
    DocumentSchemaAnalyzer,
    ImageSchemaAnalyzer,
)
from cogniverse_evaluation.plugins.video_analyzer import VideoSchemaAnalyzer


# Dual-marked so the CI ``unit-tests`` job (filter ``-m unit``) AND
# the CI ``integration-tests`` job (filter ``-m "integration and
# ci_fast"``) both pick it up. The schema-analyzer pipeline is a
# multi-component integration that runs mock-only — fits both
# semantics.
@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.ci_fast
class TestSchemaAnalyzerIntegration:
    """Test that schema analyzers correctly identify and process different schemas."""

    def setup_method(self):
        """Register analyzers for testing."""
        register_analyzer(VideoSchemaAnalyzer())
        register_analyzer(DocumentSchemaAnalyzer())
        register_analyzer(ImageSchemaAnalyzer())

    def test_video_schema_recognition(self):
        """Test that video schemas are correctly identified."""
        schema_name = "video_frames"
        schema_fields = {
            "id_fields": ["video_id", "frame_id"],
            "temporal_fields": ["start_time", "end_time"],
            "content_fields": ["frame_description", "transcript"],
        }

        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, VideoSchemaAnalyzer)
        assert analyzer.get_expected_field_name() == "expected_videos"

        # Test ID extraction for video
        video_doc = type(
            "obj", (object,), {"metadata": {"video_id": "VID123_frame_001"}}
        )()

        extracted_id = analyzer.extract_item_id(video_doc)
        assert extracted_id == "VID123"  # Should extract base video ID

    def test_document_schema_recognition(self):
        """Test that document schemas are correctly identified."""
        schema_name = "document_index"
        schema_fields = {
            "id_fields": ["document_id", "page_id"],
            "content_fields": ["content", "abstract"],
            "metadata_fields": ["author", "title", "publication_date"],
        }

        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, DocumentSchemaAnalyzer)
        assert analyzer.get_expected_field_name() == "expected_documents"

        # Test query analysis for documents
        query = 'author:"John Doe" machine learning after:2020-01-01'
        constraints = analyzer.analyze_query(query, schema_fields)

        assert constraints["query_type"] == "document_author"
        assert "John" in constraints["author_constraints"]["author"]
        assert "after_date" in constraints["date_constraints"]

    def test_image_schema_recognition(self):
        """Test that image schemas are correctly identified."""
        schema_name = "image_collection"
        schema_fields = {
            "id_fields": ["image_id"],
            "content_fields": ["caption", "alt_text"],
            "metadata_fields": ["width", "height", "format"],
        }

        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, ImageSchemaAnalyzer)
        assert analyzer.get_expected_field_name() == "expected_images"

        # Test query analysis for images
        query = "red sunset landscape larger than 1920"
        constraints = analyzer.analyze_query(query, schema_fields)

        assert constraints["query_type"] == "image_style"
        assert "red" in constraints["visual_constraints"]["colors"]
        assert "landscape" in constraints["visual_constraints"]["styles"]

    def test_unknown_schema_uses_default(self):
        """Test that unknown schemas fall back to default analyzer."""
        schema_name = "custom_data"
        schema_fields = {"id_fields": ["item_id"], "content_fields": ["description"]}

        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, DefaultSchemaAnalyzer)
        assert analyzer.get_expected_field_name() == "expected_items"


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.ci_fast
class TestGroundTruthIntegration:
    """Test ground truth extraction with different schemas."""

    @pytest.mark.asyncio
    async def test_ground_truth_extraction_with_backend(self):
        """Test that ground truth extraction uses schema analyzer correctly."""
        strategy = SchemaAwareGroundTruthStrategy()

        # Mock backend
        backend = Mock()
        backend.schema_name = "documents"
        backend.get_schema = Mock(
            return_value={
                "name": "documents",
                "fields": {
                    "id_fields": ["document_id"],
                    "content_fields": ["content", "title"],
                },
            }
        )

        # Mock search results
        search_results = [
            Mock(metadata={"document_id": "DOC001"}),
            Mock(metadata={"document_id": "DOC002"}),
        ]
        backend.search = Mock(return_value=search_results)

        # Register document analyzer
        register_analyzer(DocumentSchemaAnalyzer())

        # Extract ground truth
        trace_data = {"query": "machine learning", "metadata": {"schema": "documents"}}

        result = await strategy.extract_ground_truth(trace_data, backend)

        assert "expected_items" in result
        assert "expected_documents" in result  # Schema-specific field
        assert result["source"] == "schema_aware_backend"
        assert result["confidence"] > 0
        assert len(result["expected_items"]) == 2

    @pytest.mark.asyncio
    async def test_ground_truth_without_backend(self):
        """Test ground truth extraction fails properly without backend."""
        strategy = SchemaAwareGroundTruthStrategy()

        trace_data = {"query": "test query"}
        result = await strategy.extract_ground_truth(trace_data, None)

        assert result["expected_items"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "no_backend"
