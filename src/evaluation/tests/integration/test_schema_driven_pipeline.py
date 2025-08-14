"""
Integration tests for schema-driven evaluation pipeline.

These tests verify that the evaluation system actually works with
different schema types without hardcoded assumptions.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import pandas as pd

from src.evaluation.core.schema_analyzer import (
    get_schema_analyzer, 
    register_analyzer,
    DefaultSchemaAnalyzer
)
from src.evaluation.plugins.video_analyzer import VideoSchemaAnalyzer
from src.evaluation.plugins.document_analyzer import DocumentSchemaAnalyzer, ImageSchemaAnalyzer
from src.evaluation.core.scorers import precision_scorer, recall_scorer, diversity_scorer
from src.evaluation.core.ground_truth import SchemaAwareGroundTruthStrategy


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
            "content_fields": ["frame_description", "transcript"]
        }
        
        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, VideoSchemaAnalyzer)
        assert analyzer.get_expected_field_name() == "expected_videos"
        
        # Test ID extraction for video
        video_doc = type('obj', (object,), {
            'metadata': {'video_id': 'VID123_frame_001'}
        })()
        
        extracted_id = analyzer.extract_item_id(video_doc)
        assert extracted_id == 'VID123'  # Should extract base video ID
    
    def test_document_schema_recognition(self):
        """Test that document schemas are correctly identified."""
        schema_name = "document_index"
        schema_fields = {
            "id_fields": ["document_id", "page_id"],
            "content_fields": ["content", "abstract"],
            "metadata_fields": ["author", "title", "publication_date"]
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
            "metadata_fields": ["width", "height", "format"]
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
        schema_fields = {
            "id_fields": ["item_id"],
            "content_fields": ["description"]
        }
        
        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, DefaultSchemaAnalyzer)
        assert analyzer.get_expected_field_name() == "expected_items"


class TestScorerIntegration:
    """Test that scorers work correctly with different schemas."""
    
    @pytest.mark.asyncio
    async def test_precision_scorer_with_video_schema(self):
        """Test precision scorer correctly uses video schema analyzer."""
        # Setup state with video schema
        state = Mock()
        state.input = {"query": "person walking"}
        state.output = {"expected_items": ["VID001", "VID002"]}
        state.metadata = {
            "schema_name": "video_frames",
            "schema_fields": {
                "id_fields": ["video_id", "frame_id"],
                "temporal_fields": ["timestamp"]
            }
        }
        
        # Mock search results
        state.outputs = {
            "config1": {
                "success": True,
                "results": [
                    {"video_id": "VID001_frame_001"},
                    {"video_id": "VID002_frame_005"},
                    {"video_id": "VID003_frame_002"}  # Not in expected
                ]
            }
        }
        
        # Register video analyzer
        register_analyzer(VideoSchemaAnalyzer())
        
        # Score
        scorer = precision_scorer()
        score = await scorer(state)
        
        assert score.value == pytest.approx(2/3)  # 2 out of 3 retrieved are relevant
        assert "schema: video_frames" in score.explanation
        assert score.metadata["schema"] == "video_frames"
    
    @pytest.mark.asyncio
    async def test_recall_scorer_with_document_schema(self):
        """Test recall scorer correctly uses document schema analyzer."""
        # Setup state with document schema
        state = Mock()
        state.input = {"query": "machine learning papers"}
        state.output = {"expected_items": ["DOC001", "DOC002", "DOC003"]}
        state.metadata = {
            "schema_name": "documents",
            "schema_fields": {
                "id_fields": ["document_id", "doc_id"]
            }
        }
        
        # Mock search results (only found 2 of 3 expected)
        state.outputs = {
            "config1": {
                "success": True,
                "results": [
                    {"document_id": "DOC001"},
                    {"document_id": "DOC003"}
                ]
            }
        }
        
        # Register document analyzer
        register_analyzer(DocumentSchemaAnalyzer())
        
        # Score
        scorer = recall_scorer()
        score = await scorer(state)
        
        # Debug output
        print(f"Score value: {score.value}")
        print(f"Score explanation: {score.explanation}")
        print(f"Score metadata: {score.metadata}")
        
        assert score.value == pytest.approx(2/3)  # Found 2 out of 3 expected
        assert "schema: documents" in score.explanation
        assert score.metadata["expected_count"] == 3
    
    @pytest.mark.asyncio
    async def test_diversity_scorer_with_any_schema(self):
        """Test diversity scorer adapts to any schema."""
        # Setup state with generic schema
        state = Mock()
        state.metadata = {
            "schema_name": "products",
            "schema_fields": {
                "id_fields": ["product_id", "sku"]
            }
        }
        
        # Mock search results with duplicates
        state.outputs = {
            "config1": {
                "success": True,
                "results": [
                    {"product_id": "PROD001"},
                    {"product_id": "PROD002"},
                    {"product_id": "PROD001"},  # Duplicate
                    {"product_id": "PROD003"},
                    {"product_id": "PROD001"}   # Another duplicate
                ]
            }
        }
        
        # Score
        scorer = diversity_scorer()
        score = await scorer(state)
        
        assert score.value == pytest.approx(3/5)  # 3 unique out of 5 total
        assert "schema: products" in score.explanation
    
    @pytest.mark.asyncio
    async def test_scorer_fails_without_schema(self):
        """Test that precision/recall scorers fail gracefully without schema info."""
        state = Mock()
        state.input = {"query": "test"}
        state.output = {"expected_items": ["ITEM1"]}
        state.metadata = {}  # No schema info
        state.outputs = {"config1": {"success": True, "results": []}}
        
        scorer = precision_scorer()
        score = await scorer(state)
        
        assert score.value == 0.0  # Returns 0.0 instead of None for validation
        assert "No schema information" in score.explanation


class TestGroundTruthIntegration:
    """Test ground truth extraction with different schemas."""
    
    @pytest.mark.asyncio
    async def test_ground_truth_extraction_with_backend(self):
        """Test that ground truth extraction uses schema analyzer correctly."""
        strategy = SchemaAwareGroundTruthStrategy()
        
        # Mock backend
        backend = Mock()
        backend.schema_name = "documents"
        backend.get_schema = Mock(return_value={
            "name": "documents",
            "fields": {
                "id_fields": ["document_id"],
                "content_fields": ["content", "title"]
            }
        })
        
        # Mock search results
        search_results = [
            Mock(metadata={"document_id": "DOC001"}),
            Mock(metadata={"document_id": "DOC002"})
        ]
        backend.search = Mock(return_value=search_results)
        
        # Register document analyzer
        register_analyzer(DocumentSchemaAnalyzer())
        
        # Extract ground truth
        trace_data = {
            "query": "machine learning",
            "metadata": {"schema": "documents"}
        }
        
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


class TestEndToEndPipeline:
    """Test the complete evaluation pipeline with different schemas."""
    
    @pytest.mark.asyncio
    async def test_video_evaluation_pipeline(self):
        """Test complete pipeline with video schema."""
        # Register analyzers
        register_analyzer(VideoSchemaAnalyzer())
        
        # Create mock state with video data
        state = Mock()
        state.input = {"query": "person running in park"}
        state.output = {"expected_items": ["VID001", "VID002", "VID003"]}
        state.metadata = {
            "schema_name": "video_frames",
            "schema_fields": {
                "id_fields": ["video_id", "frame_id"],
                "temporal_fields": ["timestamp"],
                "content_fields": ["frame_description"]
            }
        }
        
        # Mock retrieved results
        state.outputs = {
            "profile1_strategy1": {
                "success": True,
                "results": [
                    {
                        "video_id": "VID001_frame_010",
                        "frame_description": "person running",
                        "timestamp": 100
                    },
                    {
                        "video_id": "VID002_frame_020",
                        "frame_description": "person jogging in park",
                        "timestamp": 200
                    },
                    {
                        "video_id": "VID004_frame_005",
                        "frame_description": "dog running",
                        "timestamp": 50
                    }
                ]
            }
        }
        
        # Run scorers
        p_scorer = precision_scorer()
        r_scorer = recall_scorer()
        d_scorer = diversity_scorer()
        
        precision = await p_scorer(state)
        recall = await r_scorer(state)
        diversity = await d_scorer(state)
        
        # Verify results
        assert precision.value == pytest.approx(2/3)  # 2 of 3 retrieved are relevant
        assert recall.value == pytest.approx(2/3)     # Found 2 of 3 expected
        assert diversity.value == 1.0                 # All unique videos
        
        # Check metadata
        assert precision.metadata["schema"] == "video_frames"
        assert recall.metadata["expected_count"] == 3
    
    @pytest.mark.asyncio
    async def test_document_evaluation_pipeline(self):
        """Test complete pipeline with document schema."""
        # Register analyzers
        register_analyzer(DocumentSchemaAnalyzer())
        
        # Create mock state with document data
        state = Mock()
        state.input = {"query": "neural networks"}
        state.output = {"expected_items": ["DOC001", "DOC002"]}
        state.metadata = {
            "schema_name": "scientific_papers",
            "schema_fields": {
                "id_fields": ["document_id", "doi"],
                "content_fields": ["abstract", "content"],
                "metadata_fields": ["author", "year"]
            }
        }
        
        # Mock retrieved results
        state.outputs = {
            "config1": {
                "success": True,
                "results": [
                    {
                        "document_id": "DOC001",
                        "abstract": "Study on neural networks",
                        "author": "Smith et al",
                        "year": 2023
                    },
                    {
                        "document_id": "DOC001",  # Duplicate
                        "abstract": "Same paper different section",
                        "author": "Smith et al",
                        "year": 2023
                    },
                    {
                        "document_id": "DOC003",  # Not expected
                        "abstract": "Unrelated paper",
                        "author": "Jones",
                        "year": 2022
                    }
                ]
            }
        }
        
        # Run scorers
        p_scorer = precision_scorer()
        r_scorer = recall_scorer()
        d_scorer = diversity_scorer()
        
        precision = await p_scorer(state)
        recall = await r_scorer(state)
        diversity = await d_scorer(state)
        
        # Verify results
        assert precision.value == pytest.approx(1/2)  # 1 of 2 unique retrieved is relevant
        assert recall.value == pytest.approx(1/2)     # Found 1 of 2 expected
        assert diversity.value == pytest.approx(2/3)  # 2 unique out of 3 total
    
    @pytest.mark.asyncio
    async def test_mixed_schema_failure(self):
        """Test that mixing schemas causes appropriate failures."""
        # Register video analyzer
        register_analyzer(VideoSchemaAnalyzer())
        
        # Create state with video schema but document-style expected items
        state = Mock()
        state.input = {"query": "test"}
        state.output = {"expected_documents": ["DOC001"]}  # Wrong field!
        state.metadata = {
            "schema_name": "video_frames",
            "schema_fields": {"id_fields": ["video_id"]}
        }
        state.outputs = {
            "config1": {
                "success": True,
                "results": [{"video_id": "VID001"}]
            }
        }
        
        # Precision scorer should return 0.0 because expected_items is missing
        scorer = precision_scorer()
        score = await scorer(state)
        
        assert score.value == 0.0  # Returns 0.0 instead of None for validation
        assert "No expected_items field" in score.explanation