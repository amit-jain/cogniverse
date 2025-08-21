"""
Test that evaluation works with any schema type.
"""

import pytest
from typing import Dict, Any

from src.evaluation.core.schema_analyzer import get_schema_analyzer, register_analyzer
from src.evaluation.plugins.document_analyzer import DocumentSchemaAnalyzer, ImageSchemaAnalyzer
from src.evaluation.plugins.video_analyzer import VideoSchemaAnalyzer


class TestSchemaAgnosticEvaluation:
    """Test evaluation works with different schemas."""
    
    def setup_method(self):
        """Register analyzers for testing."""
        # Register domain-specific analyzers
        register_analyzer(DocumentSchemaAnalyzer())
        register_analyzer(ImageSchemaAnalyzer())
        register_analyzer(VideoSchemaAnalyzer())
    
    def test_document_schema_detection(self):
        """Test document schema is properly detected and analyzed."""
        schema_name = "document_index"
        schema_fields = {
            "id_fields": ["document_id", "doc_id"],
            "content_fields": ["content", "body", "abstract"],
            "metadata_fields": ["author", "title", "publication_date"],
            "text_fields": ["title", "summary"]
        }
        
        # Get analyzer - should be DocumentSchemaAnalyzer
        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, DocumentSchemaAnalyzer)
        
        # Test query analysis
        query = 'author:"John Doe" climate change after:2020-01-01'
        constraints = analyzer.analyze_query(query, schema_fields)
        
        assert constraints["query_type"] == "document_author"
        assert constraints["author_constraints"]["author"] == "John"
        assert "after_date" in constraints["date_constraints"]
        
        # Test expected field name
        assert analyzer.get_expected_field_name() == "expected_documents"
    
    def test_image_schema_detection(self):
        """Test image schema is properly detected and analyzed."""
        schema_name = "image_collection"
        schema_fields = {
            "id_fields": ["image_id"],
            "content_fields": ["caption", "alt_text"],
            "metadata_fields": ["width", "height", "format"],
            "text_fields": ["tags", "description"]
        }
        
        # Get analyzer - should be ImageSchemaAnalyzer
        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, ImageSchemaAnalyzer)
        
        # Test query analysis
        query = "red sunset landscape larger than 1920"
        constraints = analyzer.analyze_query(query, schema_fields)
        
        assert constraints["query_type"] == "image_style"
        assert "red" in constraints["visual_constraints"]["colors"]
        assert "landscape" in constraints["visual_constraints"]["styles"]
        assert "min_size" in constraints["size_constraints"]
        
        # Test expected field name
        assert analyzer.get_expected_field_name() == "expected_images"
    
    def test_video_schema_detection(self):
        """Test video schema is still properly detected."""
        schema_name = "video_frames"
        schema_fields = {
            "id_fields": ["video_id", "frame_id"],
            "temporal_fields": ["start_time", "end_time"],
            "content_fields": ["frame_description", "audio_transcript"]
        }
        
        # Get analyzer - should be VideoSchemaAnalyzer
        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, VideoSchemaAnalyzer)
        
        # Test expected field name
        assert analyzer.get_expected_field_name() == "expected_videos"
    
    def test_unknown_schema_uses_default(self):
        """Test unknown schema uses default analyzer."""
        schema_name = "custom_data"
        schema_fields = {
            "id_fields": ["item_id"],
            "content_fields": ["description"],
            "metadata_fields": ["category", "tags"]
        }
        
        # Get analyzer - should be DefaultSchemaAnalyzer
        from src.evaluation.core.schema_analyzer import DefaultSchemaAnalyzer
        analyzer = get_schema_analyzer(schema_name, schema_fields)
        assert isinstance(analyzer, DefaultSchemaAnalyzer)
        
        # Test query analysis
        query = 'category:"electronics" smartphone'
        constraints = analyzer.analyze_query(query, schema_fields)
        
        assert constraints["query_type"] == "generic"
        assert "category" in constraints["field_constraints"]
        assert constraints["field_constraints"]["category"] == 'electronics'
        
        # Test expected field name
        assert analyzer.get_expected_field_name() == "expected_items"
    
    def test_id_extraction_flexibility(self):
        """Test ID extraction works across different document types."""
        from src.evaluation.core.schema_analyzer import DefaultSchemaAnalyzer
        
        # Create mock documents
        class MockDocument:
            def __init__(self, **kwargs):
                self.metadata = kwargs
        
        doc_analyzer = DocumentSchemaAnalyzer()
        img_analyzer = ImageSchemaAnalyzer()
        default_analyzer = DefaultSchemaAnalyzer()
        
        # Test document ID extraction
        doc = MockDocument(document_id="DOC123", title="Test Doc")
        assert doc_analyzer.extract_item_id(doc) == "DOC123"
        
        # Test image ID extraction
        img = MockDocument(image_id="IMG456", caption="Test Image")
        assert img_analyzer.extract_item_id(img) == "IMG456"
        
        # Test generic ID extraction
        item = MockDocument(item_id="ITEM789", description="Test Item")
        assert default_analyzer.extract_item_id(item) == "ITEM789"
    
    @pytest.mark.asyncio
    async def test_ground_truth_with_different_schemas(self):
        """Test ground truth extraction works with any schema."""
        from src.evaluation.core.ground_truth import SchemaAwareGroundTruthStrategy
        
        strategy = SchemaAwareGroundTruthStrategy()
        
        # Mock backend that returns schema info
        class MockBackend:
            def __init__(self, schema_name):
                self.schema_name = schema_name
            
            async def search(self, **kwargs):
                # Return mock results
                class MockResult:
                    def __init__(self, item_id):
                        self.metadata = {"item_id": item_id}
                
                return [MockResult(f"item_{i}") for i in range(3)]
        
        # Test with document schema
        doc_trace = {
            "query": "machine learning papers",
            "metadata": {"schema_name": "documents"}
        }
        doc_backend = MockBackend("documents")
        
        result = await strategy.extract_ground_truth(doc_trace, doc_backend)
        assert "expected_items" in result
        assert result["source"] == "schema_aware_backend"
        
        # Test with image schema
        img_trace = {
            "query": "sunset photos",
            "metadata": {"schema_name": "images"}
        }
        img_backend = MockBackend("images")
        
        result = await strategy.extract_ground_truth(img_trace, img_backend)
        assert "expected_items" in result
        assert result["source"] == "schema_aware_backend"