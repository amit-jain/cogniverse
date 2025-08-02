#!/usr/bin/env python3
"""
Comprehensive test suite for the refactored embedding generator
Tests all components: Document, BackendClient, EmbeddingGenerator, VespaProcessor
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import logging
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.processing.pipeline_steps.embedding_generator.base_embedding_generator import (
    Document, MediaType, TemporalInfo, SegmentInfo, 
    EmbeddingGenerator, EmbeddingResult
)
from src.processing.pipeline_steps.embedding_generator.backend_client import BackendClient
from src.processing.pipeline_steps.embedding_generator.embedding_generator_impl import EmbeddingGeneratorImpl
from src.processing.pipeline_steps.embedding_generator.vespa_embedding_processor import VespaEmbeddingProcessor
from src.processing.pipeline_steps.embedding_generator.vespa_pyvespa_client import VespaPyClient
from src.processing.pipeline_steps.embedding_generator.backend_factory import BackendFactory
from src.processing.pipeline_steps.embedding_generator.embedding_generator_factory import create_embedding_generator


class MockBackendClient(BackendClient):
    """Mock backend client for testing"""
    
    def __init__(self, config: Dict[str, Any], schema_name: str, logger=None):
        super().__init__(config, schema_name, logger)
        self.fed_documents = []
        self.processed_documents = []
        self.connection_success = True
        self.feed_success = True
        self._embedding_processor = VespaEmbeddingProcessor(logger)
        
    def connect(self) -> bool:
        return self.connection_success
    
    def process(self, doc: Document) -> Dict[str, Any]:
        """Mock document processing with full Vespa-like structure"""
        # Process embeddings like Vespa would
        processed_embeddings = self._embedding_processor.process_embeddings(doc.raw_embeddings)
        
        processed = {
            "put": f"id:test:{self.schema_name}::{doc.document_id}",
            "fields": {
                "document_id": doc.document_id,
                "source_id": doc.source_id,
                "media_type": doc.media_type.value,
            }
        }
        
        # Add processed embeddings if they're a dict
        if isinstance(processed_embeddings, dict):
            processed["fields"].update(processed_embeddings)
        else:
            # For non-dict embeddings, store as raw_embeddings field
            processed["fields"]["raw_embeddings"] = processed_embeddings
        
        # Add temporal info
        if doc.temporal_info:
            processed["fields"]["start_time"] = doc.temporal_info.start_time
            processed["fields"]["end_time"] = doc.temporal_info.end_time
            processed["fields"]["duration"] = doc.temporal_info.duration
            
        # Add segment info
        if doc.segment_info:
            processed["fields"]["segment_idx"] = doc.segment_info.segment_idx
            processed["fields"]["total_segments"] = doc.segment_info.total_segments
            if doc.segment_info.segment_id:
                processed["fields"]["segment_id"] = doc.segment_info.segment_id
        
        # Add metadata
        for key, value in doc.metadata.items():
            if key not in processed["fields"]:
                processed["fields"][key] = value
        
        self.processed_documents.append(processed)
        return processed
    
    def _feed_prepared_batch(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> Tuple[int, List[str]]:
        """Mock batch feeding with realistic behavior"""
        # Debug: print(f"MockBackendClient._feed_prepared_batch called with {len(documents)} documents")
        
        if not self.feed_success:
            return 0, [d["put"].split("::")[-1] for d in documents]
        
        success_count = 0
        failed_ids = []
        
        # Process in batches like real implementation
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                doc_id = doc["put"].split("::")[-1]
                
                # Simulate some documents failing
                if "fail" in doc_id:
                    failed_ids.append(doc_id)
                else:
                    self.fed_documents.append(doc)
                    success_count += 1
                    
        # Debug: print(f"MockBackendClient._feed_prepared_batch returning: success={success_count}, failed={len(failed_ids)}")
        return success_count, failed_ids
    
    def check_document_exists(self, doc_id: str) -> bool:
        for doc in self.fed_documents:
            if doc["put"].split("::")[-1] == doc_id:
                return True
        return False
    
    def delete_document(self, doc_id: str) -> bool:
        original_count = len(self.fed_documents)
        self.fed_documents = [d for d in self.fed_documents if d["put"].split("::")[-1] != doc_id]
        return len(self.fed_documents) < original_count
    
    def close(self):
        # Don't clear documents in tests - we want to verify them
        # Debug: print(f"MockBackendClient.close() called - NOT clearing {len(self.fed_documents)} documents for testing")
        pass


class TestDocumentStructure(unittest.TestCase):
    """Test Document class and related structures"""
    
    def test_media_type_enum(self):
        """Test MediaType enumeration"""
        self.assertEqual(MediaType.VIDEO.value, "video")
        self.assertEqual(MediaType.IMAGE.value, "image")
        self.assertEqual(MediaType.TEXT.value, "text")
        self.assertEqual(MediaType.AUDIO.value, "audio")
        self.assertEqual(MediaType.VIDEO_FRAME.value, "video_frame")
        
    def test_temporal_info(self):
        """Test TemporalInfo with auto-calculated duration"""
        # With explicit duration
        temporal1 = TemporalInfo(start_time=10.0, end_time=40.0, duration=30.0)
        self.assertEqual(temporal1.duration, 30.0)
        
        # Auto-calculated duration
        temporal2 = TemporalInfo(start_time=10.0, end_time=40.0)
        self.assertEqual(temporal2.duration, 30.0)
        
        # Edge case: zero duration
        temporal3 = TemporalInfo(start_time=5.0, end_time=5.0)
        self.assertEqual(temporal3.duration, 0.0)
        
    def test_segment_info(self):
        """Test SegmentInfo structure"""
        # Without segment_id
        segment1 = SegmentInfo(segment_idx=5, total_segments=10)
        self.assertEqual(segment1.segment_idx, 5)
        self.assertEqual(segment1.total_segments, 10)
        self.assertIsNone(segment1.segment_id)
        
        # With custom segment_id
        segment2 = SegmentInfo(segment_idx=0, total_segments=3, segment_id="custom_seg_001")
        self.assertEqual(segment2.segment_id, "custom_seg_001")
        
    def test_document_creation_all_media_types(self):
        """Test Document creation for all media types"""
        # VIDEO
        video_doc = Document(
            media_type=MediaType.VIDEO,
            document_id="video_123_seg_0",
            source_id="video_123",
            raw_embeddings=np.random.rand(256, 768),
            temporal_info=TemporalInfo(start_time=0.0, end_time=30.0),
            segment_info=SegmentInfo(segment_idx=0, total_segments=4),
            metadata={"video_title": "Test Video", "fps": 30, "resolution": "1920x1080"}
        )
        self.assertEqual(video_doc.media_type, MediaType.VIDEO)
        self.assertEqual(video_doc.raw_embeddings.shape, (256, 768))
        self.assertEqual(video_doc.metadata["fps"], 30)
        
        # VIDEO_FRAME
        frame_doc = Document(
            media_type=MediaType.VIDEO_FRAME,
            document_id="video_123_frame_42",
            source_id="video_123",
            raw_embeddings={"colpali": np.random.rand(1, 768)},
            temporal_info=TemporalInfo(start_time=1.4, end_time=1.433),
            metadata={"frame_idx": 42, "frame_path": "/frames/frame_042.jpg"}
        )
        self.assertEqual(frame_doc.media_type, MediaType.VIDEO_FRAME)
        self.assertIsInstance(frame_doc.raw_embeddings, dict)
        
        # IMAGE
        image_doc = Document(
            media_type=MediaType.IMAGE,
            document_id="img_001",
            source_id="photo.jpg",
            raw_embeddings=np.random.rand(1, 768),
            metadata={"width": 1920, "height": 1080, "caption": "Test image"}
        )
        self.assertIsNone(image_doc.temporal_info)
        self.assertIsNone(image_doc.segment_info)
        
        # TEXT
        text_doc = Document(
            media_type=MediaType.TEXT,
            document_id="doc_page5_chunk2",
            source_id="document.pdf",
            raw_embeddings=np.random.rand(768),
            metadata={"page_number": 5, "chunk_idx": 2, "text_content": "Lorem ipsum..."}
        )
        self.assertEqual(text_doc.raw_embeddings.shape, (768,))
        
        # AUDIO (future support)
        audio_doc = Document(
            media_type=MediaType.AUDIO,
            document_id="audio_seg_0",
            source_id="podcast.mp3",
            raw_embeddings=np.random.rand(128, 768),
            temporal_info=TemporalInfo(start_time=0.0, end_time=30.0),
            segment_info=SegmentInfo(segment_idx=0, total_segments=20),
            metadata={"sample_rate": 44100, "channels": 2, "transcript": "Hello world..."}
        )
        self.assertEqual(audio_doc.media_type, MediaType.AUDIO)
        

class TestVespaEmbeddingProcessor(unittest.TestCase):
    """Test Vespa embedding processor functionality"""
    
    def setUp(self):
        self.processor = VespaEmbeddingProcessor()
        
    def test_process_numpy_embeddings(self):
        """Test processing of numpy array embeddings"""
        # Single embedding matrix
        embeddings = np.random.rand(10, 768).astype(np.float32)
        processed = self.processor.process_embeddings(embeddings)
        
        # Should create both float and binary versions
        self.assertIn("embedding", processed)
        self.assertIn("embedding_binary", processed)
        
        # Check structure
        self.assertIsInstance(processed["embedding"], dict)
        self.assertIsInstance(processed["embedding_binary"], dict)
        self.assertEqual(len(processed["embedding"]), 10)
        self.assertEqual(len(processed["embedding_binary"]), 10)
        
        # Check hex encoding
        for idx in range(10):
            # Float embeddings should be hex strings
            self.assertIsInstance(processed["embedding"][idx], str)
            # Each float32 -> bfloat16 is 2 bytes -> 4 hex chars per value
            self.assertEqual(len(processed["embedding"][idx]), 768 * 4)
            
            # Binary embeddings should also be hex strings
            self.assertIsInstance(processed["embedding_binary"][idx], str)
            # 768 values -> 768 bits -> 96 bytes -> 192 hex chars
            self.assertEqual(len(processed["embedding_binary"][idx]), 192)
            
    def test_process_dict_embeddings(self):
        """Test processing of dictionary embeddings"""
        embeddings_dict = {
            "colpali_embedding": np.random.rand(5, 768).astype(np.float32),
            "colpali_binary": np.random.rand(5, 768).astype(np.float32),
            "other_embedding": np.random.rand(3, 512).astype(np.float32)
        }
        
        processed = self.processor.process_embeddings(embeddings_dict)
        
        # All keys should be preserved
        self.assertIn("colpali_embedding", processed)
        self.assertIn("colpali_binary", processed)
        self.assertIn("other_embedding", processed)
        
        # Binary field should be binarized
        self.assertEqual(len(processed["colpali_binary"]), 5)
        self.assertIsInstance(processed["colpali_binary"][0], str)
        
        # Non-binary fields should be float encoded
        self.assertEqual(len(processed["colpali_embedding"]), 5)
        self.assertEqual(len(processed["other_embedding"]), 3)
        
    def test_process_already_processed(self):
        """Test handling of already processed embeddings"""
        already_processed = {
            "embedding": {0: "ABCD1234", 1: "EFGH5678"},
            "some_other_field": "not an embedding"
        }
        
        result = self.processor.process_embeddings(already_processed)
        
        # Should pass through unchanged
        self.assertEqual(result, already_processed)
        

class TestBackendClient(unittest.TestCase):
    """Test backend client functionality"""
    
    def setUp(self):
        self.config = {
            "backend": "test",
            "test_schema": "test_schema"
        }
        self.backend = MockBackendClient(self.config, "test_schema")
        
    def test_process_document_complete(self):
        """Test complete document processing"""
        doc = Document(
            media_type=MediaType.VIDEO,
            document_id="test_video_0_0",
            source_id="test_video",
            raw_embeddings=np.random.rand(10, 768),
            temporal_info=TemporalInfo(start_time=0.0, end_time=30.0),
            segment_info=SegmentInfo(segment_idx=0, total_segments=4),
            metadata={
                "video_title": "Test Video",
                "fps": 30,
                "resolution": "1920x1080",
                "codec": "h264"
            }
        )
        
        processed = self.backend.process(doc)
        
        # Check structure
        self.assertIn("put", processed)
        self.assertIn("fields", processed)
        
        # Check document ID
        self.assertEqual(processed["put"], "id:test:test_schema::test_video_0_0")
        
        # Check all fields are present
        fields = processed["fields"]
        self.assertEqual(fields["document_id"], "test_video_0_0")
        self.assertEqual(fields["source_id"], "test_video")
        self.assertEqual(fields["media_type"], "video")
        self.assertEqual(fields["start_time"], 0.0)
        self.assertEqual(fields["end_time"], 30.0)
        self.assertEqual(fields["duration"], 30.0)
        self.assertEqual(fields["segment_idx"], 0)
        self.assertEqual(fields["total_segments"], 4)
        self.assertEqual(fields["video_title"], "Test Video")
        self.assertEqual(fields["fps"], 30)
        
        # Check embeddings were processed
        self.assertIn("embedding", fields)
        self.assertIn("embedding_binary", fields)
        
    def test_feed_single_document(self):
        """Test feeding single document"""
        doc = Document(
            media_type=MediaType.IMAGE,
            document_id="image_001",
            source_id="test.jpg",
            raw_embeddings=np.random.rand(1, 768)
        )
        
        success_count, failed_ids = self.backend.feed(doc)
        
        self.assertEqual(success_count, 1)
        self.assertEqual(len(failed_ids), 0)
        self.assertTrue(self.backend.check_document_exists("image_001"))
        
    def test_feed_multiple_documents(self):
        """Test feeding multiple documents"""
        docs = [
            Document(
                media_type=MediaType.VIDEO_FRAME,
                document_id=f"frame_{i}",
                source_id="video_123",
                raw_embeddings=np.random.rand(1, 768),
                metadata={"frame_idx": i}
            )
            for i in range(10)
        ]
        
        success_count, failed_ids = self.backend.feed(docs)
        
        self.assertEqual(success_count, 10)
        self.assertEqual(len(failed_ids), 0)
        self.assertEqual(len(self.backend.fed_documents), 10)
        
    def test_feed_with_failures(self):
        """Test handling of feed failures"""
        docs = [
            Document(
                media_type=MediaType.VIDEO,
                document_id="success_1",
                source_id="video1",
                raw_embeddings=np.random.rand(10, 768)
            ),
            Document(
                media_type=MediaType.VIDEO,
                document_id="fail_this_one",  # Will fail due to 'fail' in ID
                source_id="video2",
                raw_embeddings=np.random.rand(10, 768)
            ),
            Document(
                media_type=MediaType.VIDEO,
                document_id="success_2",
                source_id="video3",
                raw_embeddings=np.random.rand(10, 768)
            )
        ]
        
        success_count, failed_ids = self.backend.feed(docs)
        
        self.assertEqual(success_count, 2)
        self.assertEqual(len(failed_ids), 1)
        self.assertEqual(failed_ids[0], "fail_this_one")
        self.assertTrue(self.backend.check_document_exists("success_1"))
        self.assertTrue(self.backend.check_document_exists("success_2"))
        self.assertFalse(self.backend.check_document_exists("fail_this_one"))
        
    def test_batch_processing(self):
        """Test batch processing with large number of documents"""
        # Create 250 documents to test batching
        docs = [
            Document(
                media_type=MediaType.TEXT,
                document_id=f"doc_{i:04d}",
                source_id=f"source_{i // 10}",
                raw_embeddings=np.random.rand(768)
            )
            for i in range(250)
        ]
        
        success_count, failed_ids = self.backend.feed(docs, batch_size=50)
        
        self.assertEqual(success_count, 250)
        self.assertEqual(len(failed_ids), 0)
        
        # Verify all documents exist
        for i in range(250):
            self.assertTrue(self.backend.check_document_exists(f"doc_{i:04d}"))
            
    def test_delete_document(self):
        """Test document deletion"""
        doc = Document(
            media_type=MediaType.IMAGE,
            document_id="to_delete",
            source_id="image.jpg",
            raw_embeddings=np.random.rand(1, 768)
        )
        
        # Feed document
        self.backend.feed(doc)
        self.assertTrue(self.backend.check_document_exists("to_delete"))
        
        # Delete document
        deleted = self.backend.delete_document("to_delete")
        self.assertTrue(deleted)
        self.assertFalse(self.backend.check_document_exists("to_delete"))
        
        # Try deleting non-existent document
        deleted_again = self.backend.delete_document("to_delete")
        self.assertFalse(deleted_again)


class TestEmbeddingGenerator(unittest.TestCase):
    """Test EmbeddingGeneratorImpl functionality"""
    
    def setUp(self):
        self.config = {
            "backend": "test",
            "test_schema": "test_schema"
        }
        
        self.profile_config = {
            "process_type": "direct_video",
            "embedding_model": "test_model",
            "segment_duration": 30.0,
            "fps": 1.0,
            "batch_size": 32
        }
        
        self.backend = MockBackendClient(self.config, "test_schema")
        
        # Mock logger
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.INFO)
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_generator_initialization(self, mock_get_model):
        """Test generator initialization"""
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=self.profile_config,
            backend_client=self.backend
        )
        
        # Check initialization
        self.assertEqual(generator.media_type, MediaType.VIDEO)
        self.assertEqual(generator.process_type, "direct_video")
        self.assertEqual(generator.model_name, "test_model")
        self.assertIsNotNone(generator.model)
        self.assertIsNotNone(generator.processor)
        
        # Model should be loaded for direct_video
        mock_get_model.assert_called_once()
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_frame_based_initialization(self, mock_get_model):
        """Test initialization for frame-based processing"""
        self.profile_config["process_type"] = "frame_based"
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=self.profile_config,
            backend_client=self.backend
        )
        
        self.assertEqual(generator.media_type, MediaType.VIDEO_FRAME)
        
        # Model should not be loaded during init for frame-based
        mock_get_model.assert_not_called()
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_process_video_segment(self, mock_get_model):
        """Test video segment processing"""
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=self.profile_config,
            backend_client=self.backend
        )
        
        # Mock raw embedding generation
        test_embeddings = np.random.rand(256, 768)
        generator._generate_raw_embeddings = Mock(return_value=test_embeddings)
        
        # Process segment
        doc = generator._process_video_segment(
            video_path=Path("/test/video.mp4"),
            video_id="test_video",
            segment_idx=2,
            start_time=60.0,
            end_time=90.0,
            num_segments=10
        )
        
        # Verify document structure
        self.assertIsNotNone(doc)
        self.assertEqual(doc.media_type, MediaType.VIDEO)
        self.assertEqual(doc.document_id, "test_video_2_60")
        self.assertEqual(doc.source_id, "test_video")
        self.assertEqual(doc.temporal_info.start_time, 60.0)
        self.assertEqual(doc.temporal_info.end_time, 90.0)
        self.assertEqual(doc.temporal_info.duration, 30.0)
        self.assertEqual(doc.segment_info.segment_idx, 2)
        self.assertEqual(doc.segment_info.total_segments, 10)
        self.assertTrue(np.array_equal(doc.raw_embeddings, test_embeddings))
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_generate_embeddings_direct_video(self, mock_get_model):
        """Test complete embedding generation for direct video"""
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        # Create mock backend
        mock_backend = MockBackendClient(self.config, "test_schema", self.logger)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=self.profile_config,
            backend_client=mock_backend
        )
        
        # Mock raw embedding generation
        generator._generate_raw_embeddings = Mock(
            return_value=np.random.rand(256, 768)
        )
        
        # Test data
        video_data = {
            "video_id": "test_video_123",
            "video_path": "/test/video.mp4",
            "duration": 95.0  # Will create 4 segments (0-30, 30-60, 60-90, 90-95)
        }
        
        # Store backend reference for assertions
        self.test_backend = mock_backend
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # The backend will be closed after generation, so we need to check during
            result = generator.generate_embeddings(video_data, output_dir)
        
        # Check result
        self.assertEqual(result.video_id, "test_video_123")
        self.assertEqual(result.total_documents, 4)
        self.assertEqual(result.documents_processed, 4)
        
        # Verify results
        self.assertEqual(result.documents_fed, 4)
        
        self.assertEqual(len(result.errors), 0)
        self.assertGreater(result.processing_time, 0)
        
        # Verify backend state
        
        # Verify all segments were fed to backend
        self.assertEqual(len(self.test_backend.fed_documents), 4)
        
        # Check segment times
        for i, doc in enumerate(self.test_backend.fed_documents):
            fields = doc["fields"]
            expected_start = i * 30.0
            expected_end = min((i + 1) * 30.0, 95.0)
            self.assertEqual(fields["start_time"], expected_start)
            self.assertEqual(fields["end_time"], expected_end)
            self.assertEqual(fields["segment_idx"], i)
            
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')  
    def test_error_handling(self, mock_get_model):
        """Test error handling in embedding generation"""
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=self.profile_config,
            backend_client=self.backend
        )
        
        # Make embedding generation fail for some segments
        call_count = 0
        def mock_embeddings(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second segment
                return None
            return np.random.rand(256, 768)
            
        generator._generate_raw_embeddings = Mock(side_effect=mock_embeddings)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/test/video.mp4",
            "duration": 90.0  # 3 segments
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generator.generate_embeddings(video_data, Path(tmpdir))
        
        # Should process 2 out of 3 segments (second one failed)
        self.assertEqual(result.total_documents, 3)
        self.assertEqual(result.documents_processed, 2)
        self.assertEqual(result.documents_fed, 2)
        

class TestEndToEndIntegration(unittest.TestCase):
    """Test complete end-to-end integration"""
    
    def setUp(self):
        self.config = {
            "backend": "vespa",
            "vespa_url": "http://localhost",
            "vespa_port": 8080,
            "vespa_schema": "video_frame",
            "embedding_backend": "vespa",
            "active_profile": "test_profile",
            "video_processing_profiles": {
                "test_profile": {
                    "process_type": "direct_video",
                    "embedding_model": "test_model",
                    "segment_duration": 30.0
                }
            }
        }
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_factory_creation(self, mock_get_model):
        """Test factory creates generator correctly"""
        # Mock model
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        # Patch the backend factory's backends dict
        from src.processing.pipeline_steps.embedding_generator.backend_factory import BackendFactory
        original_backends = BackendFactory._backends.copy()
        BackendFactory._backends["vespa"] = MockBackendClient
        
        try:
            # Create generator through factory
            generator = create_embedding_generator(self.config)
            
            self.assertIsInstance(generator, EmbeddingGeneratorImpl)
            
            # The backend should have been created
            self.assertIsNotNone(generator.backend_client)
            self.assertIsInstance(generator.backend_client, MockBackendClient)
            
        finally:
            # Restore original backends
            BackendFactory._backends = original_backends
        
    def test_vespa_embedding_processor_edge_cases(self):
        """Test edge cases in Vespa embedding processor"""
        processor = VespaEmbeddingProcessor()
        
        # Empty array
        empty = np.array([])
        result = processor.process_embeddings(empty)
        self.assertEqual(result, [])
        
        # Single dimension array
        single_dim = np.random.rand(768)
        result = processor.process_embeddings(single_dim)
        # Should return as list since it's not 2D
        np.testing.assert_array_equal(result, single_dim.tolist())
        
        # Mixed types in dict
        mixed_dict = {
            "embeddings": np.random.rand(5, 768),
            "metadata": {"key": "value"},
            "binary_field": np.random.rand(5, 768),
            "already_hex": {0: "ABCD", 1: "EFGH"}
        }
        
        result = processor.process_embeddings(mixed_dict)
        
        # Numpy arrays should be processed
        self.assertIsInstance(result["embeddings"], dict)
        self.assertIsInstance(result["binary_field"], dict)
        
        # Non-embedding fields should pass through
        self.assertEqual(result["metadata"], {"key": "value"})
        self.assertEqual(result["already_hex"], {0: "ABCD", 1: "EFGH"})
        
    def test_document_with_large_embeddings(self):
        """Test handling of documents with large embeddings"""
        # Create document with large embedding matrix (e.g., VideoPrism)
        large_embeddings = np.random.rand(4096, 768)  # 4096 patches
        
        doc = Document(
            media_type=MediaType.VIDEO,
            document_id="large_embed_video",
            source_id="video_with_many_patches",
            raw_embeddings=large_embeddings,
            temporal_info=TemporalInfo(start_time=0.0, end_time=30.0),
            metadata={"patch_count": 4096}
        )
        
        # Process through backend
        backend = MockBackendClient(self.config, "video_frame")
        processed = backend.process(doc)
        
        # Check that all patches were processed
        self.assertIn("embedding", processed["fields"])
        self.assertEqual(len(processed["fields"]["embedding"]), 4096)
        
        # Feed to backend
        success_count, failed_ids = backend.feed(doc)
        
        self.assertEqual(success_count, 1)
        self.assertEqual(len(failed_ids), 0)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)