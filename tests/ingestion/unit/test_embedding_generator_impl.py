"""
Comprehensive unit tests for EmbeddingGeneratorImpl to improve coverage.

Tests the unified embedding generation logic with proper mocking.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np

from src.app.ingestion.processors.embedding_generator.embedding_generator_impl import (
    EmbeddingGeneratorImpl
)
from src.app.ingestion.processors.embedding_generator.embedding_generator import (
    EmbeddingResult
)


@pytest.mark.unit
@pytest.mark.ci_safe
class TestEmbeddingGeneratorImpl:
    """Tests for EmbeddingGeneratorImpl."""
    
    @pytest.fixture
    def mock_logger(self):
        return Mock()
    
    @pytest.fixture
    def mock_backend_client(self):
        client = Mock()
        client.feed_document.return_value = True
        return client
    
    @pytest.fixture
    def basic_config(self):
        return {
            "schema_name": "test_schema",
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "segment_based"  # Will load model
        }
    
    @pytest.fixture
    def frame_based_config(self):
        return {
            "schema_name": "test_schema", 
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "frame_based"  # Will NOT load model
        }
    
    @patch('src.common.models.get_or_load_model')
    def test_initialization_with_model_load(self, mock_get_model, basic_config, mock_logger, mock_backend_client):
        """Test EmbeddingGeneratorImpl initialization with model loading."""
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        generator = EmbeddingGeneratorImpl(basic_config, mock_logger, mock_backend_client)
        
        assert generator.profile_config == basic_config
        assert generator.model_name == "test_model"
        assert generator.backend_client == mock_backend_client
        assert generator.schema_name == "test_schema"
        assert generator.storage_mode == "multi_doc"
        assert generator.model == mock_model
        assert generator.processor == mock_processor
        
        mock_get_model.assert_called_once_with("test_model", basic_config, mock_logger)
    
    @patch('src.common.models.get_or_load_model')
    def test_initialization_frame_based_no_model_load(self, mock_get_model, frame_based_config, mock_logger, mock_backend_client):
        """Test EmbeddingGeneratorImpl initialization without model loading for frame-based."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        assert generator.model is None
        assert generator.processor is None
        assert generator.videoprism_loader is None
        
        # Model should not be loaded for frame_based
        mock_get_model.assert_not_called()
    
    @patch('src.common.models.get_or_load_model')
    def test_initialization_videoprism_model(self, mock_get_model, mock_logger, mock_backend_client):
        """Test EmbeddingGeneratorImpl initialization with VideoPrism model."""
        config = {
            "schema_name": "test_schema",
            "embedding_model": "videoprism_large",
            "embedding_type": "segment_based"
        }
        
        mock_loader = Mock()
        mock_get_model.return_value = (mock_loader, None)
        
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        
        assert generator.videoprism_loader == mock_loader
        assert generator.model is None
    
    @patch('src.common.models.get_or_load_model')
    def test_load_model_error_handling(self, mock_get_model, basic_config, mock_logger, mock_backend_client):
        """Test _load_model error handling."""
        mock_get_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            EmbeddingGeneratorImpl(basic_config, mock_logger, mock_backend_client)
        
        mock_logger.error.assert_called_with("Failed to load model: Model loading failed")
    
    def test_should_load_model_logic(self, mock_logger, mock_backend_client):
        """Test _should_load_model logic for different embedding types."""
        # Frame-based should NOT load model
        config = {"embedding_type": "frame_based"}
        with patch('src.app.ingestion.processors.embedding_generator.embedding_generator_impl.get_or_load_model'):
            generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
            assert generator._should_load_model() is False
        
        # Other types should load model
        test_cases = ["segment_based", "chunk_based", "global"]
        for embedding_type in test_cases:
            config = {"embedding_type": embedding_type}
            with patch('src.app.ingestion.processors.embedding_generator.embedding_generator_impl.get_or_load_model'):
                generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
                assert generator._should_load_model() is True, f"Should load model for {embedding_type}"
    
    def test_extract_segments_basic(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _extract_segments with basic segments key."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "segments": [
                {"start_time": 0.0, "end_time": 10.0},
                {"start_time": 10.0, "end_time": 20.0}
            ]
        }
        
        segments = generator._extract_segments(video_data)
        
        assert len(segments) == 2
        assert segments[0]["start_time"] == 0.0
        assert segments[1]["end_time"] == 20.0
    
    def test_extract_segments_keyframes(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _extract_segments with keyframes key."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "keyframes": [
                {"timestamp": 0.0, "frame_data": "frame1"},
                {"timestamp": 5.0, "frame_data": "frame2"}
            ]
        }
        
        segments = generator._extract_segments(video_data)
        
        assert len(segments) == 2
        assert segments[0]["timestamp"] == 0.0
        assert segments[1]["frame_data"] == "frame2"
    
    def test_extract_segments_nested_structure(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _extract_segments with nested dict structure."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "frames": {
                "keyframes": [
                    {"timestamp": 0.0, "frame_data": "frame1"},
                    {"timestamp": 5.0, "frame_data": "frame2"}
                ]
            }
        }
        
        segments = generator._extract_segments(video_data)
        
        assert len(segments) == 2
        assert segments[0]["timestamp"] == 0.0
    
    def test_extract_segments_single_vector_processing(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _extract_segments with single_vector_processing structure."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "single_vector_processing": {
                "segments": [
                    {"start_time": 0.0, "end_time": 30.0},
                    {"start_time": 30.0, "end_time": 60.0}
                ]
            }
        }
        
        segments = generator._extract_segments(video_data)
        
        assert len(segments) == 2
        assert segments[0]["start_time"] == 0.0
        assert segments[1]["end_time"] == 60.0
    
    def test_extract_segments_no_segments_found(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _extract_segments when no segments found."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {"video_id": "test", "other_data": "value"}
        
        segments = generator._extract_segments(video_data)
        
        assert segments == []
    
    def test_generate_embeddings_no_segments(self, frame_based_config, mock_logger, mock_backend_client):
        """Test generate_embeddings when no segments found."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {"video_id": "test_video"}
        
        result = generator.generate_embeddings(video_data, Path("/tmp"))
        
        assert result.video_id == "test_video"
        assert result.total_documents == 0
        assert result.documents_processed == 0
        assert result.documents_fed == 0
        assert "No segments found in video_data" in result.errors
        assert result.processing_time > 0
    
    def test_generate_embeddings_multi_doc_mode(self, frame_based_config, mock_logger, mock_backend_client):
        """Test generate_embeddings in multi-document mode."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "segments": [
                {"start_time": 0.0, "end_time": 10.0},
                {"start_time": 10.0, "end_time": 20.0}
            ],
            "storage_mode": "multi_doc"
        }
        
        expected_result = EmbeddingResult(
            video_id="test_video",
            total_documents=2,
            documents_processed=2,
            documents_fed=2,
            processing_time=0,
            errors=[],
            metadata={"num_segments": 2}
        )
        
        with patch.object(generator, '_process_multi_documents', return_value=expected_result) as mock_multi:
            result = generator.generate_embeddings(video_data, Path("/tmp"))
            
            mock_multi.assert_called_once()
            assert result.video_id == "test_video"
            assert result.total_documents == 2
            assert result.processing_time > 0  # Should be set by generate_embeddings
    
    def test_generate_embeddings_single_doc_mode(self, frame_based_config, mock_logger, mock_backend_client):
        """Test generate_embeddings in single-document mode."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "segments": [
                {"start_time": 0.0, "end_time": 10.0},
                {"start_time": 10.0, "end_time": 20.0}
            ],
            "storage_mode": "single_doc"
        }
        
        expected_result = EmbeddingResult(
            video_id="test_video",
            total_documents=1,
            documents_processed=1,
            documents_fed=1,
            processing_time=0,
            errors=[],
            metadata={"num_segments": 2}
        )
        
        with patch.object(generator, '_process_single_document', return_value=expected_result) as mock_single:
            result = generator.generate_embeddings(video_data, Path("/tmp"))
            
            mock_single.assert_called_once()
            assert result.video_id == "test_video"
            assert result.total_documents == 1
    
    def test_process_multi_documents_success(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _process_multi_documents successful processing."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4",
            "transcript": {"full_text": "Test transcript"},
            "descriptions": {"0": "First segment", "1": "Second segment"}
        }
        
        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0}
        ]
        
        # Mock dependencies
        mock_embeddings = np.random.rand(128)
        mock_doc = {"id": "test_doc"}
        
        with patch.object(generator, '_extract_transcript_text', return_value="Test transcript"), \
             patch.object(generator, '_generate_segment_embeddings', return_value=mock_embeddings), \
             patch.object(generator, '_create_segment_document', return_value=mock_doc), \
             patch.object(generator, '_feed_document', return_value=True):
            
            result = generator._process_multi_documents(video_data, segments)
            
            assert result.video_id == "test_video"
            assert result.total_documents == 2
            assert result.documents_processed == 2
            assert result.documents_fed == 2
            assert len(result.errors) == 0
    
    def test_process_multi_documents_segment_error(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _process_multi_documents with segment processing error."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4"
        }
        
        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0}
        ]
        
        with patch.object(generator, '_extract_transcript_text', return_value=""), \
             patch.object(generator, '_generate_segment_embeddings', side_effect=Exception("Embedding error")):
            
            result = generator._process_multi_documents(video_data, segments)
            
            assert result.video_id == "test_video"
            assert result.documents_processed == 0
            assert "Segment 0: Embedding error" in result.errors
            assert "Segment 1: Embedding error" in result.errors
    
    def test_process_multi_documents_no_embeddings(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _process_multi_documents when no embeddings generated."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4"
        }
        
        segments = [{"start_time": 0.0, "end_time": 10.0}]
        
        with patch.object(generator, '_extract_transcript_text', return_value=""), \
             patch.object(generator, '_generate_segment_embeddings', return_value=None):
            
            result = generator._process_multi_documents(video_data, segments)
            
            assert result.documents_processed == 0
            assert result.documents_fed == 0
    
    def test_process_single_document_success(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _process_single_document successful processing."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4"
        }
        
        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0}
        ]
        
        # Mock embeddings with different shapes to test stacking
        mock_embeddings1 = np.random.rand(1, 128)
        mock_embeddings2 = np.random.rand(1, 128)
        mock_doc = {"id": "test_doc"}
        
        with patch.object(generator, '_generate_segment_embeddings', side_effect=[mock_embeddings1, mock_embeddings2]), \
             patch.object(generator, '_create_combined_document', return_value=mock_doc), \
             patch.object(generator, '_feed_document', return_value=True):
            
            result = generator._process_single_document(video_data, segments)
            
            assert result.video_id == "test_video"
            assert result.total_documents == 1
            assert result.documents_processed == 1
            assert result.documents_fed == 1
            assert len(result.errors) == 0
    
    def test_process_single_document_no_embeddings(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _process_single_document when no embeddings generated."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4"
        }
        
        segments = [{"start_time": 0.0, "end_time": 10.0}]
        
        with patch.object(generator, '_generate_segment_embeddings', return_value=None):
            
            result = generator._process_single_document(video_data, segments)
            
            assert result.video_id == "test_video"
            assert result.total_documents == 1
            assert result.documents_processed == 0
            assert result.documents_fed == 0
            assert "No embeddings generated" in result.errors
    
    def test_process_single_document_single_embedding(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _process_single_document with only one embedding (no stacking)."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4"
        }
        
        segments = [{"start_time": 0.0, "end_time": 10.0}]
        
        mock_embeddings = np.random.rand(1, 128)
        mock_doc = {"id": "test_doc"}
        
        with patch.object(generator, '_generate_segment_embeddings', return_value=mock_embeddings), \
             patch.object(generator, '_create_combined_document', return_value=mock_doc) as mock_create, \
             patch.object(generator, '_feed_document', return_value=True):
            
            result = generator._process_single_document(video_data, segments)
            
            assert result.documents_processed == 1
            # Verify that combined embeddings is the single embedding (no stacking)
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            np.testing.assert_array_equal(call_args['embeddings'], mock_embeddings)
    
    def test_process_single_document_segment_error(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _process_single_document with segment processing error."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4"
        }
        
        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0}
        ]
        
        # First segment succeeds, second fails
        mock_embeddings = np.random.rand(1, 128)
        mock_doc = {"id": "test_doc"}
        
        with patch.object(generator, '_generate_segment_embeddings', side_effect=[mock_embeddings, Exception("Segment error")]), \
             patch.object(generator, '_create_combined_document', return_value=mock_doc), \
             patch.object(generator, '_feed_document', return_value=True):
            
            result = generator._process_single_document(video_data, segments)
            
            assert result.documents_processed == 1  # Still creates document with successful embeddings
            assert result.documents_fed == 1
            assert "Segment 1: Segment error" in result.errors
    
    def test_generate_segment_embeddings_chunk_path(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_segment_embeddings with chunk_path."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        segment = {"chunk_path": "/path/to/chunk.mp4"}
        mock_embeddings = np.random.rand(128)
        
        with patch.object(generator, '_generate_chunk_embeddings', return_value=mock_embeddings) as mock_chunk:
            result = generator._generate_segment_embeddings(segment, Path("/video.mp4"), {})
            
            mock_chunk.assert_called_once_with(Path("/path/to/chunk.mp4"))
            np.testing.assert_array_equal(result, mock_embeddings)
    
    def test_generate_segment_embeddings_video_path(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_segment_embeddings with video file path."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        segment = {"path": "/path/to/video.mp4"}
        mock_embeddings = np.random.rand(128)
        
        with patch.object(generator, '_generate_chunk_embeddings', return_value=mock_embeddings) as mock_chunk:
            result = generator._generate_segment_embeddings(segment, Path("/video.mp4"), {})
            
            mock_chunk.assert_called_once_with(Path("/path/to/video.mp4"))
            np.testing.assert_array_equal(result, mock_embeddings)
    
    def test_generate_segment_embeddings_image_path(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_segment_embeddings with image file path."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        segment = {"path": "/path/to/frame.jpg"}
        mock_embeddings = np.random.rand(128)
        
        with patch.object(generator, '_generate_frame_embeddings', return_value=mock_embeddings) as mock_frame:
            result = generator._generate_segment_embeddings(segment, Path("/video.mp4"), {})
            
            mock_frame.assert_called_once_with(Path("/path/to/frame.jpg"))
            np.testing.assert_array_equal(result, mock_embeddings)
    
    def test_generate_segment_embeddings_frame_path(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_segment_embeddings with frame_path."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        segment = {"frame_path": "/path/to/frame.png"}
        mock_embeddings = np.random.rand(128)
        
        with patch.object(generator, '_generate_frame_embeddings', return_value=mock_embeddings) as mock_frame:
            result = generator._generate_segment_embeddings(segment, Path("/video.mp4"), {})
            
            mock_frame.assert_called_once_with(Path("/path/to/frame.png"))
            np.testing.assert_array_equal(result, mock_embeddings)
    
    def test_generate_segment_embeddings_time_segment(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_segment_embeddings with time-based segment."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        segment = {"start_time": 10.0, "end_time": 20.0}
        mock_embeddings = np.random.rand(128)
        
        with patch.object(generator, '_generate_time_segment_embeddings', return_value=mock_embeddings) as mock_time:
            result = generator._generate_segment_embeddings(segment, Path("/video.mp4"), {})
            
            mock_time.assert_called_once_with(Path("/video.mp4"), 10.0, 20.0)
            np.testing.assert_array_equal(result, mock_embeddings)
    
    def test_generate_segment_embeddings_unknown_type(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_segment_embeddings with unknown segment type."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        segment = {"unknown_field": "unknown_value"}
        
        result = generator._generate_segment_embeddings(segment, Path("/video.mp4"), {})
        
        assert result is None
        mock_logger.warning.assert_called_with("Unknown segment type: dict_keys(['unknown_field'])")
    
    @patch('PIL.Image.open')
    @patch('torch.no_grad')
    def test_generate_frame_embeddings_success(self, mock_no_grad, mock_image_open, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_frame_embeddings successful processing."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        generator.model = Mock()
        generator.processor = Mock()
        generator.model.device = "cpu"
        
        # Mock image and processing
        mock_image = Mock()
        mock_image_open.return_value.convert.return_value = mock_image
        
        mock_batch_inputs = Mock()
        mock_batch_inputs.to.return_value = mock_batch_inputs
        generator.processor.process_images.return_value = mock_batch_inputs
        
        # Mock model output
        mock_embeddings = Mock()
        mock_embeddings.cpu.return_value.to.return_value.numpy.return_value = np.random.rand(1, 128)
        generator.model.return_value = mock_embeddings
        
        # Mock torch.no_grad context
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()
        
        result = generator._generate_frame_embeddings(Path("/path/to/frame.jpg"))
        
        assert result is not None
        assert result.shape == (1, 128)
        mock_image_open.assert_called_once_with(Path("/path/to/frame.jpg"))
        generator.processor.process_images.assert_called_once_with([mock_image])
    
    @patch('PIL.Image.open')
    def test_generate_frame_embeddings_no_model(self, mock_image_open, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_frame_embeddings when model not loaded."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        
        with patch.object(generator, '_load_model') as mock_load:
            # Model still None after load attempt
            result = generator._generate_frame_embeddings(Path("/path/to/frame.jpg"))
            
            assert result is None
            mock_load.assert_called_once()
            mock_logger.error.assert_called_with("Model or processor not loaded")
    
    @patch('PIL.Image.open')
    def test_generate_frame_embeddings_error(self, mock_image_open, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_frame_embeddings error handling."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        generator.model = Mock()
        generator.processor = Mock()
        
        mock_image_open.side_effect = Exception("Image load error")
        
        result = generator._generate_frame_embeddings(Path("/path/to/frame.jpg"))
        
        assert result is None
        mock_logger.error.assert_called_with("Error generating frame embeddings: Image load error")
    
    @patch('cv2.VideoCapture')
    @patch('PIL.Image.fromarray')
    @patch('cv2.cvtColor')
    @patch('torch.no_grad')
    def test_generate_chunk_embeddings_colqwen(self, mock_no_grad, mock_cvt_color, mock_from_array, 
                                              mock_video_capture, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_chunk_embeddings with ColQwen model."""
        config = {**frame_based_config, "fps": 1.0}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.model_name = "colqwen_test"
        generator.model = Mock()
        generator.processor = Mock()
        generator.model.device = "cpu"
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = [25.0, 100]  # fps, total_frames
        mock_cap.read.return_value = (True, np.random.rand(100, 100, 3))
        mock_video_capture.return_value = mock_cap
        
        # Mock image processing
        mock_pil_image = Mock()
        mock_from_array.return_value = mock_pil_image
        mock_cvt_color.return_value = np.random.rand(100, 100, 3)
        
        # Mock batch processing
        mock_batch_inputs = Mock()
        mock_batch_inputs.to.return_value = mock_batch_inputs
        generator.processor.process_images.return_value = mock_batch_inputs
        
        # Mock model output
        mock_embeddings = Mock()
        mock_embeddings.cpu.return_value.numpy.return_value = np.random.rand(10, 128)
        generator.model.return_value = mock_embeddings
        
        # Mock torch.no_grad
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()
        
        result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))
        
        assert result is not None
        assert result.shape == (128,)  # Averaged across frames
        mock_cap.release.assert_called_once()
    
    @patch('subprocess.run')
    def test_generate_chunk_embeddings_videoprism(self, mock_subprocess, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_chunk_embeddings with VideoPrism model."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        generator.videoprism_loader = Mock()
        
        # Mock ffprobe output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "30.5"
        mock_subprocess.return_value = mock_result
        
        # Mock videoprism processing
        mock_embeddings = np.random.rand(768)
        generator.videoprism_loader.process_video_segment.return_value = {
            "embeddings_np": mock_embeddings
        }
        
        result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))
        
        np.testing.assert_array_equal(result, mock_embeddings)
        generator.videoprism_loader.process_video_segment.assert_called_once_with(
            Path("/path/to/chunk.mp4"), 0, 30.5
        )
    
    @patch('cv2.VideoCapture')
    def test_generate_chunk_embeddings_no_frames(self, mock_video_capture, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_chunk_embeddings when no frames extracted."""
        config = {**frame_based_config}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.model_name = "colqwen_test"
        generator.model = Mock()
        generator.processor = Mock()
        
        # Mock video capture with no frames
        mock_cap = Mock()
        mock_cap.get.side_effect = [25.0, 100]
        mock_cap.read.return_value = (False, None)  # No frames
        mock_video_capture.return_value = mock_cap
        
        result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))
        
        assert result is None
        mock_logger.error.assert_called_with("No frames extracted from chunk")
    
    def test_generate_chunk_embeddings_error(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_chunk_embeddings error handling."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        generator.model_name = "colqwen_test"
        
        with patch('cv2.VideoCapture', side_effect=Exception("Video error")):
            result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))
            
            assert result is None
            mock_logger.error.assert_called_with("Error generating chunk embeddings: Video error")
    
    def test_generate_time_segment_embeddings_videoprism(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_time_segment_embeddings with VideoPrism."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        generator.videoprism_loader = Mock()
        
        mock_embeddings = np.random.rand(768)
        generator.videoprism_loader.process_video_segment.return_value = {
            "embeddings_np": mock_embeddings
        }
        
        result = generator._generate_time_segment_embeddings(Path("/video.mp4"), 10.0, 20.0)
        
        np.testing.assert_array_equal(result, mock_embeddings)
        generator.videoprism_loader.process_video_segment.assert_called_once_with(
            Path("/video.mp4"), 10.0, 20.0
        )
    
    def test_generate_time_segment_embeddings_videoprism_no_result(self, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_time_segment_embeddings with VideoPrism returning no result."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        generator.videoprism_loader = Mock()
        generator.videoprism_loader.process_video_segment.return_value = None
        
        result = generator._generate_time_segment_embeddings(Path("/video.mp4"), 10.0, 20.0)
        
        assert result is None
    
    @patch('cv2.VideoCapture')
    @patch('PIL.Image.fromarray')
    @patch('cv2.cvtColor')
    @patch('torch.no_grad')
    def test_generate_time_segment_embeddings_other_models(self, mock_no_grad, mock_cvt_color, mock_from_array, 
                                                          mock_video_capture, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_time_segment_embeddings with other models."""
        config = {**frame_based_config, "fps": 2.0}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.model = Mock()
        generator.processor = Mock()
        generator.model.device = "cpu"
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.return_value = 30.0  # fps
        mock_cap.read.return_value = (True, np.random.rand(100, 100, 3))
        mock_video_capture.return_value = mock_cap
        
        # Mock image processing
        mock_pil_image = Mock()
        mock_from_array.return_value = mock_pil_image
        mock_cvt_color.return_value = np.random.rand(100, 100, 3)
        
        # Mock batch processing
        mock_batch_inputs = Mock()
        mock_batch_inputs.to.return_value = mock_batch_inputs
        generator.processor.process_images.return_value = mock_batch_inputs
        
        # Mock model output
        mock_embeddings = Mock()
        mock_embeddings.cpu.return_value.numpy.return_value = np.random.rand(5, 128)
        generator.model.return_value = mock_embeddings
        
        # Mock torch.no_grad
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()
        
        result = generator._generate_time_segment_embeddings(Path("/video.mp4"), 10.0, 20.0)
        
        assert result is not None
        assert result.shape == (128,)  # Averaged across frames
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_generate_time_segment_embeddings_no_frames(self, mock_video_capture, frame_based_config, mock_logger, mock_backend_client):
        """Test _generate_time_segment_embeddings when no frames extracted."""
        generator = EmbeddingGeneratorImpl(frame_based_config, mock_logger, mock_backend_client)
        generator.model = Mock()
        generator.processor = Mock()
        
        # Mock video capture with no frames
        mock_cap = Mock()
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        result = generator._generate_time_segment_embeddings(Path("/video.mp4"), 10.0, 20.0)
        
        assert result is None