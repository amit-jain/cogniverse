"""
Comprehensive unit tests for EmbeddingGenerator to improve coverage.

Tests the main embedding generation logic with proper mocking.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np

from src.app.ingestion.processors.embedding_generator.embedding_generator import (
    EmbeddingGenerator,
    BaseEmbeddingGenerator,
    EmbeddingResult,
    ProcessingConfig
)


@pytest.mark.unit
@pytest.mark.ci_safe
class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""
    
    def test_embedding_result_creation(self):
        """Test EmbeddingResult creation."""
        result = EmbeddingResult(
            video_id="test_video",
            total_documents=10,
            documents_processed=8,
            documents_fed=7,
            processing_time=5.5,
            errors=["error1", "error2"],
            metadata={"key": "value"}
        )
        
        assert result.video_id == "test_video"
        assert result.total_documents == 10
        assert result.documents_processed == 8
        assert result.documents_fed == 7
        assert result.processing_time == 5.5
        assert result.errors == ["error1", "error2"]
        assert result.metadata == {"key": "value"}


@pytest.mark.unit
@pytest.mark.ci_safe
class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""
    
    def test_processing_config_creation(self):
        """Test ProcessingConfig creation."""
        config = ProcessingConfig(
            process_type="frame_based",
            model_name="test_model",
            backend="vespa"
        )
        
        assert config.process_type == "frame_based"
        assert config.model_name == "test_model"
        assert config.backend == "vespa"


@pytest.mark.unit
@pytest.mark.ci_safe  
class TestBaseEmbeddingGenerator:
    """Tests for BaseEmbeddingGenerator."""
    
    def test_base_embedding_generator_is_abstract(self):
        """Test that BaseEmbeddingGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbeddingGenerator({})
    
    def test_base_embedding_generator_initialization(self):
        """Test BaseEmbeddingGenerator initialization via subclass."""
        class TestGenerator(BaseEmbeddingGenerator):
            def generate_embeddings(self, video_data, output_dir):
                return EmbeddingResult("test", 0, 0, 0, 0.0, [], {})
        
        mock_logger = Mock()
        config = {"test": "value"}
        
        generator = TestGenerator(config, mock_logger)
        
        assert generator.config == config
        assert generator.logger == mock_logger
    
    def test_base_embedding_generator_default_logger(self):
        """Test BaseEmbeddingGenerator with default logger."""
        class TestGenerator(BaseEmbeddingGenerator):
            def generate_embeddings(self, video_data, output_dir):
                return EmbeddingResult("test", 0, 0, 0, 0.0, [], {})
        
        generator = TestGenerator({})
        
        assert generator.logger is not None
        assert generator.config == {}


@pytest.mark.unit
@pytest.mark.ci_safe
class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""
    
    @pytest.fixture
    def mock_logger(self):
        return Mock()
    
    @pytest.fixture
    def mock_backend_client(self):
        client = Mock()
        client.schema_name = "test_schema"
        return client
    
    @pytest.fixture
    def basic_config(self):
        return {
            "schema_name": "test_schema",
            "model_config": {}
        }
    
    @pytest.fixture
    def profile_config(self):
        return {
            "process_type": "frame_based",
            "embedding_model": "test_model"
        }
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_initialization_no_model_load(self, mock_factory, mock_processor_class, 
                                         basic_config, mock_logger, mock_backend_client):
        """Test EmbeddingGenerator initialization without model loading."""
        mock_builder = Mock()
        mock_factory.create_builder.return_value = mock_builder
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        profile_config = {"process_type": "not_direct_video"}
        
        generator = EmbeddingGenerator(
            basic_config, 
            mock_logger, 
            profile_config, 
            mock_backend_client
        )
        
        assert generator.config == basic_config
        assert generator.logger == mock_logger
        assert generator.profile_config == profile_config
        assert generator.process_type == "not_direct_video"
        assert generator.model_name == "vidore/colsmol-500m"  # default
        assert generator.backend_client == mock_backend_client
        assert generator.schema_name == "test_schema"
        assert generator.model is None  # Should not load model
        
        mock_factory.create_builder.assert_called_once_with("test_schema")
        mock_processor_class.assert_called_once_with(mock_logger)
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.get_or_load_model')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_initialization_with_model_load(self, mock_factory, mock_processor_class, 
                                           mock_get_model, basic_config, mock_logger, mock_backend_client):
        """Test EmbeddingGenerator initialization with model loading."""
        mock_builder = Mock()
        mock_factory.create_builder.return_value = mock_builder
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_model = Mock()
        mock_model_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_model_processor)
        
        profile_config = {
            "process_type": "direct_video_global",
            "embedding_model": "custom_model"
        }
        
        generator = EmbeddingGenerator(
            basic_config, 
            mock_logger, 
            profile_config, 
            mock_backend_client
        )
        
        assert generator.process_type == "direct_video_global"
        assert generator.model_name == "custom_model"
        assert generator.model == mock_model
        assert generator.processor == mock_model_processor
        
        mock_get_model.assert_called_once_with("custom_model", basic_config, mock_logger)
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.get_or_load_model')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_initialization_videoprism_model(self, mock_factory, mock_processor_class, 
                                            mock_get_model, basic_config, mock_logger, mock_backend_client):
        """Test EmbeddingGenerator initialization with VideoLLAMA model."""
        mock_builder = Mock()
        mock_factory.create_builder.return_value = mock_builder
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_loader = Mock()
        mock_get_model.return_value = (mock_loader, None)
        
        profile_config = {
            "process_type": "direct_video_global",
            "embedding_model": "videoprism_large"
        }
        
        generator = EmbeddingGenerator(
            basic_config, 
            mock_logger, 
            profile_config, 
            mock_backend_client
        )
        
        assert generator.videoprism_loader == mock_loader
        assert generator.model is None  # VideoLLAMA uses loader, not model
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.get_or_load_model')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_should_load_model_conditions(self, mock_factory, mock_processor_class, 
                                         mock_get_model, basic_config, mock_logger, mock_backend_client):
        """Test _should_load_model logic."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        mock_get_model.return_value = (Mock(), Mock())
        
        # Test cases that should load model
        test_cases = [
            "direct_video_global",
            "direct_video_local", 
            "video_chunks"
        ]
        
        for process_type in test_cases:
            profile_config = {"process_type": process_type}
            generator = EmbeddingGenerator(basic_config, mock_logger, profile_config, mock_backend_client)
            
            assert generator._should_load_model() is True, f"Should load model for {process_type}"
            # Model should actually be loaded during init
            assert generator.model is not None or generator.videoprism_loader is not None
        
        # Reset mock for next tests
        mock_get_model.reset_mock()
        
        # Test cases that should NOT load model
        test_cases = [
            "frame_based",
            "single_vector",
            "other_type"
        ]
        
        for process_type in test_cases:
            profile_config = {"process_type": process_type}
            generator = EmbeddingGenerator(basic_config, mock_logger, profile_config, mock_backend_client)
            
            assert generator._should_load_model() is False, f"Should NOT load model for {process_type}"
            # Model should NOT be loaded
            assert generator.model is None and generator.videoprism_loader is None
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.get_or_load_model')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_load_model_error_handling(self, mock_factory, mock_processor_class, 
                                      mock_get_model, basic_config, mock_logger, mock_backend_client):
        """Test _load_model error handling."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        mock_get_model.side_effect = Exception("Model loading failed")
        
        profile_config = {"process_type": "direct_video_global"}
        
        with pytest.raises(Exception, match="Model loading failed"):
            EmbeddingGenerator(basic_config, mock_logger, profile_config, mock_backend_client)
        
        mock_logger.error.assert_called_with("Failed to load model: Model loading failed")
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_generate_embeddings_unknown_processing_type(self, mock_factory, mock_processor_class,
                                                         basic_config, mock_logger, mock_backend_client):
        """Test generate_embeddings with unknown processing type."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        video_data = {
            "video_id": "test_video",
            "processing_type": "unknown_type"
        }
        
        result = generator.generate_embeddings(video_data, Path("/tmp"))
        
        assert result.video_id == "test_video"
        assert len(result.errors) > 0
        assert "Unknown processing type: unknown_type" in result.errors[0]
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_generate_embeddings_processing_method_dispatch(self, mock_factory, mock_processor_class,
                                                           basic_config, mock_logger, mock_backend_client):
        """Test that generate_embeddings correctly dispatches to processing methods."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        # Mock all processing methods
        expected_result = EmbeddingResult("test", 5, 4, 3, 2.0, [], {})
        
        with patch.object(generator, '_generate_frame_based_embeddings', return_value=expected_result) as mock_frame, \
             patch.object(generator, '_generate_single_vector_embeddings', return_value=expected_result) as mock_single, \
             patch.object(generator, '_generate_video_chunks_embeddings', return_value=expected_result) as mock_chunks, \
             patch.object(generator, '_generate_direct_video_embeddings', return_value=expected_result) as mock_direct:
            
            # Test frame_based dispatch
            video_data = {"video_id": "test", "processing_type": "frame_based"}
            result = generator.generate_embeddings(video_data, Path("/tmp"))
            mock_frame.assert_called_once_with(video_data, Path("/tmp"))
            assert result.processing_time > 0  # Should set processing time
            
            # Test single_vector dispatch
            video_data = {"video_id": "test", "processing_type": "single_vector"}
            generator.generate_embeddings(video_data, Path("/tmp"))
            mock_single.assert_called_once_with(video_data, Path("/tmp"))
            
            # Test video_chunks dispatch
            video_data = {"video_id": "test", "processing_type": "video_chunks"}
            generator.generate_embeddings(video_data, Path("/tmp"))
            mock_chunks.assert_called_once_with(video_data, Path("/tmp"))
            
            # Test direct_video variants dispatch
            for variant in ["direct_video", "direct_video_global", "direct_video_local"]:
                video_data = {"video_id": "test", "processing_type": variant}
                generator.generate_embeddings(video_data, Path("/tmp"))
            
            # Should call direct video method for all variants
            assert mock_direct.call_count == 3
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_generate_embeddings_uses_fallback_processing_type(self, mock_factory, mock_processor_class,
                                                              basic_config, mock_logger, mock_backend_client):
        """Test that generate_embeddings falls back to profile process_type."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        profile_config = {"process_type": "frame_based"}
        generator = EmbeddingGenerator(basic_config, mock_logger, profile_config, mock_backend_client)
        
        expected_result = EmbeddingResult("test", 5, 4, 3, 2.0, [], {})
        
        with patch.object(generator, '_generate_frame_based_embeddings', return_value=expected_result) as mock_frame:
            # Video data without processing_type should use profile process_type
            video_data = {"video_id": "test"}
            generator.generate_embeddings(video_data, Path("/tmp"))
            mock_frame.assert_called_once_with(video_data, Path("/tmp"))
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_generate_embeddings_error_handling(self, mock_factory, mock_processor_class,
                                               basic_config, mock_logger, mock_backend_client):
        """Test generate_embeddings error handling."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        with patch.object(generator, '_generate_frame_based_embeddings', side_effect=Exception("Processing failed")):
            video_data = {"video_id": "test_video", "processing_type": "frame_based"}
            result = generator.generate_embeddings(video_data, Path("/tmp"))
            
            assert result.video_id == "test_video"
            assert result.total_documents == 0
            assert result.documents_processed == 0
            assert result.documents_fed == 0
            assert len(result.errors) == 1
            assert "Processing failed" in result.errors[0]
            assert result.processing_time > 0
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_generate_embeddings_logs_info(self, mock_factory, mock_processor_class,
                                          basic_config, mock_logger, mock_backend_client):
        """Test that generate_embeddings logs appropriate info messages."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        profile_config = {
            "process_type": "frame_based",
            "embedding_model": "test_model"
        }
        generator = EmbeddingGenerator(basic_config, mock_logger, profile_config, mock_backend_client)
        
        expected_result = EmbeddingResult("test_video", 10, 8, 7, 0.0, [], {})
        
        with patch.object(generator, '_generate_frame_based_embeddings', return_value=expected_result):
            video_data = {"video_id": "test_video", "processing_type": "frame_based"}
            generator.generate_embeddings(video_data, Path("/tmp"))
            
            # Check that info messages were logged
            mock_logger.info.assert_any_call("Starting embedding generation for video: test_video")
            mock_logger.info.assert_any_call("Process type: frame_based")
            mock_logger.info.assert_any_call("Model: test_model")
            
            # Check completion log
            completion_calls = [call for call in mock_logger.info.call_args_list 
                              if "Completed embedding generation" in str(call)]
            assert len(completion_calls) >= 1
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_get_video_path_from_video_data(self, mock_factory, mock_processor_class,
                                           basic_config, mock_logger, mock_backend_client):
        """Test _get_video_path method with existing video path."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        with patch('pathlib.Path.exists', return_value=True):
            video_data = {
                "video_id": "test_video",
                "video_path": "/path/to/video.mp4"
            }
            
            video_path = generator._get_video_path(video_data)
            
            assert video_path == Path("/path/to/video.mp4")
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_get_video_path_search_video_dir(self, mock_factory, mock_processor_class,
                                            basic_config, mock_logger, mock_backend_client):
        """Test _get_video_path method searching video directory."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        config_with_dir = {**basic_config, "video_data_dir": "/videos"}
        generator = EmbeddingGenerator(config_with_dir, mock_logger, {}, mock_backend_client)
        
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.glob') as mock_glob:
            
            # Mock finding video file
            mock_glob.return_value = [Path("/videos/test_video.mp4")]
            
            video_data = {
                "video_id": "test_video",
                "video_path": "/nonexistent/path.mp4"
            }
            
            video_path = generator._get_video_path(video_data)
            
            assert video_path == Path("/videos/test_video.mp4")
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_get_video_path_not_found(self, mock_factory, mock_processor_class,
                                     basic_config, mock_logger, mock_backend_client):
        """Test _get_video_path method when video not found."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.glob', return_value=[]):
            
            video_data = {
                "video_id": "missing_video",
                "video_path": "/nonexistent/path.mp4"
            }
            
            video_path = generator._get_video_path(video_data)
            
            assert video_path is None
    
    @patch('cv2.VideoCapture')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_get_video_info(self, mock_factory, mock_processor_class, mock_cv2,
                           basic_config, mock_logger, mock_backend_client):
        """Test _get_video_info method."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        # Mock cv2.VideoCapture
        mock_cap = Mock()
        mock_cap.get.side_effect = [30.0, 900, 1920, 1080]  # fps, frames, width, height
        mock_cv2.return_value = mock_cap
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        video_info = generator._get_video_info(Path("/test/video.mp4"))
        
        assert video_info["fps"] == 30.0
        assert video_info["total_frames"] == 900
        assert video_info["duration"] == 30.0  # 900 / 30
        mock_cap.release.assert_called_once()
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_process_video_segment_videoprism(self, mock_factory, mock_processor_class,
                                             basic_config, mock_logger, mock_backend_client):
        """Test _process_video_segment with VideoPrism model."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        generator.videoprism_loader = Mock()  # Set as VideoPrism
        generator.embedding_processor = Mock()
        generator.document_builder = Mock()
        
        # Mock embedding processor result
        embeddings_result = {"embeddings_np": np.random.rand(768)}
        generator.embedding_processor.process_videoprism_segment.return_value = embeddings_result
        
        # Mock document builder
        expected_doc = {"id": "test_doc"}
        generator.document_builder.build_document.return_value = expected_doc
        
        result = generator._process_video_segment(
            Path("/test/video.mp4"),
            "test_video",
            0,
            0.0,
            30.0,
            1
        )
        
        assert result == expected_doc
        generator.embedding_processor.process_videoprism_segment.assert_called_once()
        generator.document_builder.build_document.assert_called_once()
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_process_video_segment_colqwen(self, mock_factory, mock_processor_class,
                                          basic_config, mock_logger, mock_backend_client):
        """Test _process_video_segment with ColQwen model."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        generator.model = Mock()  # Set as ColQwen
        generator.processor = Mock()
        generator.embedding_processor = Mock()
        generator.document_builder = Mock()
        
        # Mock embedding processor result
        embeddings = np.random.rand(128)
        generator.embedding_processor.generate_embeddings_from_video_segment.return_value = embeddings
        
        # Mock document builder
        expected_doc = {"id": "test_doc"}
        generator.document_builder.build_document.return_value = expected_doc
        
        result = generator._process_video_segment(
            Path("/test/video.mp4"),
            "test_video",
            0,
            0.0,
            30.0,
            1
        )
        
        assert result == expected_doc
        generator.embedding_processor.generate_embeddings_from_video_segment.assert_called_once()
        generator.document_builder.build_document.assert_called_once()
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_process_video_segment_no_embeddings(self, mock_factory, mock_processor_class,
                                                basic_config, mock_logger, mock_backend_client):
        """Test _process_video_segment when no embeddings generated."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        generator.model = Mock()
        generator.processor = Mock()
        generator.embedding_processor = Mock()
        
        # Mock no embeddings
        generator.embedding_processor.generate_embeddings_from_video_segment.return_value = None
        
        result = generator._process_video_segment(
            Path("/test/video.mp4"),
            "test_video",
            0,
            0.0,
            30.0,
            1
        )
        
        assert result is None
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_direct_video_embeddings_no_video_path(self, mock_factory, mock_processor_class,
                                                   basic_config, mock_logger, mock_backend_client):
        """Test _generate_direct_video_embeddings when video path not found."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        with patch.object(generator, '_get_video_path', return_value=None):
            video_data = {"video_id": "test_video"}
            
            result = generator._generate_direct_video_embeddings(video_data, Path("/tmp"))
            
            assert result.video_id == "test_video"
            assert result.total_documents == 0
            assert "Video file not found" in result.errors
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_direct_video_embeddings_success(self, mock_factory, mock_processor_class,
                                            basic_config, mock_logger, mock_backend_client):
        """Test _generate_direct_video_embeddings successful processing."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        # Mock dependencies
        video_path = Path("/test/video.mp4")
        video_info = {"duration": 90.0}
        test_doc = {"id": "test_doc"}
        
        mock_backend_client.feed_document.return_value = True
        
        with patch.object(generator, '_get_video_path', return_value=video_path), \
             patch.object(generator, '_get_video_info', return_value=video_info), \
             patch.object(generator, '_process_video_segment', return_value=test_doc):
            
            profile_config = {"model_specific": {"segment_duration": 30.0}}
            generator.profile_config = profile_config
            
            video_data = {"video_id": "test_video"}
            
            result = generator._generate_direct_video_embeddings(video_data, Path("/tmp"))
            
            assert result.video_id == "test_video"
            assert result.total_documents == 3  # ceil(90/30)
            assert result.documents_processed == 3
            assert result.documents_fed == 3
            assert len(result.errors) == 0
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_direct_video_embeddings_segment_error(self, mock_factory, mock_processor_class,
                                                   basic_config, mock_logger, mock_backend_client):
        """Test _generate_direct_video_embeddings with segment processing error."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        # Mock dependencies
        video_path = Path("/test/video.mp4")
        video_info = {"duration": 30.0}
        
        with patch.object(generator, '_get_video_path', return_value=video_path), \
             patch.object(generator, '_get_video_info', return_value=video_info), \
             patch.object(generator, '_process_video_segment', side_effect=Exception("Segment error")):
            
            video_data = {"video_id": "test_video"}
            
            result = generator._generate_direct_video_embeddings(video_data, Path("/tmp"))
            
            assert result.video_id == "test_video"
            assert result.documents_processed == 0
            assert "Segment 0: Segment error" in result.errors
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory') 
    def test_backend_client_close_on_finally(self, mock_factory, mock_processor_class,
                                            basic_config, mock_logger, mock_backend_client):
        """Test that backend client is closed in finally block."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        with patch.object(generator, '_generate_frame_based_embeddings', side_effect=Exception("Test error")):
            video_data = {"video_id": "test", "processing_type": "frame_based"}
            
            result = generator.generate_embeddings(video_data, Path("/tmp"))
            
            # Backend client should be closed even on error
            mock_backend_client.close.assert_called_once()
            assert "Test error" in result.errors[0]
    
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.EmbeddingProcessor')
    @patch('src.app.ingestion.processors.embedding_generator.embedding_generator.DocumentBuilderFactory')
    def test_feed_single_document(self, mock_factory, mock_processor_class,
                                 basic_config, mock_logger, mock_backend_client):
        """Test _feed_single_document method."""
        mock_factory.create_builder.return_value = Mock()
        mock_processor_class.return_value = Mock()
        
        generator = EmbeddingGenerator(basic_config, mock_logger, {}, mock_backend_client)
        
        # Test with backend client
        mock_backend_client.feed_document.return_value = True
        document = {"id": "test_doc"}
        
        result = generator._feed_single_document(document)
        
        assert result is True
        mock_backend_client.feed_document.assert_called_once_with(document)
        
        # Test without backend client
        generator.backend_client = None
        result = generator._feed_single_document(document)
        
        assert result is False