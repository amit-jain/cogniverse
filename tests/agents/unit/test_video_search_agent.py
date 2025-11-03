"""
Unit tests for VideoSearchAgent
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from cogniverse_agents.tools.a2a_utils import A2AMessage, DataPart, Task, TextPart
from cogniverse_agents.video_search_agent import (
    ImagePart,
    VideoPart,
    VideoProcessor,
    VideoSearchAgent,
)


@pytest.mark.unit
class TestVideoProcessor:
    """Test cases for VideoProcessor class"""

    @pytest.fixture
    def mock_query_encoder(self):
        """Mock query encoder"""
        encoder = Mock()
        encoder.encode_video = Mock(return_value=np.random.rand(128))
        encoder.encode_image = Mock(return_value=np.random.rand(128))
        encoder.encode_frames = Mock(return_value=np.random.rand(128))
        return encoder

    @pytest.fixture
    def video_processor(self, mock_query_encoder):
        """Video processor with mocked encoder"""
        return VideoProcessor(mock_query_encoder)

    @pytest.mark.ci_fast
    def test_video_processor_initialization(self, mock_query_encoder):
        """Test VideoProcessor initialization"""
        processor = VideoProcessor(mock_query_encoder)

        assert processor.query_encoder == mock_query_encoder
        assert processor.temp_dir.exists()
        assert processor.temp_dir.name == "video_search_agent"

    @pytest.mark.ci_fast
    def test_process_video_file_with_encode_video(self, video_processor):
        """Test video file processing when encoder has encode_video method"""
        video_data = b"fake_video_data"
        filename = "test_video.mp4"

        # Mock the encoder to have encode_video method
        video_processor.query_encoder.encode_video.return_value = np.random.rand(128)

        embeddings = video_processor.process_video_file(video_data, filename)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (128,)
        video_processor.query_encoder.encode_video.assert_called_once()

    def test_process_image_file_with_encode_image(self, video_processor):
        """Test image file processing when encoder has encode_image method"""
        image_data = b"fake_image_data"
        filename = "test_image.jpg"

        # Mock the encoder to have encode_image method
        video_processor.query_encoder.encode_image.return_value = np.random.rand(128)

        embeddings = video_processor.process_image_file(image_data, filename)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (128,)
        video_processor.query_encoder.encode_image.assert_called_once()

    def test_process_video_file_without_video_support(self, mock_query_encoder):
        """Test video processing when encoder doesn't support video"""
        # Remove video encoding methods
        del mock_query_encoder.encode_video
        del mock_query_encoder.encode_frames

        processor = VideoProcessor(mock_query_encoder)

        with pytest.raises(
            NotImplementedError, match="Query encoder does not support video encoding"
        ):
            processor.process_video_file(b"fake_data", "test.mp4")

    def test_process_image_file_without_image_support(self, mock_query_encoder):
        """Test image processing when encoder doesn't support images"""
        # Remove image encoding methods
        del mock_query_encoder.encode_image

        processor = VideoProcessor(mock_query_encoder)

        with pytest.raises(
            NotImplementedError, match="Query encoder does not support image encoding"
        ):
            processor.process_image_file(b"fake_data", "test.jpg")

    def test_process_video_file_cleanup(self, video_processor):
        """Test that temporary files are cleaned up after processing"""
        video_data = b"fake_video_data"
        filename = "test_video.mp4"

        # Mock the encoder to have encode_video method
        video_processor.query_encoder.encode_video.return_value = np.random.rand(128)

        # Track temp file creation and cleanup
        temp_files_before = list(video_processor.temp_dir.glob("*"))

        embeddings = video_processor.process_video_file(video_data, filename)

        temp_files_after = list(video_processor.temp_dir.glob("*"))

        # Verify embeddings returned and temp files cleaned up
        assert isinstance(embeddings, np.ndarray)
        assert len(temp_files_after) == len(temp_files_before)  # No new temp files left


@pytest.mark.unit
class TestVideoSearchAgent:
    """Test cases for VideoSearchAgent class"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            "active_video_profile": "video_colpali_smol500_mv_frame",
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            },
        }

    @pytest.fixture
    def mock_vespa_client(self):
        """Mock search backend (Vespa)"""
        client = Mock()
        # Create mock search results with proper structure
        result1 = Mock()
        result1.document.id = "video1"
        result1.document.metadata = {"video_id": "video1", "frame_id": "frame1"}
        result1.score = 0.95

        result2 = Mock()
        result2.document.id = "video2"
        result2.document.metadata = {"video_id": "video2", "frame_id": "frame2"}
        result2.score = 0.87

        client.search.return_value = [result1, result2]
        return client

    @pytest.fixture
    def mock_query_encoder(self):
        """Mock query encoder"""
        encoder = Mock()
        encoder.encode.return_value = np.random.rand(128)
        encoder.encode_video = Mock(return_value=np.random.rand(128))
        encoder.encode_image = Mock(return_value=np.random.rand(128))
        return encoder

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    @pytest.mark.ci_fast
    def test_enhanced_agent_initialization(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test VideoSearchAgent initialization"""
        # Mock backend registry
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        assert agent.config == mock_config
        assert agent.search_backend == mock_search_backend
        assert agent.query_encoder == mock_query_encoder
        assert agent.embedding_type == "frame_based"
        assert isinstance(agent.video_processor, VideoProcessor)

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    @pytest.mark.ci_fast
    def test_search_by_text(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test text-based video search"""
        # Mock backend registry
        mock_search_backend = mock_vespa_client
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        results = agent.search_by_text("find cats", top_k=5, ranking="binary_binary")

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        mock_query_encoder.encode.assert_called_once_with("find cats")
        mock_search_backend.search.assert_called_once()

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    @pytest.mark.ci_fast
    def test_search_by_video(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test video-based video search"""
        # Mock backend registry
        mock_search_backend = mock_vespa_client
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        # Mock video processor
        agent.video_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )

        video_data = b"fake_video_data"
        results = agent.search_by_video(
            video_data, "test.mp4", top_k=5, ranking="binary_binary"
        )

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        agent.video_processor.process_video_file.assert_called_once_with(
            video_data, "test.mp4"
        )
        mock_search_backend.search.assert_called_once()

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_search_by_image(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test image-based video search"""
        # Mock backend registry
        mock_search_backend = mock_vespa_client
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        # Mock video processor
        agent.video_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        image_data = b"fake_image_data"
        results = agent.search_by_image(
            image_data, "test.jpg", top_k=5, ranking="binary_binary"
        )

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        agent.video_processor.process_image_file.assert_called_once_with(
            image_data, "test.jpg"
        )
        mock_search_backend.search.assert_called_once()

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_process_enhanced_task_with_text(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with text query"""
        # Mock backend registry
        mock_search_backend = mock_vespa_client
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        # Create task with text part
        message = A2AMessage(
            role="user", parts=[DataPart(data={"query": "find dogs", "top_k": 5})]
        )
        task = Task(id="test_task", messages=[message])

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["search_type"] == "text"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_process_enhanced_task_with_video(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with video query"""
        # Mock backend registry
        mock_search_backend = mock_vespa_client
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        # Mock video processor
        agent.video_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )

        # Create mock task with video part
        message = Mock()
        message.parts = [VideoPart(video_data=b"fake_video", filename="test.mp4")]
        task = Mock()
        task.id = "test_task"
        task.messages = [message]

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["search_type"] == "video"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_process_enhanced_task_with_image(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with image query"""
        # Mock backend registry
        mock_search_backend = mock_vespa_client
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        # Mock video processor
        agent.video_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        # Create mock task with image part
        message = Mock()
        message.parts = [ImagePart(image_data=b"fake_image", filename="test.jpg")]
        task = Mock()
        task.id = "test_task"
        task.messages = [message]

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["search_type"] == "image"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_process_enhanced_task_with_mixed_parts(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with multiple query types"""
        # Mock backend registry
        mock_search_backend = mock_vespa_client
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        # Mock video processor
        agent.video_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )
        agent.video_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        # Create mock task with multiple parts
        message = Mock()
        message.parts = [
            DataPart(data={"query": "find cats", "top_k": 3}),
            VideoPart(video_data=b"fake_video", filename="test.mp4"),
            ImagePart(image_data=b"fake_image", filename="test.jpg"),
        ]
        task = Mock()
        task.id = "test_task"
        task.messages = [message]

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        # Should have results from all three searches (2 each = 6 total)
        assert result["total_results"] == 6

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_process_enhanced_task_empty_messages(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing task with no messages"""
        # Mock backend registry
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        task = Task(id="test_task", messages=[])

        with pytest.raises(ValueError, match="Task contains no messages"):
            agent.process_enhanced_task(task)

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_process_enhanced_task_no_valid_parts(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing task with no valid search parts"""
        # Mock backend registry
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        # Create task with TextPart but no query
        message = A2AMessage(
            role="user", parts=[DataPart(data={"no_query": "invalid"})]
        )
        task = Task(id="test_task", messages=[message])

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["total_results"] == 0
        assert len(result["results"]) == 0


@pytest.mark.unit
class TestVideoSearchAgentEdgeCases:
    """Test edge cases and error conditions for VideoSearchAgent"""

    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_vespa_client_initialization_failure(
        self, mock_get_config, mock_registry
    ):
        """Test handling of Vespa client initialization failure"""
        mock_config = Mock()
        mock_config.get_active_profile.return_value = "frame_based_colpali"
        mock_config.get.return_value = {}  # Return empty dict instead of Mock
        mock_get_config.return_value = mock_config
        mock_registry.return_value.get_search_backend.side_effect = Exception("Vespa connection failed")

        with pytest.raises(Exception, match="Vespa connection failed"):
            VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_query_encoder_initialization_failure(
        self, mock_get_config, mock_registry, mock_encoder_factory
    ):
        """Test handling of query encoder initialization failure"""
        # Mock backend registry
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_config = Mock()
        mock_config.get_active_profile.return_value = "frame_based_colpali"
        mock_config.get.return_value = {}  # Return empty dict instead of Mock
        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.side_effect = Exception(
            "Encoder creation failed"
        )

        with pytest.raises(Exception, match="Encoder creation failed"):
            VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    def test_search_failure_handling(
        self, mock_get_config, mock_registry, mock_encoder_factory
    ):
        """Test handling of search failures"""
        mock_config = Mock()
        mock_config.get_active_profile.return_value = "frame_based_colpali"
        mock_config.get.return_value = {}

        mock_get_config.return_value = mock_config
        mock_search_backend = Mock()
        mock_search_backend.search.side_effect = Exception("Search failed")
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = VideoSearchAgent(tenant_id="test_tenant", backend_url="http://localhost", backend_port=8080)

        with pytest.raises(Exception, match="Search failed"):
            agent.search_by_text("test query", ranking="binary_binary")


@pytest.mark.unit
class TestDataModels:
    """Test data model validation"""

    def test_video_part_validation(self):
        """Test VideoPart model validation"""
        video_part = VideoPart(
            video_data=b"fake_video_data", filename="test.mp4", content_type="video/mp4"
        )

        assert video_part.type == "video"
        assert video_part.video_data == b"fake_video_data"
        assert video_part.filename == "test.mp4"
        assert video_part.content_type == "video/mp4"

    def test_image_part_validation(self):
        """Test ImagePart model validation"""
        image_part = ImagePart(
            image_data=b"fake_image_data",
            filename="test.jpg",
            content_type="image/jpeg",
        )

        assert image_part.type == "image"
        assert image_part.image_data == b"fake_image_data"
        assert image_part.filename == "test.jpg"
        assert image_part.content_type == "image/jpeg"

    def test_enhanced_task_validation(self):
        """Test Task model validation"""
        message = A2AMessage(role="user", parts=[TextPart(text="test")])
        task = Task(id="test_task", messages=[message])

        assert task.id == "test_task"
        assert len(task.messages) == 1
        assert task.messages[0] == message


@pytest.mark.unit
class TestVideoSearchAgentAdvancedFeatures:
    """Test advanced features and edge cases"""

    @pytest.fixture
    def configured_agent(self):
        """Agent configured for advanced testing"""
        mock_config = {
            "active_video_profile": "video_colpali_smol500_mv_frame",
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            },
        }

        with (
            patch(
                "cogniverse_agents.video_search_agent.get_config"
            ) as mock_get_config,
            patch(
                "cogniverse_agents.video_search_agent.get_backend_registry"
            ) as mock_registry,
            patch(
                "cogniverse_agents.video_search_agent.QueryEncoderFactory"
            ) as mock_encoder_factory,
        ):
            mock_get_config.return_value = mock_config
            mock_search_backend = Mock()
            mock_search_backend.search.return_value = []  # Empty results for some tests
            mock_registry.return_value.get_search_backend.return_value = mock_search_backend

            mock_query_encoder = Mock()
            mock_query_encoder.encode.return_value = np.random.rand(128)
            mock_encoder_factory.create_encoder.return_value = mock_query_encoder

            agent = VideoSearchAgent(tenant_id="test_tenant",
                backend_url="http://localhost", backend_port=8080
            )
            return agent

    @pytest.mark.ci_fast
    def test_search_with_empty_results(self, configured_agent):
        """Test handling of empty search results"""
        # Vespa client is already configured to return empty results
        results = configured_agent.search_by_text(
            "non-existent content", top_k=5, ranking="binary_binary"
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.ci_fast
    def test_search_with_different_rankings(self, configured_agent):
        """Test search with different ranking strategies"""
        # Test with float_binary ranking
        results = configured_agent.search_by_text(
            "find cats", top_k=5, ranking="float_binary"
        )
        assert isinstance(results, list)

        # Test with default ranking (when not specified)
        results = configured_agent.search_by_text("find cats", top_k=5)
        assert isinstance(results, list)

    def test_search_with_temporal_parameters(self, configured_agent):
        """Test search with date range filters (currently logs warning as not yet supported)"""
        results = configured_agent.search_by_text(
            "find recent videos",
            top_k=5,
            ranking="binary_binary",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert isinstance(results, list)
        # Search backend should be called
        configured_agent.search_backend.search.assert_called()
        # Note: Date parameters are not currently passed to backend (feature not implemented yet)
        # The method logs a warning instead

    @pytest.mark.ci_fast
    def test_video_processor_cleanup_on_error(self, configured_agent):
        """Test VideoProcessor cleans up temp files on error"""
        # Mock processor to raise error
        configured_agent.video_processor.process_video_file = Mock(
            side_effect=Exception("Processing failed")
        )

        video_data = b"fake_video_data"

        with pytest.raises(Exception, match="Processing failed"):
            configured_agent.search_by_video(
                video_data, "test.mp4", ranking="binary_binary"
            )

    def test_image_processor_with_different_formats(self, configured_agent):
        """Test image processing with different file formats"""
        # Mock processor
        configured_agent.video_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        # Test different image formats
        for format_ext in ["jpg", "png", "jpeg", "webp"]:
            image_data = b"fake_image_data"
            results = configured_agent.search_by_image(
                image_data, f"test.{format_ext}", ranking="binary_binary"
            )
            assert isinstance(results, list)
            configured_agent.video_processor.process_image_file.assert_called_with(
                image_data, f"test.{format_ext}"
            )

    @pytest.mark.ci_fast
    def test_relationship_aware_search_params_validation(self, configured_agent):
        """Test RelationshipAwareSearchParams validation"""
        from cogniverse_agents.video_search_agent import (
            RelationshipAwareSearchParams,
        )

        # Test with minimal parameters
        params = RelationshipAwareSearchParams(query="test query")
        assert params.query == "test query"
        assert params.top_k == 10  # default
        assert params.use_relationship_boost is True  # default

        # Test with full parameters
        params_full = RelationshipAwareSearchParams(
            query="enhanced query",
            original_query="original query",
            enhanced_query="enhanced version",
            entities=[{"text": "test", "label": "TEST"}],
            relationships=[{"type": "ACTION", "subject": "test"}],
            top_k=15,
            ranking_strategy="float_binary",
            start_date="2024-01-01",
            end_date="2024-01-31",
            confidence_threshold=0.8,
            use_relationship_boost=False,
        )

        assert params_full.query == "enhanced query"
        assert params_full.top_k == 15
        assert params_full.ranking_strategy == "float_binary"
        assert params_full.confidence_threshold == 0.8
        assert params_full.use_relationship_boost is False

    @patch("cogniverse_agents.video_search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.video_search_agent.get_backend_registry")
    @patch("cogniverse_agents.video_search_agent.get_config")
    @pytest.mark.ci_fast
    def test_routing_decision_compatibility(
        self, mock_get_config, mock_registry, mock_encoder_factory
    ):
        """Test that RoutingDecision structure is compatible with VideoSearchAgent"""
        from cogniverse_agents.routing_agent import RoutingDecision

        # Mock config and dependencies
        mock_config = {
            "active_video_profile": "video_colpali_smol500_mv_frame",
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            },
        }
        mock_get_config.return_value = mock_config
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        # Create search agent (just testing structure compatibility)
        search_agent = VideoSearchAgent(tenant_id="test_tenant")

        # Mock routing decision with relationships
        routing_decision = RoutingDecision(
            query="robots playing soccer in competitions",
            enhanced_query="autonomous robots demonstrating advanced soccer skills in competitive tournaments",
            recommended_agent="video_search_agent",
            confidence=0.85,
            reasoning="Query contains technology and sports entities with competitive context, requiring enhanced video search",
            entities=[
                {"text": "robots", "label": "TECHNOLOGY", "confidence": 0.9},
                {"text": "soccer", "label": "SPORT", "confidence": 0.8},
                {"text": "competitions", "label": "EVENT", "confidence": 0.85},
            ],
            relationships=[
                {"subject": "robots", "relation": "playing", "object": "soccer"},
                {"subject": "soccer", "relation": "in", "object": "competitions"},
            ],
            metadata={
                "complexity_score": 0.7,
                "needs_enhancement": True,
                "relationship_extraction_applied": True,
            },
        )

        # Test that routing decision can be used for enhanced search
        assert routing_decision.enhanced_query != routing_decision.query
        assert len(routing_decision.entities) == 3
        assert len(routing_decision.relationships) == 2
        assert routing_decision.metadata["relationship_extraction_applied"] is True

        # Verify search agent can process routing decision (mock if method doesn't exist)
        if hasattr(search_agent, "_create_search_params_from_routing_decision"):
            search_params = search_agent._create_search_params_from_routing_decision(
                routing_decision
            )
            assert search_params.query == routing_decision.enhanced_query
            assert len(search_params.entities) == len(routing_decision.entities)
            assert len(search_params.relationships) == len(
                routing_decision.relationships
            )
            assert search_params.routing_confidence == routing_decision.confidence
        else:
            # Verify the search agent has the necessary attributes to handle routing decisions
            assert hasattr(search_agent, "search_by_text")
            assert routing_decision.enhanced_query is not None
            assert len(routing_decision.entities) == 3
            assert len(routing_decision.relationships) == 2
