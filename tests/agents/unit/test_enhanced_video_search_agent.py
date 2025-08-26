"""
Unit tests for EnhancedVideoSearchAgent
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.app.agents.enhanced_video_search_agent import (
    EnhancedA2AMessage,
    EnhancedTask,
    EnhancedVideoSearchAgent,
    ImagePart,
    VideoPart,
    VideoProcessor,
)
from src.tools.a2a_utils import DataPart, TextPart


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
class TestEnhancedVideoSearchAgent:
    """Test cases for EnhancedVideoSearchAgent class"""

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
        """Mock Vespa client"""
        client = Mock()
        client.search.return_value = [
            {"video_id": "video1", "frame_id": "frame1", "relevance": 0.95},
            {"video_id": "video2", "frame_id": "frame2", "relevance": 0.87},
        ]
        return client

    @pytest.fixture
    def mock_query_encoder(self):
        """Mock query encoder"""
        encoder = Mock()
        encoder.encode.return_value = np.random.rand(128)
        encoder.encode_video = Mock(return_value=np.random.rand(128))
        encoder.encode_image = Mock(return_value=np.random.rand(128))
        return encoder

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    @pytest.mark.ci_fast
    def test_enhanced_agent_initialization(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test EnhancedVideoSearchAgent initialization"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        assert agent.config == mock_config
        assert agent.vespa_client == mock_vespa_client
        assert agent.query_encoder == mock_query_encoder
        assert agent.embedding_type == "frame_based"
        assert isinstance(agent.video_processor, VideoProcessor)

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    @pytest.mark.ci_fast
    def test_search_by_text(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test text-based video search"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        results = agent.search_by_text("find cats", top_k=5, ranking="binary_binary")

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        mock_query_encoder.encode.assert_called_once_with("find cats")
        mock_vespa_client.search.assert_called_once()

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    @pytest.mark.ci_fast
    def test_search_by_video(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test video-based video search"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        # Mock video processor
        agent.video_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )

        video_data = b"fake_video_data"
        results = agent.search_by_video(video_data, "test.mp4", top_k=5, ranking="binary_binary")

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        agent.video_processor.process_video_file.assert_called_once_with(
            video_data, "test.mp4"
        )
        mock_vespa_client.search.assert_called_once()

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_search_by_image(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test image-based video search"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        # Mock video processor
        agent.video_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        image_data = b"fake_image_data"
        results = agent.search_by_image(image_data, "test.jpg", top_k=5)

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        agent.video_processor.process_image_file.assert_called_once_with(
            image_data, "test.jpg"
        )
        mock_vespa_client.search.assert_called_once()

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_process_enhanced_task_with_text(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with text query"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        # Create task with text part
        message = EnhancedA2AMessage(
            role="user", parts=[DataPart(data={"query": "find dogs", "top_k": 5})]
        )
        task = EnhancedTask(id="test_task", messages=[message])

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["search_type"] == "text"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_process_enhanced_task_with_video(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with video query"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        # Mock video processor
        agent.video_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )

        # Create task with video part
        message = EnhancedA2AMessage(
            role="user",
            parts=[VideoPart(video_data=b"fake_video", filename="test.mp4")],
        )
        task = EnhancedTask(id="test_task", messages=[message])

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["search_type"] == "video"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_process_enhanced_task_with_image(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with image query"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        # Mock video processor
        agent.video_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        # Create task with image part
        message = EnhancedA2AMessage(
            role="user",
            parts=[ImagePart(image_data=b"fake_image", filename="test.jpg")],
        )
        task = EnhancedTask(id="test_task", messages=[message])

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["search_type"] == "image"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_process_enhanced_task_with_mixed_parts(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing enhanced task with multiple query types"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        # Mock video processor
        agent.video_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )
        agent.video_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        # Create task with multiple parts
        message = EnhancedA2AMessage(
            role="user",
            parts=[
                DataPart(data={"query": "find cats", "top_k": 3}),
                VideoPart(video_data=b"fake_video", filename="test.mp4"),
                ImagePart(image_data=b"fake_image", filename="test.jpg"),
            ],
        )
        task = EnhancedTask(id="test_task", messages=[message])

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        # Should have results from all three searches (2 each = 6 total)
        assert result["total_results"] == 6

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_process_enhanced_task_empty_messages(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing task with no messages"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        task = EnhancedTask(id="test_task", messages=[])

        with pytest.raises(ValueError, match="Task contains no messages"):
            agent.process_enhanced_task(task)

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_process_enhanced_task_no_valid_parts(
        self,
        mock_get_config,
        mock_vespa_class,
        mock_encoder_factory,
        mock_config,
        mock_vespa_client,
        mock_query_encoder,
    ):
        """Test processing task with no valid search parts"""
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        # Create task with TextPart but no query
        message = EnhancedA2AMessage(
            role="user", parts=[DataPart(data={"no_query": "invalid"})]
        )
        task = EnhancedTask(id="test_task", messages=[message])

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["total_results"] == 0
        assert len(result["results"]) == 0


@pytest.mark.unit
class TestEnhancedVideoSearchAgentEdgeCases:
    """Test edge cases and error conditions for EnhancedVideoSearchAgent"""

    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_vespa_client_initialization_failure(
        self, mock_get_config, mock_vespa_class
    ):
        """Test handling of Vespa client initialization failure"""
        mock_config = Mock()
        mock_config.get_active_profile.return_value = "frame_based_colpali"
        mock_config.get.return_value = {}  # Return empty dict instead of Mock
        mock_get_config.return_value = mock_config
        mock_vespa_class.side_effect = Exception("Vespa connection failed")

        with pytest.raises(Exception, match="Vespa connection failed"):
            EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_query_encoder_initialization_failure(
        self, mock_get_config, mock_vespa_class, mock_encoder_factory
    ):
        """Test handling of query encoder initialization failure"""
        mock_config = Mock()
        mock_config.get_active_profile.return_value = "frame_based_colpali"
        mock_config.get.return_value = {}  # Return empty dict instead of Mock
        mock_get_config.return_value = mock_config
        mock_vespa_class.return_value = Mock()
        mock_encoder_factory.create_encoder.side_effect = Exception(
            "Encoder creation failed"
        )

        with pytest.raises(Exception, match="Encoder creation failed"):
            EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

    @patch("src.app.agents.enhanced_video_search_agent.QueryEncoderFactory")
    @patch("src.app.agents.enhanced_video_search_agent.VespaVideoSearchClient")
    @patch("src.app.agents.enhanced_video_search_agent.get_config")
    def test_search_failure_handling(
        self, mock_get_config, mock_vespa_class, mock_encoder_factory
    ):
        """Test handling of search failures"""
        mock_config = Mock()
        mock_config.get_active_profile.return_value = "frame_based_colpali"
        mock_config.get.return_value = {}

        mock_get_config.return_value = mock_config
        mock_vespa_client = Mock()
        mock_vespa_client.search.side_effect = Exception("Search failed")
        mock_vespa_class.return_value = mock_vespa_client
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = EnhancedVideoSearchAgent(vespa_url="http://localhost", vespa_port=8080)

        with pytest.raises(Exception, match="Search failed"):
            agent.search_by_text("test query")


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

    def test_enhanced_message_validation(self):
        """Test EnhancedA2AMessage model validation"""
        message = EnhancedA2AMessage(
            role="user",
            parts=[
                TextPart(text="test query"),
                DataPart(data={"key": "value"}),
                VideoPart(video_data=b"video"),
                ImagePart(image_data=b"image"),
            ],
        )

        assert message.role == "user"
        assert len(message.parts) == 4
        assert isinstance(message.parts[0], TextPart)
        assert isinstance(message.parts[1], DataPart)
        assert isinstance(message.parts[2], VideoPart)
        assert isinstance(message.parts[3], ImagePart)

    def test_enhanced_task_validation(self):
        """Test EnhancedTask model validation"""
        message = EnhancedA2AMessage(role="user", parts=[TextPart(text="test")])
        task = EnhancedTask(id="test_task", messages=[message])

        assert task.id == "test_task"
        assert len(task.messages) == 1
        assert task.messages[0] == message
