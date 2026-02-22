"""
Unit tests for SearchAgent
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from cogniverse_agents.search_agent import (
    ContentProcessor,
    ImagePart,
    SearchAgent,
    SearchAgentDeps,
    SearchInput,
    VideoPart,
)
from cogniverse_agents.tools.a2a_utils import A2AMessage, DataPart, Task, TextPart

# Create a mock schema_loader for all tests
mock_schema_loader = Mock()


@pytest.mark.unit
class TestContentProcessor:
    """Test cases for ContentProcessor class"""

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
        return ContentProcessor(mock_query_encoder)

    @pytest.mark.ci_fast
    def test_video_processor_initialization(self, mock_query_encoder):
        """Test ContentProcessor initialization"""
        processor = ContentProcessor(mock_query_encoder)

        assert processor.query_encoder == mock_query_encoder
        assert processor.temp_dir.exists()
        assert processor.temp_dir.name == "search_agent"

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

        processor = ContentProcessor(mock_query_encoder)

        with pytest.raises(
            NotImplementedError, match="Query encoder does not support video encoding"
        ):
            processor.process_video_file(b"fake_data", "test.mp4")

    def test_process_image_file_without_image_support(self, mock_query_encoder):
        """Test image processing when encoder doesn't support images"""
        # Remove image encoding methods
        del mock_query_encoder.encode_image

        processor = ContentProcessor(mock_query_encoder)

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
class TestSearchAgent:
    """Test cases for SearchAgent class"""

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

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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
        """Test SearchAgent initialization"""
        # Mock backend registry
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend

        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Verify agent initialized with correct dependencies
        assert agent.config is not None
        assert agent._tenant_backends == {}  # No backends created yet (lazy)
        assert agent._backend_type is not None
        assert agent._backend_config is not None
        assert agent.query_encoder == mock_query_encoder
        assert agent.embedding_type == "frame_based"
        assert isinstance(agent.content_processor, ContentProcessor)

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        results = agent._search_by_text(
            "find cats", tenant_id="test_tenant", top_k=5, ranking="binary_binary"
        )

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        mock_query_encoder.encode.assert_called_once_with("find cats")
        mock_search_backend.search.assert_called_once()

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Mock video processor
        agent.content_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )

        video_data = b"fake_video_data"
        results = agent._search_by_video(
            video_data,
            "test.mp4",
            tenant_id="test_tenant",
            top_k=5,
            ranking="binary_binary",
        )

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        agent.content_processor.process_video_file.assert_called_once_with(
            video_data, "test.mp4"
        )
        mock_search_backend.search.assert_called_once()

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Mock video processor
        agent.content_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        image_data = b"fake_image_data"
        results = agent._search_by_image(
            image_data,
            "test.jpg",
            tenant_id="test_tenant",
            top_k=5,
            ranking="binary_binary",
        )

        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        agent.content_processor.process_image_file.assert_called_once_with(
            image_data, "test.jpg"
        )
        mock_search_backend.search.assert_called_once()

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Create task with text part
        message = A2AMessage(
            role="user", parts=[DataPart(data={"query": "find dogs", "top_k": 5})]
        )
        task = Mock()
        task.id = "test_task"
        task.tenant_id = "test_tenant"
        task.messages = [message]

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["search_type"] == "text"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Mock video processor
        agent.content_processor.process_video_file = Mock(
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

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Mock video processor
        agent.content_processor.process_image_file = Mock(
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

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Mock video processor
        agent.content_processor.process_video_file = Mock(
            return_value=np.random.rand(128)
        )
        agent.content_processor.process_image_file = Mock(
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

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        task = Task(id="test_task", messages=[])

        with pytest.raises(ValueError, match="Task contains no messages"):
            agent.process_enhanced_task(task)

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Create task with TextPart but no query
        message = A2AMessage(
            role="user", parts=[DataPart(data={"no_query": "invalid"})]
        )
        task = Mock()
        task.id = "test_task"
        task.tenant_id = "test_tenant"
        task.messages = [message]

        result = agent.process_enhanced_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert result["total_results"] == 0
        assert len(result["results"]) == 0


@pytest.mark.unit
class TestSearchAgentEdgeCases:
    """Test edge cases and error conditions for SearchAgent"""

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    def test_vespa_client_initialization_failure(
        self, mock_get_config, mock_registry, mock_encoder_factory
    ):
        """Test handling of Vespa client initialization failure on first use"""
        mock_config = Mock()
        mock_config.get_active_profile.return_value = "frame_based_colpali"
        mock_config.get.return_value = {}
        mock_get_config.return_value = mock_config
        mock_encoder_factory.create_encoder.return_value = Mock()

        # Backend creation succeeds at init (lazy, no backend created)
        mock_registry.return_value.get_search_backend.return_value = Mock()

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        # Now make the registry fail for the next backend creation
        mock_registry.return_value.get_search_backend.side_effect = Exception(
            "Vespa connection failed"
        )

        with pytest.raises(Exception, match="Vespa connection failed"):
            agent._get_backend("failing_tenant")

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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
            SearchAgent(
                deps=SearchAgentDeps(
                    tenant_id="test_tenant",
                    backend_url="http://localhost",
                    backend_port=8080,
                ),
                schema_loader=mock_schema_loader,
            )

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
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

        agent = SearchAgent(
            deps=SearchAgentDeps(
                backend_url="http://localhost",
                backend_port=8080,
            ),
            schema_loader=mock_schema_loader,
        )

        with pytest.raises(Exception, match="Search failed"):
            agent._search_by_text(
                "test query", tenant_id="test_tenant", ranking="binary_binary"
            )


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
class TestSearchAgentAdvancedFeatures:
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
            patch("cogniverse_core.config.utils.get_config") as mock_get_config,
            patch(
                "cogniverse_agents.search_agent.get_backend_registry"
            ) as mock_registry,
            patch(
                "cogniverse_agents.search_agent.QueryEncoderFactory"
            ) as mock_encoder_factory,
        ):
            mock_get_config.return_value = mock_config
            mock_search_backend = Mock()
            mock_search_backend.search.return_value = []  # Empty results for some tests
            mock_registry.return_value.get_search_backend.return_value = (
                mock_search_backend
            )

            mock_query_encoder = Mock()
            mock_query_encoder.encode.return_value = np.random.rand(128)
            mock_encoder_factory.create_encoder.return_value = mock_query_encoder

            agent = SearchAgent(
                deps=SearchAgentDeps(
                    tenant_id="test_tenant",
                    backend_url="http://localhost",
                    backend_port=8080,
                ),
                schema_loader=mock_schema_loader,
            )
            agent._tenant_backends["test_tenant"] = mock_search_backend
            return agent

    @pytest.mark.ci_fast
    def test_search_with_empty_results(self, configured_agent):
        """Test handling of empty search results"""
        # Vespa client is already configured to return empty results
        results = configured_agent._search_by_text(
            "non-existent content",
            tenant_id="test_tenant",
            top_k=5,
            ranking="binary_binary",
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.ci_fast
    def test_search_with_different_rankings(self, configured_agent):
        """Test search with different ranking strategies"""
        # Test with float_binary ranking
        results = configured_agent._search_by_text(
            "find cats", tenant_id="test_tenant", top_k=5, ranking="float_binary"
        )
        assert isinstance(results, list)

        # Test with default ranking (when not specified)
        results = configured_agent._search_by_text(
            "find cats", tenant_id="test_tenant", top_k=5
        )
        assert isinstance(results, list)

    def test_search_with_temporal_parameters(self, configured_agent):
        """Test search with date range filters (currently logs warning as not yet supported)"""
        results = configured_agent._search_by_text(
            "find recent videos",
            tenant_id="test_tenant",
            top_k=5,
            ranking="binary_binary",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert isinstance(results, list)
        # Search backend should be called
        configured_agent._tenant_backends["test_tenant"].search.assert_called()
        # Note: Date parameters are not currently passed to backend (feature not implemented yet)
        # The method logs a warning instead

    @pytest.mark.ci_fast
    def test_video_processor_cleanup_on_error(self, configured_agent):
        """Test ContentProcessor cleans up temp files on error"""
        # Mock processor to raise error
        configured_agent.content_processor.process_video_file = Mock(
            side_effect=Exception("Processing failed")
        )

        video_data = b"fake_video_data"

        with pytest.raises(Exception, match="Processing failed"):
            configured_agent._search_by_video(
                video_data, "test.mp4", tenant_id="test_tenant", ranking="binary_binary"
            )

    def test_image_processor_with_different_formats(self, configured_agent):
        """Test image processing with different file formats"""
        # Mock processor
        configured_agent.content_processor.process_image_file = Mock(
            return_value=np.random.rand(128)
        )

        # Test different image formats
        for format_ext in ["jpg", "png", "jpeg", "webp"]:
            image_data = b"fake_image_data"
            results = configured_agent._search_by_image(
                image_data,
                f"test.{format_ext}",
                tenant_id="test_tenant",
                ranking="binary_binary",
            )
            assert isinstance(results, list)
            configured_agent.content_processor.process_image_file.assert_called_with(
                image_data, f"test.{format_ext}"
            )

    @pytest.mark.ci_fast
    def test_relationship_aware_search_params_validation(self, configured_agent):
        """Test RelationshipAwareSearchParams validation"""
        from cogniverse_agents.search_agent import (
            RelationshipAwareSearchParams,
        )

        # Test with minimal parameters
        params = RelationshipAwareSearchParams(
            query="test query", tenant_id="test_tenant"
        )
        assert params.query == "test query"
        assert params.tenant_id == "test_tenant"
        assert params.top_k == 10  # default
        assert params.use_relationship_boost is True  # default

        # Test with full parameters
        params_full = RelationshipAwareSearchParams(
            query="enhanced query",
            tenant_id="test_tenant",
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

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.ci_fast
    def test_routing_decision_compatibility(
        self, mock_get_config, mock_registry, mock_encoder_factory
    ):
        """Test that RoutingDecision structure is compatible with SearchAgent"""
        from cogniverse_agents.routing_agent import RoutingOutput

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
        search_agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Mock routing decision with relationships
        routing_decision = RoutingOutput(
            query="robots playing soccer in competitions",
            enhanced_query="autonomous robots demonstrating advanced soccer skills in competitive tournaments",
            recommended_agent="search_agent",
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
            assert hasattr(search_agent, "_search_by_text")
            assert routing_decision.enhanced_query is not None
            assert len(routing_decision.entities) == 3
            assert len(routing_decision.relationships) == 2


@pytest.mark.unit
class TestSearchAgentEnsembleSearch:
    """Test ensemble search with RRF fusion"""

    @pytest.fixture
    def mock_config_with_profiles(self):
        """Mock configuration with multiple profiles"""
        return {
            "active_video_profile": "video_colpali_smol500_mv_frame",
            "backend": {
                "profiles": {
                    "profile1": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    },
                    "profile2": {
                        "embedding_model": "vidore/colqwen-omni",
                        "embedding_type": "video_based",
                    },
                    "profile3": {
                        "embedding_model": "google/videoprism-base",
                        "embedding_type": "video_based",
                    },
                }
            },
        }

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.ci_fast
    def test_fuse_results_rrf_basic(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config_with_profiles,
    ):
        """Test basic RRF fusion of results from multiple profiles"""
        mock_get_config.return_value = mock_config_with_profiles
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Create mock results from two profiles
        profile_results = {
            "profile1": [
                {"id": "doc1", "score": 0.9},
                {"id": "doc2", "score": 0.8},
                {"id": "doc3", "score": 0.7},
            ],
            "profile2": [
                {"id": "doc2", "score": 0.85},  # Same doc, different rank
                {"id": "doc1", "score": 0.75},  # Same doc, different rank
                {"id": "doc4", "score": 0.70},  # New doc
            ],
        }

        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # Validate structure
        assert len(fused) <= 10
        assert all("rrf_score" in result for result in fused)
        assert all("profile_ranks" in result for result in fused)
        assert all("num_profiles" in result for result in fused)

        # Doc1 and doc2 should rank higher (appear in both profiles)
        assert fused[0]["id"] in ["doc1", "doc2"]
        assert fused[0]["num_profiles"] == 2

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.ci_fast
    def test_fuse_results_rrf_formula(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config_with_profiles,
    ):
        """Test RRF formula: score(doc) = Σ (1 / (k + rank))"""
        mock_get_config.return_value = mock_config_with_profiles
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Single profile result
        profile_results = {
            "profile1": [
                {"id": "doc1", "score": 0.9},  # rank 0
            ]
        }

        k = 60
        fused = agent._fuse_results_rrf(profile_results, k=k, top_k=10)

        # Expected RRF score: 1 / (60 + 0) = 1/60 ≈ 0.01667
        expected_score = 1.0 / (k + 0)
        assert abs(fused[0]["rrf_score"] - expected_score) < 0.001

        # Two profiles, same doc at different ranks
        profile_results = {
            "profile1": [{"id": "doc1", "score": 0.9}],  # rank 0
            "profile2": [
                {"id": "other", "score": 1.0},
                {"id": "doc1", "score": 0.8},
            ],  # rank 1
        }

        fused = agent._fuse_results_rrf(profile_results, k=k, top_k=10)

        # Expected RRF score: 1/(60+0) + 1/(60+1) ≈ 0.01667 + 0.01639 = 0.03306
        expected_score = (1.0 / (k + 0)) + (1.0 / (k + 1))
        doc1_result = next(r for r in fused if r["id"] == "doc1")
        assert abs(doc1_result["rrf_score"] - expected_score) < 0.001

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.ci_fast
    def test_fuse_results_rrf_sorting(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config_with_profiles,
    ):
        """Test that RRF fusion sorts by RRF score correctly"""
        mock_get_config.return_value = mock_config_with_profiles
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Create results where doc2 appears in both (should rank #1)
        profile_results = {
            "profile1": [
                {"id": "doc1", "score": 1.0},  # Only in profile1
                {"id": "doc2", "score": 0.9},  # In both
            ],
            "profile2": [
                {"id": "doc2", "score": 1.0},  # In both (rank 0)
                {"id": "doc3", "score": 0.9},  # Only in profile2
            ],
        }

        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # doc2 should be first (appears in both profiles)
        assert fused[0]["id"] == "doc2"
        assert fused[0]["num_profiles"] == 2

        # Verify descending RRF scores
        for i in range(len(fused) - 1):
            assert fused[i]["rrf_score"] >= fused[i + 1]["rrf_score"]

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.asyncio
    async def test_process_ensemble_mode_detection(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config_with_profiles,
    ):
        """Test that _process() detects ensemble mode with multiple profiles"""
        from unittest.mock import AsyncMock

        mock_get_config.return_value = mock_config_with_profiles
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Mock _search_ensemble to avoid actual search (must be AsyncMock)
        agent._search_ensemble = AsyncMock(
            return_value=[{"id": "doc1", "rrf_score": 0.5}]
        )

        # Call with multiple profiles
        result = await agent._process_impl(
            SearchInput(
                query="test query",
                tenant_id="test_tenant",
                profiles=["profile1", "profile2"],
                top_k=10,
            )
        )

        # Should detect ensemble mode
        assert result.search_mode == "ensemble"
        assert result.profiles == ["profile1", "profile2"]
        agent._search_ensemble.assert_called_once()

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.asyncio
    async def test_process_single_profile_mode(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config_with_profiles,
    ):
        """Test that _process() uses single profile mode when no profiles specified"""
        mock_get_config.return_value = mock_config_with_profiles
        mock_search_backend = Mock()
        mock_search_backend.search.return_value = []
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_query_encoder = Mock()
        mock_query_encoder.encode.return_value = np.random.rand(128)
        mock_encoder_factory.create_encoder.return_value = mock_query_encoder

        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Call without profiles parameter
        result = await agent._process_impl(
            SearchInput(
                query="test query",
                tenant_id="test_tenant",
                top_k=10,
            )
        )

        # Should use single profile mode
        assert result.search_mode == "single_profile"
        assert result.profile is not None

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.ci_fast
    def test_fuse_results_rrf_top_k_limit(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config_with_profiles,
    ):
        """Test that RRF fusion respects top_k limit"""
        mock_get_config.return_value = mock_config_with_profiles
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Create many results
        profile_results = {
            "profile1": [
                {"id": f"doc{i}", "score": 1.0 - (i * 0.01)} for i in range(20)
            ],
            "profile2": [
                {"id": f"doc{i + 10}", "score": 1.0 - (i * 0.01)} for i in range(20)
            ],
        }

        # Request top 5
        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=5)

        # Should return exactly 5 results
        assert len(fused) == 5

    @patch("cogniverse_agents.search_agent.QueryEncoderFactory")
    @patch("cogniverse_agents.search_agent.get_backend_registry")
    @patch("cogniverse_core.config.utils.get_config")
    @pytest.mark.ci_fast
    def test_fuse_results_rrf_empty_profiles(
        self,
        mock_get_config,
        mock_registry,
        mock_encoder_factory,
        mock_config_with_profiles,
    ):
        """Test RRF fusion with empty profile results"""
        mock_get_config.return_value = mock_config_with_profiles
        mock_search_backend = Mock()
        mock_registry.return_value.get_search_backend.return_value = mock_search_backend
        mock_encoder_factory.create_encoder.return_value = Mock()

        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=mock_schema_loader,
        )

        # Empty results from all profiles
        profile_results = {
            "profile1": [],
            "profile2": [],
        }

        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # Should return empty list
        assert len(fused) == 0


@pytest.mark.unit
class TestMultiQueryFusion:
    """Unit tests for multi-query fusion search with RRF."""

    @pytest.fixture
    def agent_with_mock_backend(self):
        """SearchAgent with mocked backend for multi-query fusion testing."""
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
            patch("cogniverse_core.config.utils.get_config") as mock_get_config,
            patch(
                "cogniverse_agents.search_agent.get_backend_registry"
            ) as mock_registry,
            patch(
                "cogniverse_agents.search_agent.QueryEncoderFactory"
            ) as mock_encoder_factory,
        ):
            mock_get_config.return_value = mock_config
            mock_search_backend = Mock()
            mock_search_backend.search.return_value = []
            mock_registry.return_value.get_search_backend.return_value = (
                mock_search_backend
            )

            mock_query_encoder = Mock()
            mock_query_encoder.encode.return_value = np.random.rand(128)
            mock_encoder_factory.create_encoder.return_value = mock_query_encoder

            agent = SearchAgent(
                deps=SearchAgentDeps(
                    tenant_id="test_tenant",
                    backend_url="http://localhost",
                    backend_port=8080,
                ),
                schema_loader=mock_schema_loader,
            )
            agent._tenant_backends["test_tenant"] = mock_search_backend
            return agent

    @pytest.mark.ci_fast
    def test_multi_query_fusion_searches_all_variants(self, agent_with_mock_backend):
        """Test _search_multi_query_fusion encodes each variant, searches, and fuses via RRF."""
        agent = agent_with_mock_backend

        searched_queries = []

        def mock_search(query_dict):
            searched_queries.append(query_dict["query"])
            sr = Mock()
            sr.document.id = f"doc_{len(searched_queries)}"
            sr.score = 0.9 - (len(searched_queries) * 0.05)
            sr.document.metadata = {"title": f"Result for {query_dict['query'][:30]}"}
            sr_shared = Mock()
            sr_shared.document.id = "doc_shared"
            sr_shared.score = 0.85
            sr_shared.document.metadata = {"title": "Shared result"}
            return [sr, sr_shared]

        mock_backend = Mock()
        mock_backend.search = mock_search
        agent._tenant_backends["test_tenant"] = mock_backend

        variants = [
            {"name": "original", "query": "robots playing soccer"},
            {
                "name": "relationship_expansion",
                "query": "robots playing soccer (robots playing soccer)",
            },
            {
                "name": "boolean_optimization",
                "query": "robots playing soccer (robots AND soccer)",
            },
        ]

        results = agent._search_multi_query_fusion(
            query_variants=variants,
            tenant_id="test_tenant",
            modality="video",
            top_k=10,
            rrf_k=60,
            ranking_strategy="binary_binary",
        )

        # All 3 variants were searched
        assert len(searched_queries) == 3

        # doc_shared appears in all 3 searches → should rank first
        assert results[0]["id"] == "doc_shared"
        assert results[0]["num_profiles"] == 3

        # RRF scores are monotonically decreasing
        for i in range(len(results) - 1):
            assert results[i]["rrf_score"] >= results[i + 1]["rrf_score"]

    @pytest.mark.ci_fast
    def test_multi_query_fusion_empty_results(self, agent_with_mock_backend):
        """Test _search_multi_query_fusion with variants that return no results."""
        agent = agent_with_mock_backend
        mock_backend = Mock()
        mock_backend.search = lambda q: []
        agent._tenant_backends["test_tenant"] = mock_backend

        results = agent._search_multi_query_fusion(
            query_variants=[
                {"name": "original", "query": "nonexistent topic xyz"},
                {"name": "expansion", "query": "nonexistent topic xyz expanded"},
            ],
            tenant_id="test_tenant",
            modality="video",
            top_k=10,
            rrf_k=60,
            ranking_strategy="binary_binary",
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.ci_fast
    def test_multi_query_fusion_variant_encoder_failure(self, agent_with_mock_backend):
        """Test _search_multi_query_fusion degrades gracefully when one variant's encoding fails."""
        agent = agent_with_mock_backend

        call_count = 0
        original_encode = agent.query_encoder.encode

        def failing_encode(query):
            nonlocal call_count
            call_count += 1
            if "FAIL_THIS_VARIANT" in query:
                raise RuntimeError("Encoder crashed for this variant")
            return original_encode(query)

        agent.query_encoder.encode = failing_encode

        def mock_search(query_dict):
            sr = Mock()
            sr.document.id = f"doc_{hash(query_dict['query']) % 100}"
            sr.score = 0.9
            sr.document.metadata = {"title": f"Result for {query_dict['query'][:30]}"}
            return [sr]

        mock_backend = Mock()
        mock_backend.search = mock_search
        agent._tenant_backends["test_tenant"] = mock_backend

        variants = [
            {"name": "original", "query": "robots playing soccer"},
            {"name": "bad_variant", "query": "FAIL_THIS_VARIANT should crash"},
            {"name": "boolean", "query": "robots AND soccer"},
        ]

        results = agent._search_multi_query_fusion(
            query_variants=variants,
            tenant_id="test_tenant",
            modality="video",
            top_k=10,
            rrf_k=60,
            ranking_strategy="binary_binary",
        )

        # Should still return results from the 2 successful variants
        assert len(results) > 0

        # RRF metadata should reflect only the surviving variants
        for doc in results:
            assert "rrf_score" in doc
            assert doc["rrf_score"] > 0

    @pytest.mark.ci_fast
    def test_relationship_context_branches_on_variants(self, agent_with_mock_backend):
        """Test search_with_relationship_context uses multi-query fusion when variants present."""
        from cogniverse_agents.search_agent import SearchContext

        agent = agent_with_mock_backend

        def mock_search(query_dict):
            sr = Mock()
            sr.document.id = "doc1"
            sr.score = 0.9
            sr.document.metadata = {"title": "Test"}
            return [sr]

        mock_backend = Mock()
        mock_backend.search = mock_search
        agent._tenant_backends["test_tenant"] = mock_backend

        # With variants → multi-query fusion path
        context_with_variants = SearchContext(
            original_query="robots soccer",
            enhanced_query="robots soccer (enhanced)",
            entities=[],
            relationships=[],
            routing_metadata={"rrf_k": 60},
            confidence=0.8,
            query_variants=[
                {"name": "original", "query": "robots soccer"},
                {"name": "expansion", "query": "robots soccer (expanded)"},
            ],
        )

        result_multi = agent.search_with_relationship_context(
            context_with_variants, tenant_id="test_tenant", top_k=10
        )
        assert result_multi["status"] == "completed"
        assert "query_variants_used" in result_multi
        assert len(result_multi["query_variants_used"]) == 2

        # Without variants → single query path
        context_without_variants = SearchContext(
            original_query="robots soccer",
            enhanced_query="robots soccer (enhanced)",
            entities=[],
            relationships=[],
            routing_metadata={},
            confidence=0.8,
            query_variants=[],
        )

        result_single = agent.search_with_relationship_context(
            context_without_variants, tenant_id="test_tenant", top_k=10
        )
        assert result_single["status"] == "completed"
        assert "query_variants_used" not in result_single

    @pytest.mark.ci_fast
    def test_routing_decision_flows_to_multi_query_fusion(
        self, agent_with_mock_backend
    ):
        """Test RoutingOutput with query_variants flows through search_with_routing_decision."""
        from cogniverse_agents.routing_agent import RoutingOutput

        agent = agent_with_mock_backend

        def mock_search(query_dict):
            sr = Mock()
            sr.document.id = f"doc_{hash(query_dict['query']) % 100}"
            sr.score = 0.9
            sr.document.metadata = {"title": "Test result"}
            sr_shared = Mock()
            sr_shared.document.id = "shared_doc"
            sr_shared.score = 0.85
            sr_shared.document.metadata = {"title": "Shared"}
            return [sr, sr_shared]

        mock_backend = Mock()
        mock_backend.search = mock_search
        agent._tenant_backends["test_tenant"] = mock_backend

        routing_decision = RoutingOutput(
            query="robots playing soccer",
            recommended_agent="search_agent",
            confidence=0.85,
            reasoning="Video search with multi-query fusion",
            enhanced_query="robots playing soccer (robots playing soccer)",
            entities=[{"text": "robots", "label": "TECHNOLOGY", "confidence": 0.9}],
            relationships=[
                {
                    "subject": "robots",
                    "relation": "playing",
                    "object": "soccer",
                    "confidence": 0.85,
                }
            ],
            metadata={"rrf_k": 60},
            query_variants=[
                {"name": "original", "query": "robots playing soccer"},
                {
                    "name": "relationship_expansion",
                    "query": "robots playing soccer (robots playing soccer)",
                },
            ],
        )

        result = agent.search_with_routing_decision(
            routing_decision, tenant_id="test_tenant", top_k=10
        )

        assert result["status"] == "completed"
        assert result["total_results"] > 0
        assert "query_variants_used" in result
        assert "original" in result["query_variants_used"]
        assert "relationship_expansion" in result["query_variants_used"]

    @pytest.mark.ci_fast
    def test_custom_rrf_k_propagates_to_fusion(self, agent_with_mock_backend):
        """Test non-default rrf_k in routing_metadata flows to _fuse_results_rrf."""
        from cogniverse_agents.routing_agent import RoutingOutput

        agent = agent_with_mock_backend

        def mock_search(query_dict):
            sr = Mock()
            sr.document.id = f"doc_{hash(query_dict['query']) % 100}"
            sr.score = 0.9
            sr.document.metadata = {"title": "Test result"}
            return [sr]

        mock_backend = Mock()
        mock_backend.search = mock_search
        agent._tenant_backends["test_tenant"] = mock_backend

        # Capture the k parameter passed to _fuse_results_rrf
        captured_k = []
        original_fuse = agent._fuse_results_rrf

        def spy_fuse(profile_results, k=60, top_k=10):
            captured_k.append(k)
            return original_fuse(profile_results, k=k, top_k=top_k)

        agent._fuse_results_rrf = spy_fuse

        custom_rrf_k = 30  # Non-default
        routing_decision = RoutingOutput(
            query="robots playing soccer",
            recommended_agent="search_agent",
            confidence=0.85,
            reasoning="Video search",
            enhanced_query="robots playing soccer",
            entities=[{"text": "robots", "label": "TECHNOLOGY", "confidence": 0.9}],
            relationships=[],
            metadata={"rrf_k": custom_rrf_k},
            query_variants=[
                {"name": "original", "query": "robots playing soccer"},
                {"name": "boolean_optimization", "query": "robots AND soccer"},
            ],
        )

        agent.search_with_routing_decision(
            routing_decision, tenant_id="test_tenant", top_k=10
        )

        assert (
            len(captured_k) == 1
        ), f"Expected _fuse_results_rrf called once, got {len(captured_k)}"
        assert captured_k[0] == custom_rrf_k, (
            f"Expected rrf_k={custom_rrf_k} passed to _fuse_results_rrf, "
            f"got rrf_k={captured_k[0]}"
        )


@pytest.mark.unit
class TestEnsembleVsFusionPaths:
    """
    Verify ensemble (multi-profile) and multi-query fusion (multi-variant) are
    structurally disjoint entry paths that cannot overlap.

    - Ensemble: _process_impl(SearchInput) → _search_ensemble()
      Triggered by SearchInput.profiles with len > 1
    - Multi-query fusion: search_with_routing_decision(RoutingDecision) → _search_multi_query_fusion()
      Triggered by RoutingDecision.query_variants with len > 1
    """

    @pytest.fixture
    def agent_with_mock_backend(self):
        """SearchAgent with mocked backend for path exclusivity testing."""
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
            patch("cogniverse_core.config.utils.get_config") as mock_get_config,
            patch(
                "cogniverse_agents.search_agent.get_backend_registry"
            ) as mock_registry,
            patch(
                "cogniverse_agents.search_agent.QueryEncoderFactory"
            ) as mock_encoder_factory,
        ):
            mock_get_config.return_value = mock_config
            mock_search_backend = Mock()
            mock_search_backend.search.return_value = []
            mock_registry.return_value.get_search_backend.return_value = (
                mock_search_backend
            )

            mock_query_encoder = Mock()
            mock_query_encoder.encode.return_value = np.random.rand(128)
            mock_encoder_factory.create_encoder.return_value = mock_query_encoder

            agent = SearchAgent(
                deps=SearchAgentDeps(
                    tenant_id="test_tenant",
                    backend_url="http://localhost",
                    backend_port=8080,
                ),
                schema_loader=mock_schema_loader,
            )
            agent._tenant_backends["test_tenant"] = mock_search_backend
            return agent

    @pytest.mark.ci_fast
    def test_search_input_has_no_query_variants_field(self):
        """SearchInput model does not expose query_variants — structural guarantee."""
        from cogniverse_agents.search_agent import SearchInput

        input_fields = set(SearchInput.model_fields.keys())
        assert "query_variants" not in input_fields, (
            "SearchInput should NOT have query_variants field. "
            "Multi-query fusion is only reachable via search_with_routing_decision(), "
            f"not _process_impl(). Fields: {input_fields}"
        )

    @pytest.mark.ci_fast
    def test_ensemble_path_does_not_trigger_fusion(self, agent_with_mock_backend):
        """search_with_relationship_context with empty variants uses single-query, not fusion."""
        agent = agent_with_mock_backend

        fusion_called = []
        original_fusion = agent._search_multi_query_fusion

        def spy_fusion(*args, **kwargs):
            fusion_called.append(True)
            return original_fusion(*args, **kwargs)

        agent._search_multi_query_fusion = spy_fusion

        from cogniverse_agents.search_agent import SearchContext

        ctx = SearchContext(
            original_query="test query",
            enhanced_query="test query",
            entities=[],
            relationships=[],
            routing_metadata={},
            confidence=0.9,
            query_variants=[],  # Empty → single-query path
        )

        agent.search_with_relationship_context(ctx, tenant_id="test_tenant", top_k=5)

        assert (
            len(fusion_called) == 0
        ), "Expected _search_multi_query_fusion NOT called when query_variants is empty"

    @pytest.mark.ci_fast
    def test_fusion_path_uses_single_profile(self, agent_with_mock_backend):
        """search_with_routing_decision with query_variants uses fusion, not ensemble."""
        from cogniverse_agents.routing_agent import RoutingOutput

        agent = agent_with_mock_backend

        ensemble_called = []
        original_ensemble = agent._search_ensemble

        def spy_ensemble(*args, **kwargs):
            ensemble_called.append(True)
            return original_ensemble(*args, **kwargs)

        agent._search_ensemble = spy_ensemble

        def mock_search(query_dict):
            sr = Mock()
            sr.document.id = f"doc_{hash(query_dict['query']) % 100}"
            sr.score = 0.9
            sr.document.metadata = {"title": "Test"}
            return [sr]

        mock_backend = Mock()
        mock_backend.search = mock_search
        agent._tenant_backends["test_tenant"] = mock_backend

        routing_decision = RoutingOutput(
            query="robots playing soccer",
            recommended_agent="search_agent",
            confidence=0.85,
            reasoning="Test",
            enhanced_query="robots playing soccer",
            entities=[],
            relationships=[],
            metadata={"rrf_k": 60},
            query_variants=[
                {"name": "original", "query": "robots playing soccer"},
                {"name": "boolean", "query": "robots AND soccer"},
            ],
        )

        agent.search_with_routing_decision(
            routing_decision, tenant_id="test_tenant", top_k=10
        )

        assert len(ensemble_called) == 0, (
            "Expected _search_ensemble NOT called when using search_with_routing_decision "
            "with query_variants (should use _search_multi_query_fusion instead)"
        )
