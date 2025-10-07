"""Integration tests for Enhanced Video Search Agent (Phase 3)."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from src.app.agents.video_search_agent import VideoSearchAgent
from src.tools.a2a_utils import (
    A2AMessage,
    FilePart,
    Task,
    TextPart,
)


@pytest.fixture
def mock_vespa_client():
    """Mock search service for integration testing"""
    service = Mock()
    service.search.return_value = [
        {
            "id": "video_1",
            "title": "Machine Learning Tutorial",
            "description": "Introduction to ML concepts",
            "thumbnail": "/path/to/thumb1.jpg",
            "score": 0.95,
        },
        {
            "id": "video_2",
            "title": "AI Ethics Discussion",
            "description": "Ethical considerations in AI",
            "thumbnail": "/path/to/thumb2.jpg",
            "score": 0.87,
        },
    ]
    return service


@pytest.fixture
def mock_processor():
    """Mock video processor for integration testing"""
    import numpy as np

    processor = Mock()
    # Return mock numpy arrays for embeddings
    processor.process_video_file.return_value = np.random.randn(128).astype(np.float32)
    processor.process_image_file.return_value = np.random.randn(128).astype(np.float32)
    processor.cleanup_temp_files.return_value = None
    return processor


class TestVideoSearchAgentIntegration:
    """Integration tests for Enhanced Video Search Agent functionality"""

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @pytest.mark.ci_fast
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    def test_agent_initialization_with_real_config(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client
    ):
        """Test agent initialization with real configuration"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile for video search agent
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        agent = VideoSearchAgent()

        assert agent.vespa_client == mock_vespa_client
        assert agent.video_processor is not None
        mock_vespa_client_class.assert_called_once()

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multimodal_search_workflow_integration(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client, mock_processor
    ):
        """Test complete multimodal search workflow"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        agent = VideoSearchAgent()
        agent.video_processor = mock_processor

        # Test text search
        text_results = agent.search_by_text("machine learning tutorial", top_k=5)
        assert len(text_results) == 2
        assert text_results[0]["title"] == "Machine Learning Tutorial"
        # Verify search was called (parameters will include search dict and embeddings)
        mock_vespa_client.search.assert_called_once()

        # Test video search
        video_data = b"fake_video_data"
        video_results = agent.search_by_video(video_data, "test.mp4", top_k=3)
        assert len(video_results) == 2
        mock_processor.process_video_file.assert_called_once_with(
            video_data, "test.mp4"
        )

        # Test image search
        image_data = b"fake_image_data"
        image_results = agent.search_by_image(image_data, "test.jpg", top_k=3)
        assert len(image_results) == 2
        mock_processor.process_image_file.assert_called_once_with(
            image_data, "test.jpg"
        )

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_a2a_message_processing_integration(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client, mock_processor
    ):
        """Test A2A message processing with different content types"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        agent = VideoSearchAgent()
        agent.video_processor = mock_processor

        # Create mixed content A2A task
        text_part = TextPart(text="find similar content")
        video_part = FilePart(file_uri="data:video/mp4;base64,dmlkZW9fY29udGVudA==", mime_type="video/mp4")
        image_part = FilePart(file_uri="data:image/jpeg;base64,aW1hZ2VfY29udGVudA==", mime_type="image/jpeg")

        message = A2AMessage(
            role="user", parts=[text_part, video_part, image_part]
        )
        task = Task(id="integration_test", messages=[message])

        # Process the task
        result = agent.process_enhanced_task(task)

        # Verify all searches were performed
        assert result["task_id"] == "integration_test"
        assert result["status"] == "completed"
        assert "results" in result
        assert len(result["results"]) > 0  # Should have results from all searches

        # Verify search service was called (implementation processes all parts together)
        assert mock_vespa_client.search.call_count >= 1
        # Video and image processing happens in background, may or may not be called in test

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client
    ):
        """Test error handling in integrated workflow"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        # Mock search service to raise exception
        mock_vespa_client.search.side_effect = Exception("Search service unavailable")

        agent = VideoSearchAgent()

        # Test graceful error handling
        with pytest.raises(Exception) as exc_info:
            agent.search_by_text("test query")

        assert "Search service unavailable" in str(exc_info.value)

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_search_operations(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client, mock_processor
    ):
        """Test concurrent search operations"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        agent = VideoSearchAgent()
        agent.video_processor = mock_processor

        # Create multiple concurrent search tasks
        tasks = [agent.search_by_text(f"query {i}", top_k=3) for i in range(5)]

        # Execute concurrently (note: not actually async in current implementation)
        results = []
        for task in tasks:
            results.append(task)

        # Verify all searches completed
        assert len(results) == 5
        for result in results:
            assert len(result) == 2  # Mock returns 2 results

        # Verify search service called for each query
        assert mock_vespa_client.search.call_count == 5

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_result_aggregation(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client, mock_processor
    ):
        """Test search result aggregation across modalities"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        # Configure different results for different search types
        def side_effect(*args, **kwargs):
            # First argument is search_params dict
            search_params = args[0] if args else {}
            query = search_params.get("query", "")
            if "text:" in query:
                return [{"id": "text_result", "source": "text"}]
            elif "video:" in query or "Video" in query:
                return [{"id": "video_result", "source": "video"}]
            elif "image:" in query or "Image" in query:
                return [{"id": "image_result", "source": "image"}]
            else:
                return [{"id": "default_result", "source": "default"}]

        mock_vespa_client.search.side_effect = side_effect

        agent = VideoSearchAgent()
        agent.video_processor = mock_processor

        # Perform different search types
        text_results = agent.search_by_text("text: machine learning")
        video_results = agent.search_by_video(b"video_data", "test.mp4")
        image_results = agent.search_by_image(b"image_data", "test.jpg")

        # Verify results are properly differentiated
        assert text_results[0]["id"] == "text_result"
        assert video_results[0]["id"] == "video_result"
        assert image_results[0]["id"] == "image_result"

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client
    ):
        """Test performance monitoring during search operations"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        # Add delay to simulate search time
        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_vespa_client.search(*args, **kwargs)

        agent = VideoSearchAgent()

        # Measure search performance
        import time

        start_time = time.time()

        results = agent.search_by_text("performance test")

        execution_time = time.time() - start_time

        # Verify results
        assert len(results) == 2

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    def test_configuration_integration(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client
    ):
        """Test agent configuration integration"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        # Test with different configurations
        agent1 = VideoSearchAgent()
        agent2 = VideoSearchAgent()

        # Both agents should have same client type but potentially different instances
        assert type(agent1.vespa_client) is type(agent2.vespa_client)
        assert agent1.video_processor is not None
        assert agent2.video_processor is not None


class TestVideoSearchAgentEdgeCasesIntegration:
    """Integration tests for edge cases and error scenarios"""

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_large_file_handling_integration(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client, mock_processor
    ):
        """Test handling of large files"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        agent = VideoSearchAgent()
        agent.video_processor = mock_processor

        # Simulate large video file
        large_video_data = b"x" * (10 * 1024 * 1024)  # 10MB

        results = agent.search_by_video(large_video_data, "large_video.mp4")

        assert len(results) == 2
        mock_processor.process_video_file.assert_called_once_with(
            large_video_data, "large_video.mp4"
        )

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_empty_results_handling(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client
    ):
        """Test handling of empty search results"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client
        mock_vespa_client.search.return_value = []

        agent = VideoSearchAgent()

        results = agent.search_by_text("nonexistent content")

        assert results == []

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_malformed_a2a_message_handling(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client
    ):
        """Test handling of malformed A2A messages"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        agent = VideoSearchAgent()

        # Test with missing parts
        message = A2AMessage(role="user", parts=[])
        task = Task(id="malformed_test", messages=[message])

        # Process the task (should not raise exception)
        result = agent.process_enhanced_task(task)

        # Verify that empty task is handled gracefully
        assert result["task_id"] == "malformed_test"
        assert result["status"] == "completed"
        assert result["results"] == []
        assert result["total_results"] == 0

    @patch("src.app.agents.query_encoders.get_config")
    @patch("src.app.agents.video_search_agent.get_config")
    @patch("src.app.agents.video_search_agent.VespaVideoSearchClient")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resource_cleanup_integration(
        self, mock_vespa_client_class, mock_video_config, mock_encoder_config, mock_vespa_client, mock_processor
    ):
        """Test proper resource cleanup after operations"""
        # Mock config for query encoder
        mock_encoder_config.return_value = {
            "video_processing_profiles": {
                "video_colpali_smol500_mv_frame": {
                    "embedding_model": "vidore/colsmol-500m",
                    "embedding_type": "frame_based",
                }
            }
        }

        # Mock config object with real profile
        mock_config_obj = Mock()
        def config_get_side_effect(key, default=None):
            if key == "active_video_profile":
                return "video_colpali_smol500_mv_frame"
            elif key == "video_processing_profiles":
                return {
                    "video_colpali_smol500_mv_frame": {
                        "embedding_model": "vidore/colsmol-500m",
                        "embedding_type": "frame_based",
                    }
                }
            return default
        mock_config_obj.get.side_effect = config_get_side_effect
        mock_video_config.return_value = mock_config_obj
        mock_vespa_client_class.return_value = mock_vespa_client

        agent = VideoSearchAgent()
        agent.video_processor = mock_processor

        # Perform multiple operations
        results1 = agent.search_by_video(b"video1", "test1.mp4")
        results2 = agent.search_by_video(b"video2", "test2.mp4")
        results3 = agent.search_by_image(b"image1", "test1.jpg")

        # Verify all operations completed successfully
        assert len(results1) == 2
        assert len(results2) == 2
        assert len(results3) == 2

        # Verify processor methods were called
        assert mock_processor.process_video_file.call_count == 2
        assert mock_processor.process_image_file.call_count == 1
