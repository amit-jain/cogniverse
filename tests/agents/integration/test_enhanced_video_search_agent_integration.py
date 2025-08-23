"""Integration tests for Enhanced Video Search Agent (Phase 3)."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.app.agents.enhanced_video_search_agent import EnhancedVideoSearchAgent
from src.tools.a2a_utils import A2AMessage, DataPart, VideoPart, ImagePart, Task


@pytest.fixture
def mock_search_service():
    """Mock search service for integration testing"""
    service = Mock()
    service.search.return_value = [
        {
            "id": "video_1",
            "title": "Machine Learning Tutorial",
            "description": "Introduction to ML concepts",
            "thumbnail": "/path/to/thumb1.jpg",
            "score": 0.95
        },
        {
            "id": "video_2", 
            "title": "AI Ethics Discussion",
            "description": "Ethical considerations in AI",
            "thumbnail": "/path/to/thumb2.jpg",
            "score": 0.87
        }
    ]
    return service


@pytest.fixture
def mock_processor():
    """Mock video processor for integration testing"""
    processor = Mock()
    processor.encode_video.return_value = b"encoded_video_data"
    processor.encode_image.return_value = b"encoded_image_data"
    processor.cleanup_temp_files.return_value = None
    return processor


class TestEnhancedVideoSearchAgentIntegration:
    """Integration tests for Enhanced Video Search Agent functionality"""
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    def test_agent_initialization_with_real_config(self, mock_search_service_class, mock_search_service):
        """Test agent initialization with real configuration"""
        mock_search_service_class.return_value = mock_search_service
        
        agent = EnhancedVideoSearchAgent()
        
        assert agent.search_service == mock_search_service
        assert agent.video_processor is not None
        mock_search_service_class.assert_called_once()
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_multimodal_search_workflow_integration(self, mock_search_service_class, mock_search_service, mock_processor):
        """Test complete multimodal search workflow"""
        mock_search_service_class.return_value = mock_search_service
        
        agent = EnhancedVideoSearchAgent()
        agent.video_processor = mock_processor
        
        # Test text search
        text_results = agent.search_by_text("machine learning tutorial", top_k=5)
        assert len(text_results) == 2
        assert text_results[0]["title"] == "Machine Learning Tutorial"
        mock_search_service.search.assert_called_with(
            query="machine learning tutorial", top_k=5
        )
        
        # Test video search
        video_data = b"fake_video_data"
        video_results = agent.search_by_video(video_data, "test.mp4", top_k=3)
        assert len(video_results) == 2
        mock_processor.encode_video.assert_called_once_with(video_data, "test.mp4")
        
        # Test image search
        image_data = b"fake_image_data"
        image_results = agent.search_by_image(image_data, "test.jpg", top_k=3)
        assert len(image_results) == 2
        mock_processor.encode_image.assert_called_once_with(image_data, "test.jpg")
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_a2a_message_processing_integration(self, mock_search_service_class, mock_search_service, mock_processor):
        """Test A2A message processing with different content types"""
        mock_search_service_class.return_value = mock_search_service
        
        agent = EnhancedVideoSearchAgent()
        agent.video_processor = mock_processor
        
        # Create mixed content A2A task
        data_part = DataPart(data={"query": "find similar content", "top_k": 5})
        video_part = VideoPart(data=b"video_content", filename="query_video.mp4")
        image_part = ImagePart(data=b"image_content", filename="query_image.jpg")
        
        message = A2AMessage(role="user", parts=[data_part, video_part, image_part])
        task = Task(id="integration_test", messages=[message])
        
        # Process the task
        result = await agent.process_enhanced_task(task)
        
        # Verify all searches were performed
        assert result["task_id"] == "integration_test"
        assert result["status"] == "completed"
        assert "text_results" in result["results"]
        assert "video_results" in result["results"]
        assert "image_results" in result["results"]
        
        # Verify search service was called multiple times
        assert mock_search_service.search.call_count == 3
        mock_processor.encode_video.assert_called_once()
        mock_processor.encode_image.assert_called_once()
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_search_service_class, mock_search_service):
        """Test error handling in integrated workflow"""
        mock_search_service_class.return_value = mock_search_service
        
        # Mock search service to raise exception
        mock_search_service.search.side_effect = Exception("Search service unavailable")
        
        agent = EnhancedVideoSearchAgent()
        
        # Test graceful error handling
        with pytest.raises(Exception) as exc_info:
            agent.search_by_text("test query")
        
        assert "Search service unavailable" in str(exc_info.value)
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self, mock_search_service_class, mock_search_service, mock_processor):
        """Test concurrent search operations"""
        mock_search_service_class.return_value = mock_search_service
        
        agent = EnhancedVideoSearchAgent()
        agent.video_processor = mock_processor
        
        # Create multiple concurrent search tasks
        tasks = [
            agent.search_by_text(f"query {i}", top_k=3) 
            for i in range(5)
        ]
        
        # Execute concurrently (note: not actually async in current implementation)
        results = []
        for task in tasks:
            results.append(task)
        
        # Verify all searches completed
        assert len(results) == 5
        for result in results:
            assert len(result) == 2  # Mock returns 2 results
        
        # Verify search service called for each query
        assert mock_search_service.search.call_count == 5
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_search_result_aggregation(self, mock_search_service_class, mock_search_service, mock_processor):
        """Test search result aggregation across modalities"""
        mock_search_service_class.return_value = mock_search_service
        
        # Configure different results for different search types
        def side_effect(*args, **kwargs):
            query = kwargs.get('query', '')
            if 'text:' in query:
                return [{"id": "text_result", "source": "text"}]
            elif 'video:' in query:
                return [{"id": "video_result", "source": "video"}]
            else:
                return [{"id": "image_result", "source": "image"}]
        
        mock_search_service.search.side_effect = side_effect
        
        agent = EnhancedVideoSearchAgent()
        agent.video_processor = mock_processor
        
        # Perform different search types
        text_results = agent.search_by_text("text: machine learning")
        video_results = agent.search_by_video(b"video_data", "test.mp4")
        image_results = agent.search_by_image(b"image_data", "test.jpg")
        
        # Verify results are properly differentiated
        assert text_results[0]["id"] == "text_result"
        assert video_results[0]["id"] == "video_result"  
        assert image_results[0]["id"] == "image_result"
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, mock_search_service_class, mock_search_service):
        """Test performance monitoring during search operations"""
        mock_search_service_class.return_value = mock_search_service
        
        # Add delay to simulate search time
        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_search_service.search(*args, **kwargs)
        
        agent = EnhancedVideoSearchAgent()
        
        # Measure search performance
        import time
        start_time = time.time()
        
        results = agent.search_by_text("performance test")
        
        execution_time = time.time() - start_time
        
        # Verify results and timing
        assert len(results) == 2
        assert execution_time < 1.0  # Should be fast for mock
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    def test_configuration_integration(self, mock_search_service_class, mock_search_service):
        """Test agent configuration integration"""
        mock_search_service_class.return_value = mock_search_service
        
        # Test with different configurations
        agent1 = EnhancedVideoSearchAgent()
        agent2 = EnhancedVideoSearchAgent()
        
        # Both agents should have same service type but potentially different instances
        assert type(agent1.search_service) == type(agent2.search_service)
        assert agent1.video_processor is not None
        assert agent2.video_processor is not None


class TestEnhancedVideoSearchAgentEdgeCasesIntegration:
    """Integration tests for edge cases and error scenarios"""
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_large_file_handling_integration(self, mock_search_service_class, mock_search_service, mock_processor):
        """Test handling of large files"""
        mock_search_service_class.return_value = mock_search_service
        
        agent = EnhancedVideoSearchAgent()
        agent.video_processor = mock_processor
        
        # Simulate large video file
        large_video_data = b"x" * (10 * 1024 * 1024)  # 10MB
        
        results = agent.search_by_video(large_video_data, "large_video.mp4")
        
        assert len(results) == 2
        mock_processor.encode_video.assert_called_once_with(large_video_data, "large_video.mp4")
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_empty_results_handling(self, mock_search_service_class, mock_search_service):
        """Test handling of empty search results"""
        mock_search_service_class.return_value = mock_search_service
        mock_search_service.search.return_value = []
        
        agent = EnhancedVideoSearchAgent()
        
        results = agent.search_by_text("nonexistent content")
        
        assert results == []
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService')
    @pytest.mark.asyncio
    async def test_malformed_a2a_message_handling(self, mock_search_service_class, mock_search_service):
        """Test handling of malformed A2A messages"""
        mock_search_service_class.return_value = mock_search_service
        
        agent = EnhancedVideoSearchAgent()
        
        # Test with missing parts
        message = A2AMessage(role="user", parts=[])
        task = Task(id="malformed_test", messages=[message])
        
        with pytest.raises(Exception) as exc_info:
            await agent.process_enhanced_task(task)
        
        assert "No valid parts found" in str(exc_info.value)
    
    @patch('src.app.agents.enhanced_video_search_agent.SearchService') 
    @pytest.mark.asyncio
    async def test_resource_cleanup_integration(self, mock_search_service_class, mock_search_service, mock_processor):
        """Test proper resource cleanup after operations"""
        mock_search_service_class.return_value = mock_search_service
        
        agent = EnhancedVideoSearchAgent()
        agent.video_processor = mock_processor
        
        # Perform multiple operations
        agent.search_by_video(b"video1", "test1.mp4")
        agent.search_by_video(b"video2", "test2.mp4")
        agent.search_by_image(b"image1", "test1.jpg")
        
        # Verify cleanup was called for each operation
        assert mock_processor.cleanup_temp_files.call_count == 3