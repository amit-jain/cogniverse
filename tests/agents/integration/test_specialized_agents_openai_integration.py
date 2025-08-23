"""Integration tests for Specialized Agents (Phase 4) with OpenAI-compatible APIs.

These tests use OpenAI API format but can be configured to use local models via Ollama's
OpenAI compatibility layer or other compatible services like litellm.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from src.app.agents.summarizer_agent import SummarizerAgent
from src.app.agents.detailed_report_agent import DetailedReportAgent
from src.tools.a2a_utils import A2AMessage, DataPart, Task


@pytest.fixture
def openai_compatible_config():
    """Configuration for OpenAI-compatible API (local via Ollama or litellm)"""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", "local-test-key"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        "models": {
            "small": os.getenv("SMALL_MODEL", "smollm3:8b"),
            "medium": os.getenv("MEDIUM_MODEL", "qwen:7b"),
            "vision": os.getenv("VISION_MODEL", "qwen:7b")
        },
        "timeout": 30
    }


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        {
            "id": "video_1",
            "title": "Introduction to Machine Learning",
            "description": "Basic concepts and fundamentals of ML",
            "score": 0.95,
            "duration": 300,
            "thumbnail": "/path/to/thumb1.jpg",
            "content_type": "video",
            "timestamp": "2024-01-15T10:00:00Z"
        },
        {
            "id": "video_2", 
            "title": "Deep Learning Applications",
            "description": "Real-world applications of deep learning",
            "score": 0.87,
            "duration": 450,
            "thumbnail": "/path/to/thumb2.jpg",
            "content_type": "video",
            "timestamp": "2024-01-16T14:30:00Z"
        },
        {
            "id": "image_1",
            "title": "Neural Network Diagram",
            "description": "Visual representation of neural networks",
            "score": 0.78,
            "image_path": "/path/to/image1.jpg",
            "content_type": "image",
            "timestamp": "2024-01-17T09:15:00Z"
        }
    ]


class MockOpenAIClient:
    """Mock OpenAI-compatible client for testing"""
    
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = AsyncMock(side_effect=self.chat_completions_create)
        
    async def chat_completions_create(self, model: str, messages: list, **kwargs):
        """Mock chat completion using OpenAI format"""
        user_content = messages[-1]["content"] if messages else ""
        
        if "smollm3" in model or "small" in model:
            return MockChatCompletion(
                content="This is a brief summary from small model. The content covers machine learning fundamentals and applications.",
                model=model
            )
        elif "qwen" in model or "medium" in model:
            return MockChatCompletion(
                content="This is a comprehensive analysis from medium model. The content provides detailed insights into machine learning concepts, covering both theoretical foundations and practical applications with specific examples and use cases.",
                model=model
            )
        else:
            return MockChatCompletion(
                content=f"Generic response from {model} for: {user_content[:50]}...",
                model=model
            )


class MockChatCompletion:
    """Mock chat completion response"""
    
    def __init__(self, content: str, model: str):
        self.choices = [MockChoice(content)]
        self.model = model
        self.usage = MockUsage()


class MockChoice:
    """Mock choice in chat completion"""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message in chat completion"""
    
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


class MockUsage:
    """Mock usage statistics"""
    
    def __init__(self):
        self.prompt_tokens = 100
        self.completion_tokens = 200
        self.total_tokens = 300


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI-compatible client fixture"""
    return MockOpenAIClient("local-test-key", "http://localhost:11434/v1")


class TestSummarizerAgentOpenAIIntegration:
    """Integration tests for SummarizerAgent with OpenAI-compatible APIs"""
    
    @pytest.mark.asyncio
    async def test_summarizer_with_small_model(self, openai_compatible_config, sample_search_results, mock_openai_client):
        """Test SummarizerAgent with small model via OpenAI API"""
        with patch('src.app.agents.summarizer_agent.get_config') as mock_config:
            mock_config.return_value = openai_compatible_config
            
            # Mock the VLM interface to use OpenAI client
            with patch('src.app.agents.summarizer_agent.openai.OpenAI', return_value=mock_openai_client):
                agent = SummarizerAgent(vlm_model="smollm3:8b")
                agent.vlm.client = mock_openai_client
                agent.vlm.client_type = "openai"
                
                # Create summary request
                from src.app.agents.summarizer_agent import SummaryRequest
                request = SummaryRequest(
                    query="machine learning fundamentals",
                    search_results=sample_search_results,
                    summary_type="brief",
                    include_visual_analysis=False
                )
                
                # Generate summary
                result = await agent.summarize(request)
                
                # Verify results
                assert result.summary is not None
                assert "machine learning" in result.summary.lower()
                assert len(result.key_points) >= 1
                assert result.confidence_score > 0
                assert result.metadata["summary_type"] == "brief"
    
    @pytest.mark.asyncio
    async def test_summarizer_with_medium_model_comprehensive(self, openai_compatible_config, sample_search_results, mock_openai_client):
        """Test SummarizerAgent with medium model for comprehensive summary"""
        with patch('src.app.agents.summarizer_agent.get_config') as mock_config:
            mock_config.return_value = openai_compatible_config
            
            with patch('src.app.agents.summarizer_agent.openai.OpenAI', return_value=mock_openai_client):
                agent = SummarizerAgent(vlm_model="qwen:7b")
                agent.vlm.client = mock_openai_client
                agent.vlm.client_type = "openai"
                
                from src.app.agents.summarizer_agent import SummaryRequest
                request = SummaryRequest(
                    query="deep learning applications",
                    search_results=sample_search_results,
                    summary_type="comprehensive",
                    include_visual_analysis=True,
                    max_results_to_analyze=3
                )
                
                # Mock visual analysis
                with patch.object(agent.vlm, 'analyze_visual_content', new_callable=AsyncMock) as mock_visual:
                    mock_visual.return_value = {
                        "insights": ["Neural network architecture visible"],
                        "descriptions": ["Diagram showing layers and connections"]
                    }
                    
                    result = await agent.summarize(request)
                    
                    # Verify comprehensive analysis
                    assert result.summary is not None
                    assert len(result.summary) > 200  # Comprehensive should be longer
                    assert len(result.key_points) >= 3
                    assert len(result.visual_insights) > 0
                    assert result.thinking_phase.reasoning is not None
    
    @pytest.mark.asyncio
    async def test_summarizer_a2a_with_openai_api(self, openai_compatible_config, sample_search_results, mock_openai_client):
        """Test SummarizerAgent A2A processing with OpenAI-compatible API"""
        with patch('src.app.agents.summarizer_agent.get_config') as mock_config:
            mock_config.return_value = openai_compatible_config
            
            with patch('src.app.agents.summarizer_agent.openai.OpenAI', return_value=mock_openai_client):
                agent = SummarizerAgent(vlm_model="smollm3:8b")
                agent.vlm.client = mock_openai_client
                agent.vlm.client_type = "openai"
                
                # Create A2A task
                data_part = DataPart(data={
                    "query": "summarize AI research",
                    "search_results": sample_search_results,
                    "summary_type": "bullet_points",
                    "include_visual_analysis": False
                })
                message = A2AMessage(role="user", parts=[data_part])
                task = Task(id="openai_summary_test", messages=[message])
                
                # Process task
                result = await agent.process_a2a_task(task)
                
                # Verify A2A response
                assert result["task_id"] == "openai_summary_test"
                assert result["status"] == "completed"
                assert "result" in result
                assert result["result"]["summary"] is not None
                assert result["result"]["confidence_score"] > 0


class TestDetailedReportAgentOpenAIIntegration:
    """Integration tests for DetailedReportAgent with OpenAI-compatible APIs"""
    
    @pytest.mark.asyncio
    async def test_detailed_report_with_medium_model(self, openai_compatible_config, sample_search_results, mock_openai_client):
        """Test DetailedReportAgent with medium model"""
        with patch('src.app.agents.detailed_report_agent.get_config') as mock_config:
            mock_config.return_value = openai_compatible_config
            
            with patch('src.app.agents.detailed_report_agent.openai.OpenAI', return_value=mock_openai_client):
                agent = DetailedReportAgent(vlm_model="qwen:7b")
                agent.vlm.client = mock_openai_client
                agent.vlm.client_type = "openai"
                
                from src.app.agents.detailed_report_agent import ReportRequest
                request = ReportRequest(
                    query="comprehensive analysis of AI trends",
                    search_results=sample_search_results,
                    report_type="comprehensive",
                    include_visual_analysis=True,
                    include_technical_details=True,
                    include_recommendations=True
                )
                
                # Mock visual analysis
                with patch.object(agent.vlm, 'analyze_visual_content_detailed', new_callable=AsyncMock) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["Comprehensive visual analysis"],
                        "technical_analysis": ["Technical finding 1", "Technical finding 2"],
                        "visual_patterns": ["Pattern A", "Pattern B"],
                        "quality_assessment": {"overall": 0.85, "clarity": 0.9},
                        "annotations": [{"element": "neural_network", "confidence": 0.9}]
                    }
                    
                    result = await agent.generate_report(request)
                    
                    # Verify comprehensive report
                    assert result.executive_summary is not None
                    assert len(result.detailed_findings) >= 3
                    assert len(result.visual_analysis) > 0
                    assert len(result.technical_details) >= 2
                    assert len(result.recommendations) > 0
                    assert result.confidence_assessment["overall"] > 0
    
    @pytest.mark.asyncio
    async def test_detailed_report_a2a_with_openai_api(self, openai_compatible_config, sample_search_results, mock_openai_client):
        """Test DetailedReportAgent A2A processing with OpenAI-compatible API"""
        with patch('src.app.agents.detailed_report_agent.get_config') as mock_config:
            mock_config.return_value = openai_compatible_config
            
            with patch('src.app.agents.detailed_report_agent.openai.OpenAI', return_value=mock_openai_client):
                agent = DetailedReportAgent(vlm_model="qwen:7b")
                agent.vlm.client = mock_openai_client
                agent.vlm.client_type = "openai"
                
                # Create A2A task
                data_part = DataPart(data={
                    "query": "detailed AI analysis report",
                    "search_results": sample_search_results,
                    "report_type": "analytical",
                    "include_visual_analysis": True,
                    "include_recommendations": True
                })
                message = A2AMessage(role="user", parts=[data_part])
                task = Task(id="openai_report_test", messages=[message])
                
                # Mock visual analysis for A2A test
                with patch.object(agent.vlm, 'analyze_visual_content_detailed', new_callable=AsyncMock) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["A2A visual analysis"],
                        "technical_analysis": ["A2A technical finding"],
                        "quality_assessment": {"overall": 0.8}
                    }
                    
                    result = await agent.process_a2a_task(task)
                    
                    # Verify A2A response
                    assert result["task_id"] == "openai_report_test"
                    assert result["status"] == "completed"
                    assert "result" in result
                    assert result["result"]["executive_summary"] is not None
                    assert len(result["result"]["detailed_findings"]) > 0
                    assert len(result["result"]["recommendations"]) > 0


class TestCrossAgentIntegrationWithOpenAI:
    """Integration tests across multiple agents with OpenAI-compatible APIs"""
    
    @pytest.mark.asyncio
    async def test_summarizer_to_detailed_report_workflow(self, openai_compatible_config, sample_search_results, mock_openai_client):
        """Test workflow from summarizer to detailed report"""
        with patch('src.app.agents.summarizer_agent.get_config') as mock_config1, \
             patch('src.app.agents.detailed_report_agent.get_config') as mock_config2:
            
            mock_config1.return_value = openai_compatible_config
            mock_config2.return_value = openai_compatible_config
            
            # Initialize agents
            with patch('src.app.agents.summarizer_agent.openai.OpenAI', return_value=mock_openai_client), \
                 patch('src.app.agents.detailed_report_agent.openai.OpenAI', return_value=mock_openai_client):
                
                summarizer = SummarizerAgent(vlm_model="smollm3:8b")
                summarizer.vlm.client = mock_openai_client
                summarizer.vlm.client_type = "openai"
                
                report_agent = DetailedReportAgent(vlm_model="qwen:7b") 
                report_agent.vlm.client = mock_openai_client
                report_agent.vlm.client_type = "openai"
                
                # Step 1: Generate summary
                from src.app.agents.summarizer_agent import SummaryRequest
                summary_request = SummaryRequest(
                    query="AI research overview",
                    search_results=sample_search_results,
                    summary_type="comprehensive"
                )
                
                summary_result = await summarizer.summarize(summary_request)
                
                # Step 2: Use summary for detailed report
                enhanced_results = sample_search_results.copy()
                enhanced_results.append({
                    "id": "summary_insight",
                    "title": "Summary Insights",
                    "description": summary_result.summary,
                    "score": 1.0,
                    "content_type": "analysis"
                })
                
                from src.app.agents.detailed_report_agent import ReportRequest
                report_request = ReportRequest(
                    query="comprehensive AI research report based on summary",
                    search_results=enhanced_results,
                    report_type="comprehensive"
                )
                
                with patch.object(report_agent.vlm, 'analyze_visual_content_detailed', new_callable=AsyncMock) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["Cross-agent analysis"],
                        "technical_analysis": ["Workflow integration finding"],
                        "quality_assessment": {"overall": 0.9}
                    }
                    
                    report_result = await report_agent.generate_report(report_request)
                    
                    # Verify integrated workflow
                    assert summary_result.summary is not None
                    assert report_result.executive_summary is not None
                    assert len(report_result.detailed_findings) > len(sample_search_results)
                    assert report_result.confidence_assessment["overall"] > 0.7


class TestOpenAIConfigurationIntegration:
    """Integration tests for OpenAI-compatible configuration and setup"""
    
    def test_openai_connection_configuration(self, openai_compatible_config):
        """Test OpenAI-compatible connection configuration"""
        # Test configuration parsing
        assert "openai_api_key" in openai_compatible_config
        assert "openai_base_url" in openai_compatible_config
        assert "smollm3" in openai_compatible_config["models"]["small"]
        assert "qwen" in openai_compatible_config["models"]["medium"]
    
    @pytest.mark.asyncio
    async def test_openai_client_initialization(self, openai_compatible_config, mock_openai_client):
        """Test OpenAI client initialization"""
        # Test client creation
        assert mock_openai_client.api_key == "local-test-key"
        assert mock_openai_client.base_url == "http://localhost:11434/v1"
        
        # Test chat completion interface
        messages = [{"role": "user", "content": "test message"}]
        response = await mock_openai_client.chat_completions_create("smollm3:8b", messages)
        
        assert response.choices[0].message.content is not None
        assert response.model == "smollm3:8b"
    
    @pytest.mark.asyncio 
    async def test_openai_error_handling(self, openai_compatible_config, mock_openai_client):
        """Test OpenAI API error handling"""
        # Mock connection error
        mock_openai_client.chat.completions.create.side_effect = Exception("Connection refused")
        
        with patch('src.app.agents.summarizer_agent.get_config') as mock_config:
            mock_config.return_value = openai_compatible_config
            
            with patch('src.app.agents.summarizer_agent.openai.OpenAI', return_value=mock_openai_client):
                agent = SummarizerAgent(vlm_model="smollm3:8b")
                agent.vlm.client = mock_openai_client
                agent.vlm.client_type = "openai"
                
                from src.app.agents.summarizer_agent import SummaryRequest
                request = SummaryRequest(
                    query="test error handling",
                    search_results=[],
                    summary_type="brief"
                )
                
                # Should handle the error gracefully
                with pytest.raises(Exception) as exc_info:
                    await agent.summarize(request)
                
                assert "Connection refused" in str(exc_info.value)


# Configuration hints for running with real local models
"""
To run these integration tests with real local models via OpenAI-compatible API:

Option 1: Using Ollama with OpenAI compatibility
1. Install Ollama: https://ollama.ai
2. Pull models:
   ollama pull smollm3:8b
   ollama pull qwen:7b
3. Start Ollama with OpenAI compatibility:
   OLLAMA_ORIGINS="*" ollama serve
4. Set environment variables:
   export OPENAI_BASE_URL=http://localhost:11434/v1
   export OPENAI_API_KEY=local-test-key
   export SMALL_MODEL=smollm3:8b
   export MEDIUM_MODEL=qwen:7b

Option 2: Using litellm proxy
1. Install litellm: pip install litellm
2. Start litellm proxy:
   litellm --model ollama/smollm3:8b --model ollama/qwen:7b
3. Set environment variables:
   export OPENAI_BASE_URL=http://localhost:8000/v1
   export OPENAI_API_KEY=sk-1234

Run tests with: pytest -m integration -v
"""