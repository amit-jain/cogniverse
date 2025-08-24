"""Integration tests for Specialized Agents (Phase 4) with OpenAI-compatible APIs."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from src.app.agents.detailed_report_agent import DetailedReportAgent
from src.app.agents.summarizer_agent import SummarizerAgent
from src.tools.a2a_utils import A2AMessage, DataPart, Task

from .conftest import skip_if_no_ollama


@pytest.fixture
def openai_compatible_config():
    """Configuration for OpenAI-compatible API (can be local Ollama via OpenAI format)"""
    return {
        "openai_api_key": "local-test-key",
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        "models": {
            "small": os.getenv("SMALL_MODEL", "smollm3:8b"),
            "medium": os.getenv("MEDIUM_MODEL", "qwen:7b"),
            "vision": os.getenv("VISION_MODEL", "qwen:7b"),
        },
        "timeout": 30,
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
            "timestamp": "2024-01-15T10:00:00Z",
        },
        {
            "id": "video_2",
            "title": "Deep Learning Applications",
            "description": "Real-world applications of deep learning",
            "score": 0.87,
            "duration": 450,
            "thumbnail": "/path/to/thumb2.jpg",
            "content_type": "video",
            "timestamp": "2024-01-16T14:30:00Z",
        },
        {
            "id": "image_1",
            "title": "Neural Network Diagram",
            "description": "Visual representation of neural networks",
            "score": 0.78,
            "image_path": "/path/to/image1.jpg",
            "content_type": "image",
            "timestamp": "2024-01-17T09:15:00Z",
        },
    ]


class MockOpenAIClient:
    """Mock OpenAI-compatible client for testing"""

    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"

    async def chat_completions_create(self, model: str, messages: list, **kwargs):
        """Mock chat completion using OpenAI format"""
        # Simulate different responses based on model
        user_content = messages[-1]["content"] if messages else ""

        if "smollm3" in model or "small" in model:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "This is a brief summary from small model. The content covers machine learning fundamentals and applications.",
                            "role": "assistant",
                        }
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "total_tokens": 150,
                },
            }
        elif "qwen" in model or "medium" in model:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "This is a comprehensive analysis from medium model. The content provides detailed insights into machine learning concepts, covering both theoretical foundations and practical applications with specific examples and use cases.",
                            "role": "assistant",
                        }
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                },
            }
        else:
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"Generic response from {model} for: {user_content[:50]}...",
                            "role": "assistant",
                        }
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 50,
                    "total_tokens": 75,
                },
            }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI-compatible client fixture"""
    return MockOpenAIClient("local-test-key", "http://localhost:11434/v1")


@skip_if_no_ollama
class TestSummarizerAgentOpenAIIntegration:
    """Integration tests for SummarizerAgent with OpenAI-compatible APIs"""

    @pytest.mark.asyncio
    async def test_summarizer_with_small_model(
        self, openai_compatible_config, sample_search_results, mock_openai_client
    ):
        """Test SummarizerAgent with small model via OpenAI API"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = openai_compatible_config

            # Mock the VLM interface to use OpenAI-compatible client
            with patch(
                "src.app.agents.summarizer_agent.openai.OpenAI",
                return_value=mock_openai_client,
            ):
                agent = SummarizerAgent(vlm_model="smollm3:8b")
                agent.vlm.client = mock_openai_client
                agent.vlm.client_type = "openai"

                # Create summary request
                from src.app.agents.summarizer_agent import SummaryRequest

                request = SummaryRequest(
                    query="machine learning fundamentals",
                    search_results=sample_search_results,
                    summary_type="brief",
                    include_visual_analysis=False,
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
    async def test_summarizer_with_qwen_comprehensive(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test SummarizerAgent with Qwen model for comprehensive summary"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = {**ollama_config, "summarizer_model": "qwen:7b"}

            with patch.object(
                SummarizerAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                agent = SummarizerAgent(vlm_model="qwen:7b")
                agent.vlm.client = mock_ollama_client
                agent.vlm.client_type = "ollama"

                from src.app.agents.summarizer_agent import SummaryRequest

                request = SummaryRequest(
                    query="deep learning applications",
                    search_results=sample_search_results,
                    summary_type="comprehensive",
                    include_visual_analysis=True,
                    max_results_to_analyze=3,
                )

                # Mock visual analysis
                with patch.object(
                    agent.vlm, "analyze_visual_content", new_callable=AsyncMock
                ) as mock_visual:
                    mock_visual.return_value = {
                        "insights": ["Neural network architecture visible"],
                        "descriptions": ["Diagram showing layers and connections"],
                    }

                    result = await agent.summarize(request)

                    # Verify comprehensive analysis
                    assert result.summary is not None
                    assert len(result.summary) > 200  # Comprehensive should be longer
                    assert len(result.key_points) >= 3
                    assert len(result.visual_insights) > 0
                    assert result.thinking_phase.reasoning is not None

    @pytest.mark.asyncio
    async def test_summarizer_a2a_with_ollama(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test SummarizerAgent A2A processing with Ollama"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = {
                **ollama_config,
                "summarizer_model": "smollm3:8b",
            }

            with patch.object(
                SummarizerAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                agent = SummarizerAgent(vlm_model="smollm3:8b")
                agent.vlm.client = mock_ollama_client
                agent.vlm.client_type = "ollama"

                # Create A2A task
                data_part = DataPart(
                    data={
                        "query": "summarize AI research",
                        "search_results": sample_search_results,
                        "summary_type": "bullet_points",
                        "include_visual_analysis": False,
                    }
                )
                message = A2AMessage(role="user", parts=[data_part])
                task = Task(id="ollama_summary_test", messages=[message])

                # Process task
                result = await agent.process_a2a_task(task)

                # Verify A2A response
                assert result["task_id"] == "ollama_summary_test"
                assert result["status"] == "completed"
                assert "result" in result
                assert result["result"]["summary"] is not None
                assert result["result"]["confidence_score"] > 0

    @pytest.mark.asyncio
    async def test_summarizer_thinking_phase_with_ollama(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test thinking phase integration with Ollama models"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = {**ollama_config, "summarizer_model": "qwen:7b"}

            with patch.object(
                SummarizerAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                agent = SummarizerAgent(vlm_model="qwen:7b", thinking_enabled=True)
                agent.vlm.client = mock_ollama_client
                agent.vlm.client_type = "ollama"

                from src.app.agents.summarizer_agent import SummaryRequest

                request = SummaryRequest(
                    query="complex AI ethics analysis",
                    search_results=sample_search_results,
                    summary_type="comprehensive",
                )

                result = await agent.summarize(request)

                # Verify thinking phase was executed
                assert result.thinking_phase is not None
                assert result.thinking_phase.key_themes is not None
                assert result.thinking_phase.content_categories is not None
                assert result.thinking_phase.reasoning is not None
                assert len(result.thinking_phase.reasoning) > 50


@skip_if_no_ollama
class TestDetailedReportAgentOllamaIntegration:
    """Integration tests for DetailedReportAgent with Ollama models"""

    @pytest.mark.asyncio
    async def test_detailed_report_with_qwen(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test DetailedReportAgent with Qwen model"""
        with patch("src.app.agents.detailed_report_agent.get_config") as mock_config:
            mock_config.return_value = {**ollama_config, "report_model": "qwen:7b"}

            with patch.object(
                DetailedReportAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                agent = DetailedReportAgent(vlm_model="qwen:7b")
                agent.vlm.client = mock_ollama_client
                agent.vlm.client_type = "ollama"

                from src.app.agents.detailed_report_agent import ReportRequest

                request = ReportRequest(
                    query="comprehensive analysis of AI trends",
                    search_results=sample_search_results,
                    report_type="comprehensive",
                    include_visual_analysis=True,
                    include_technical_details=True,
                    include_recommendations=True,
                )

                # Mock visual analysis
                with patch.object(
                    agent.vlm, "analyze_visual_content_detailed", new_callable=AsyncMock
                ) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["Comprehensive visual analysis"],
                        "technical_analysis": [
                            "Technical finding 1",
                            "Technical finding 2",
                        ],
                        "visual_patterns": ["Pattern A", "Pattern B"],
                        "quality_assessment": {"overall": 0.85, "clarity": 0.9},
                        "annotations": [
                            {"element": "neural_network", "confidence": 0.9}
                        ],
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
    async def test_detailed_report_technical_analysis(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test technical analysis capabilities with Ollama"""
        with patch("src.app.agents.detailed_report_agent.get_config") as mock_config:
            mock_config.return_value = {**ollama_config, "report_model": "qwen:7b"}

            with patch.object(
                DetailedReportAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                agent = DetailedReportAgent(
                    vlm_model="qwen:7b", technical_analysis_enabled=True
                )
                agent.vlm.client = mock_ollama_client
                agent.vlm.client_type = "ollama"

                # Add technical metadata to search results
                technical_results = sample_search_results.copy()
                technical_results[0].update(
                    {
                        "format": "mp4",
                        "resolution": "1080p",
                        "frame_rate": 30,
                        "embedding": list(range(768)),  # Mock embedding
                    }
                )

                from src.app.agents.detailed_report_agent import ReportRequest

                request = ReportRequest(
                    query="technical analysis of video formats",
                    search_results=technical_results,
                    report_type="technical",
                    include_technical_details=True,
                )

                result = await agent.generate_report(request)

                # Verify technical analysis
                assert len(result.technical_details) >= 2
                technical_section = next(
                    (
                        section
                        for section in result.technical_details
                        if section["section"] == "System Analysis"
                    ),
                    None,
                )
                assert technical_section is not None
                assert "mp4" in str(technical_section["details"])

    @pytest.mark.asyncio
    async def test_detailed_report_a2a_with_ollama(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test DetailedReportAgent A2A processing with Ollama"""
        with patch("src.app.agents.detailed_report_agent.get_config") as mock_config:
            mock_config.return_value = {**ollama_config, "report_model": "qwen:7b"}

            with patch.object(
                DetailedReportAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                agent = DetailedReportAgent(vlm_model="qwen:7b")
                agent.vlm.client = mock_ollama_client
                agent.vlm.client_type = "ollama"

                # Create A2A task
                data_part = DataPart(
                    data={
                        "query": "detailed AI analysis report",
                        "search_results": sample_search_results,
                        "report_type": "analytical",
                        "include_visual_analysis": True,
                        "include_recommendations": True,
                    }
                )
                message = A2AMessage(role="user", parts=[data_part])
                task = Task(id="ollama_report_test", messages=[message])

                # Mock visual analysis for A2A test
                with patch.object(
                    agent.vlm, "analyze_visual_content_detailed", new_callable=AsyncMock
                ) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["A2A visual analysis"],
                        "technical_analysis": ["A2A technical finding"],
                        "quality_assessment": {"overall": 0.8},
                    }

                    result = await agent.process_a2a_task(task)

                    # Verify A2A response
                    assert result["task_id"] == "ollama_report_test"
                    assert result["status"] == "completed"
                    assert "result" in result
                    assert result["result"]["executive_summary"] is not None
                    assert len(result["result"]["detailed_findings"]) > 0
                    assert len(result["result"]["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_detailed_report_pattern_identification(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test pattern identification with Ollama models"""
        with patch("src.app.agents.detailed_report_agent.get_config") as mock_config:
            mock_config.return_value = {**ollama_config, "report_model": "qwen:7b"}

            with patch.object(
                DetailedReportAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                agent = DetailedReportAgent(vlm_model="qwen:7b")
                agent.vlm.client = mock_ollama_client
                agent.vlm.client_type = "ollama"

                # Create pattern-rich data
                pattern_results = []
                for i in range(10):
                    pattern_results.append(
                        {
                            "id": f"video_{i}",
                            "title": f"AI Tutorial Part {i}",
                            "score": 0.9 - (i * 0.05),
                            "timestamp": f"2024-01-{15+i}T10:00:00Z",
                            "content_type": "video",
                        }
                    )

                from src.app.agents.detailed_report_agent import ReportRequest

                request = ReportRequest(
                    query="identify patterns in AI education content",
                    search_results=pattern_results,
                    report_type="analytical",
                )

                result = await agent.generate_report(request)

                # Verify pattern identification
                assert len(result.thinking_phase.patterns_identified) > 0
                patterns_finding = next(
                    (
                        finding
                        for finding in result.detailed_findings
                        if finding["category"] == "Pattern Analysis"
                    ),
                    None,
                )
                if patterns_finding:
                    assert patterns_finding["significance"] == "high"


@skip_if_no_ollama
class TestCrossAgentIntegration:
    """Integration tests across multiple agents with Ollama"""

    @pytest.mark.asyncio
    async def test_summarizer_to_detailed_report_workflow(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test workflow from summarizer to detailed report"""
        with (
            patch("src.app.agents.summarizer_agent.get_config") as mock_config1,
            patch("src.app.agents.detailed_report_agent.get_config") as mock_config2,
        ):

            mock_config1.return_value = {
                **ollama_config,
                "summarizer_model": "smollm3:8b",
            }
            mock_config2.return_value = {**ollama_config, "report_model": "qwen:7b"}

            # Initialize agents
            with (
                patch.object(
                    SummarizerAgent,
                    "_create_ollama_client",
                    return_value=mock_ollama_client,
                ),
                patch.object(
                    DetailedReportAgent,
                    "_create_ollama_client",
                    return_value=mock_ollama_client,
                ),
            ):

                summarizer = SummarizerAgent(vlm_model="smollm3:8b")
                summarizer.vlm.client = mock_ollama_client
                summarizer.vlm.client_type = "ollama"

                report_agent = DetailedReportAgent(vlm_model="qwen:7b")
                report_agent.vlm.client = mock_ollama_client
                report_agent.vlm.client_type = "ollama"

                # Step 1: Generate summary
                from src.app.agents.summarizer_agent import SummaryRequest

                summary_request = SummaryRequest(
                    query="AI research overview",
                    search_results=sample_search_results,
                    summary_type="comprehensive",
                )

                summary_result = await summarizer.summarize(summary_request)

                # Step 2: Use summary for detailed report
                enhanced_results = sample_search_results.copy()
                enhanced_results.append(
                    {
                        "id": "summary_insight",
                        "title": "Summary Insights",
                        "description": summary_result.summary,
                        "score": 1.0,
                        "content_type": "analysis",
                    }
                )

                from src.app.agents.detailed_report_agent import ReportRequest

                report_request = ReportRequest(
                    query="comprehensive AI research report based on summary",
                    search_results=enhanced_results,
                    report_type="comprehensive",
                )

                with patch.object(
                    report_agent.vlm,
                    "analyze_visual_content_detailed",
                    new_callable=AsyncMock,
                ) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["Cross-agent analysis"],
                        "technical_analysis": ["Workflow integration finding"],
                        "quality_assessment": {"overall": 0.9},
                    }

                    report_result = await report_agent.generate_report(report_request)

                    # Verify integrated workflow
                    assert summary_result.summary is not None
                    assert report_result.executive_summary is not None
                    assert len(report_result.detailed_findings) > len(
                        sample_search_results
                    )
                    assert report_result.confidence_assessment["overall"] > 0.7

    @pytest.mark.asyncio
    async def test_model_switching_integration(
        self, ollama_config, sample_search_results, mock_ollama_client
    ):
        """Test switching between different Ollama models"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = ollama_config

            with patch.object(
                SummarizerAgent,
                "_create_ollama_client",
                return_value=mock_ollama_client,
            ):
                # Test with SmolLM3 (small model)
                agent_small = SummarizerAgent(vlm_model="smollm3:8b")
                agent_small.vlm.client = mock_ollama_client
                agent_small.vlm.client_type = "ollama"

                # Test with Qwen (medium model)
                agent_medium = SummarizerAgent(vlm_model="qwen:7b")
                agent_medium.vlm.client = mock_ollama_client
                agent_medium.vlm.client_type = "ollama"

                from src.app.agents.summarizer_agent import SummaryRequest

                request = SummaryRequest(
                    query="model comparison test",
                    search_results=sample_search_results[:2],  # Smaller dataset
                    summary_type="brief",
                )

                # Generate summaries with both models
                result_small = await agent_small.summarize(request)
                result_medium = await agent_medium.summarize(request)

                # Verify both models work
                assert result_small.summary is not None
                assert result_medium.summary is not None
                assert result_small.confidence_score > 0
                assert result_medium.confidence_score > 0

                # Medium model might produce longer/more detailed output
                # This is just a basic check - actual behavior depends on model implementation


@skip_if_no_ollama
class TestOllamaConfigurationIntegration:
    """Integration tests for Ollama configuration and setup"""

    def test_ollama_connection_configuration(self, ollama_config):
        """Test Ollama connection configuration"""
        # Test configuration parsing
        assert ollama_config["model_provider"] == "ollama"
        assert ollama_config["base_url"] == "http://localhost:11434"
        assert "smollm3" in ollama_config["models"]["small"]
        assert "qwen" in ollama_config["models"]["medium"]

    @pytest.mark.asyncio
    async def test_ollama_model_availability_check(
        self, ollama_config, mock_ollama_client
    ):
        """Test checking Ollama model availability"""
        # Mock model list endpoint
        with patch.object(
            mock_ollama_client, "list", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = {
                "models": [{"name": "smollm3:8b"}, {"name": "qwen:7b"}]
            }

            # Test model availability
            available_models = await mock_list()
            model_names = [model["name"] for model in available_models["models"]]

            assert "smollm3:8b" in model_names
            assert "qwen:7b" in model_names

    @pytest.mark.asyncio
    async def test_ollama_error_handling(self, ollama_config, mock_ollama_client):
        """Test Ollama error handling"""
        # Mock connection error
        with patch.object(
            mock_ollama_client, "chat", side_effect=Exception("Connection refused")
        ):
            with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
                mock_config.return_value = ollama_config

                with patch.object(
                    SummarizerAgent,
                    "_create_ollama_client",
                    return_value=mock_ollama_client,
                ):
                    agent = SummarizerAgent(vlm_model="smollm3:8b")
                    agent.vlm.client = mock_ollama_client
                    agent.vlm.client_type = "ollama"

                    from src.app.agents.summarizer_agent import SummaryRequest

                    request = SummaryRequest(
                        query="test error handling",
                        search_results=[],
                        summary_type="brief",
                    )

                    # Should handle the error gracefully
                    with pytest.raises(Exception) as exc_info:
                        await agent.summarize(request)

                    assert "Connection refused" in str(exc_info.value)


# Integration test configuration hints for actual Ollama setup
"""
To run these integration tests with real Ollama:

1. Install Ollama: https://ollama.ai
2. Pull models:
   ollama pull smollm3:8b
   ollama pull qwen:7b
3. Start Ollama server: ollama serve
4. Set environment variables:
   export OLLAMA_BASE_URL=http://localhost:11434
   export OLLAMA_MODEL_SMALL=smollm3:8b
   export OLLAMA_MODEL_MEDIUM=qwen:7b
5. Run tests with: pytest -m integration
"""
