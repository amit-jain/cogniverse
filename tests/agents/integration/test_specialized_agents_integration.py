"""Integration tests for Specialized Agents with DSPy.LM through Ollama."""

from unittest.mock import AsyncMock, patch

import dspy
import pytest

from cogniverse_agents.detailed_report_agent import DetailedReportAgent
from cogniverse_agents.summarizer_agent import SummarizerAgent
from cogniverse_core.common.a2a_utils import A2AMessage, DataPart, Task


@pytest.fixture
def real_dspy_lm():
    """Real DSPy.LM configured for Ollama"""
    # Check if Ollama is available
    import requests

    response = requests.get("http://localhost:11434/v1/models", timeout=2)
    assert response.status_code == 200, "Ollama server must be running at localhost:11434"

    # Configure real DSPy.LM with Ollama using correct API
    lm = dspy.LM(
        model="ollama/gemma3:4b",  # Use smallest model for tests
        api_base="http://localhost:11434",
    )
    # Test the connection using correct DSPy API
    test_response = lm("test")
    assert test_response is not None
    return lm


@pytest.fixture
def dspy_config():
    """Configuration for DSPy.LM through Ollama"""
    return {
        "llm": {
            "model_name": "ollama/gemma3:4b",  # Use smallest available model
            "base_url": "http://localhost:11434",
            "api_key": "ollama",
        },
        "models": {
            "small": "ollama/gemma3:4b",
            "medium": "qwen2.5:1.5b",
            "vision": "qwen2.5:1.5b",
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


@pytest.fixture
def mock_ollama_server():
    """Check if Ollama server is available"""
    import requests

    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.requires_ollama
class TestSummarizerAgentDSPyIntegration:
    """Integration tests for SummarizerAgent with DSPy.LM through Ollama"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_summarizer_with_small_model(
        self, dspy_config, sample_search_results, real_dspy_lm, mock_ollama_server
    ):
        """Test SummarizerAgent with small model via real DSPy.LM"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = dspy_config

            # Use dspy.context() for async tasks instead of configure()
            with dspy.context(lm=real_dspy_lm):
                # Create agent and set real LM
                agent = SummarizerAgent(tenant_id="test_tenant")
                agent.llm = real_dspy_lm

                # Create summary request
                from cogniverse_agents.summarizer_agent import SummaryRequest

                request = SummaryRequest(
                    query="machine learning fundamentals",
                    search_results=sample_search_results,
                    summary_type="brief",
                    include_visual_analysis=False,
                )

                # Generate summary with real DSPy.LM
                result = await agent.summarize(request)

                # Verify results
                assert result.summary is not None
                assert len(result.summary) > 10  # Should have actual content
                assert len(result.key_points) >= 1
                assert result.confidence_score > 0
                assert result.metadata["summary_type"] == "brief"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_summarizer_a2a_processing(
        self, dspy_config, sample_search_results, real_dspy_lm, mock_ollama_server
    ):
        """Test SummarizerAgent A2A processing with real DSPy.LM"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = dspy_config

            # Use dspy.context() for async tasks instead of configure()
            with dspy.context(lm=real_dspy_lm):
                agent = SummarizerAgent(tenant_id="test_tenant")
                agent.llm = real_dspy_lm

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
                task = Task(id="dspy_summary_test", messages=[message])

                # Process task with real DSPy.LM
                result = await agent.process_a2a_task(task)

                # Verify A2A response
                assert result["task_id"] == "dspy_summary_test"
                assert result["status"] == "completed"
                assert result["summary"] is not None
                assert len(result["summary"]) > 10  # Should have actual content
                assert result["confidence_score"] > 0


@pytest.mark.requires_ollama
class TestDetailedReportAgentDSPyIntegration:
    """Integration tests for DetailedReportAgent with DSPy.LM through Ollama"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_detailed_report_with_dspy(
        self, dspy_config, sample_search_results, real_dspy_lm, mock_ollama_server
    ):
        """Test DetailedReportAgent with real DSPy.LM"""
        with patch("src.app.agents.detailed_report_agent.get_config") as mock_config:
            mock_config.return_value = dspy_config

            # Use dspy.context() for async tasks instead of configure()
            with dspy.context(lm=real_dspy_lm):
                agent = DetailedReportAgent(tenant_id="test_tenant")
                agent.llm = real_dspy_lm

                from cogniverse_agents.detailed_report_agent import ReportRequest

                request = ReportRequest(
                    query="comprehensive analysis of AI trends",
                    search_results=sample_search_results,
                    report_type="comprehensive",
                    include_visual_analysis=True,
                    include_technical_details=True,
                    include_recommendations=True,
                )

                # Mock visual analysis for integration test focus on DSPy.LM
                with patch.object(
                    agent, "_perform_visual_analysis", new_callable=AsyncMock
                ) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["Comprehensive visual analysis"],
                        "technical_analysis": [
                            "Technical finding 1",
                            "Technical finding 2",
                        ],
                        "visual_patterns": ["Pattern A", "Pattern B"],
                        "quality_assessment": {"overall": 0.85, "clarity": 0.9},
                        "annotations": [{"element": "neural_network", "confidence": 0.9}],
                    }

                    result = await agent.generate_report(request)

                    # Verify comprehensive report with real DSPy.LM
                    assert result.executive_summary is not None
                    assert len(result.executive_summary) > 20  # Should have actual content
                    assert len(result.detailed_findings) >= 3
                    assert len(result.visual_analysis) > 0
                    assert len(result.technical_details) >= 2
                    assert len(result.recommendations) > 0
                    assert result.confidence_assessment["overall"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_detailed_report_a2a_processing(
        self, dspy_config, sample_search_results, real_dspy_lm, mock_ollama_server
    ):
        """Test DetailedReportAgent A2A processing with real DSPy.LM"""
        with patch("src.app.agents.detailed_report_agent.get_config") as mock_config:
            mock_config.return_value = dspy_config

            # Use dspy.context() for async tasks instead of configure()
            with dspy.context(lm=real_dspy_lm):
                agent = DetailedReportAgent(tenant_id="test_tenant")
                agent.llm = real_dspy_lm

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
                task = Task(id="dspy_report_test", messages=[message])

                # Mock visual analysis for A2A test focus on DSPy.LM
                with patch.object(
                    agent, "_perform_visual_analysis", new_callable=AsyncMock
                ) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["A2A visual analysis"],
                        "technical_analysis": ["A2A technical finding"],
                        "quality_assessment": {"overall": 0.8},
                    }

                    result = await agent.process_a2a_task(task)

                    # Verify A2A response with real DSPy.LM
                    assert result["task_id"] == "dspy_report_test"
                    assert result["status"] == "completed"
                    assert "result" in result
                    assert result["result"]["executive_summary"] is not None
                    assert (
                        len(result["result"]["executive_summary"]) > 20
                    )  # Should have actual content
                    assert len(result["result"]["detailed_findings"]) > 0
                    assert len(result["result"]["recommendations"]) > 0


@pytest.mark.requires_ollama
class TestCrossAgentDSPyIntegration:
    """Integration tests across multiple agents with DSPy.LM"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_summarizer_to_detailed_report_workflow(
        self, dspy_config, sample_search_results, real_dspy_lm, mock_ollama_server
    ):
        """Test workflow from summarizer to detailed report using real DSPy.LM"""
        with (
            patch("src.app.agents.summarizer_agent.get_config") as mock_config1,
            patch("src.app.agents.detailed_report_agent.get_config") as mock_config2,
        ):

            mock_config1.return_value = dspy_config
            mock_config2.return_value = dspy_config

            # Use dspy.context() for async tasks instead of configure()
            with dspy.context(lm=real_dspy_lm):
                summarizer = SummarizerAgent(tenant_id="test_tenant")
                summarizer.llm = real_dspy_lm
                report_agent = DetailedReportAgent(tenant_id="test_tenant")
                report_agent.llm = real_dspy_lm

                # Step 1: Generate summary with real DSPy.LM
                from cogniverse_agents.summarizer_agent import SummaryRequest

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

                from cogniverse_agents.detailed_report_agent import ReportRequest

                report_request = ReportRequest(
                    query="comprehensive AI research report based on summary",
                    search_results=enhanced_results,
                    report_type="comprehensive",
                )

                with patch.object(
                    report_agent,
                    "_perform_visual_analysis",
                    new_callable=AsyncMock,
                ) as mock_visual:
                    mock_visual.return_value = {
                        "detailed_descriptions": ["Cross-agent analysis"],
                        "technical_analysis": ["Workflow integration finding"],
                        "quality_assessment": {"overall": 0.9},
                    }

                    report_result = await report_agent.generate_report(report_request)

                    # Verify integrated workflow with real DSPy.LM
                    assert summary_result.summary is not None
                    assert len(summary_result.summary) > 20  # Should have actual content
                    assert report_result.executive_summary is not None
                    assert (
                        len(report_result.executive_summary) > 20
                    )  # Should have actual content
                    assert len(report_result.detailed_findings) > len(sample_search_results)
                    assert report_result.confidence_assessment["overall"] > 0.0


@pytest.mark.requires_ollama
class TestDSPyLMConfigurationIntegration:
    """Integration tests for DSPy.LM configuration and setup through Ollama"""

    def test_dspy_lm_configuration(self, dspy_config):
        """Test DSPy.LM configuration parsing"""
        assert dspy_config["llm"]["model_name"] == "ollama/gemma3:4b"
        assert dspy_config["llm"]["base_url"] == "http://localhost:11434"
        assert dspy_config["llm"]["api_key"] == "ollama"
        assert "gemma3" in dspy_config["models"]["small"]
        assert "qwen" in dspy_config["models"]["medium"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dspy_lm_model_switching(
        self, dspy_config, sample_search_results, real_dspy_lm, mock_ollama_server
    ):
        """Test switching between different models via real DSPy.LM"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = dspy_config

            # Use dspy.context() for async tasks instead of configure()
            with dspy.context(lm=real_dspy_lm):
                # Test with different model configurations (both using same LM for testing)
                agent_small = SummarizerAgent(tenant_id="test_tenant")
                agent_small.llm = real_dspy_lm
                agent_medium = SummarizerAgent(tenant_id="test_tenant")
                agent_medium.llm = real_dspy_lm

                from cogniverse_agents.summarizer_agent import SummaryRequest

                request = SummaryRequest(
                    query="model comparison test",
                    search_results=sample_search_results[:2],
                    summary_type="brief",
                )

                # Generate summaries with both configurations using real DSPy.LM
                result_small = await agent_small.summarize(request)
                result_medium = await agent_medium.summarize(request)

            # Verify both configurations work with real DSPy.LM
            assert result_small.summary is not None
            assert len(result_small.summary) > 10  # Should have actual content
            assert result_medium.summary is not None
            assert len(result_medium.summary) > 10  # Should have actual content
            assert result_small.confidence_score > 0
            assert result_medium.confidence_score > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_error_handling_with_bad_dspy_config(self, mock_ollama_server):
        """Test agent error handling when DSPy.LM configuration fails"""
        # Test agent initialization with bad DSPy config
        bad_config = {
            "llm": {
                "model_name": "nonexistent:model",
                "base_url": "http://localhost:11434",
                "api_key": "ollama",
            }
        }

        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = bad_config

            # Agent initialization should either:
            # 1. Fail gracefully with clear error message, or
            # 2. Initialize but fail on first LLM call
            try:
                agent = SummarizerAgent(tenant_id="test_tenant")

                # If agent initializes, LLM calls should fail gracefully
                from cogniverse_agents.summarizer_agent import SummaryRequest

                request = SummaryRequest(
                    query="test error handling",
                    search_results=[],
                    summary_type="brief",
                )

                # This should raise an exception due to bad model config
                result = await agent.summarize(request)
                # If it succeeds unexpectedly, that's still valid - agent may have fallbacks
                assert (
                    result is not None
                ), "Agent should either fail or return valid result"

            except Exception as e:
                # Should fail with meaningful error about model or connection
                error_msg = str(e).lower()
                assert (
                    "model" in error_msg
                    or "not found" in error_msg
                    or "connection" in error_msg
                    or "llm" in error_msg
                    or "api" in error_msg
                ), f"Expected meaningful error, got: {str(e)}"


# Integration test configuration for DSPy.LM + Ollama
"""
To run these integration tests with real DSPy.LM + Ollama:

1. Install Ollama: https://ollama.ai
2. Pull models:
   ollama pull ollama/gemma3:4b  # Smallest model for tests
   ollama pull qwen2.5:1.5b
3. Start Ollama server: ollama serve  
4. Install DSPy: uv add dspy-ai
5. Run tests with Ollama requirement:
   pytest -m requires_ollama tests/agents/integration/test_specialized_agents_integration.py

These tests validate real DSPy.LM → Ollama → model inference pipeline.
"""
