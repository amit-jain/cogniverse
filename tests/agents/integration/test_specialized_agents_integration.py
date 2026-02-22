"""Integration tests for Specialized Agents with DSPy.LM through Ollama."""

from unittest.mock import AsyncMock, patch

import dspy
import pytest

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
)
from cogniverse_agents.summarizer_agent import SummarizerAgent, SummarizerDeps
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager


@pytest.fixture
def real_dspy_lm():
    """Real DSPy.LM configured for Ollama"""
    # Check if Ollama is available
    import requests

    response = requests.get("http://localhost:11434/v1/models", timeout=2)
    assert (
        response.status_code == 200
    ), "Ollama server must be running at localhost:11434"

    # Configure real DSPy.LM with Ollama via factory
    lm = create_dspy_lm(
        LLMEndpointConfig(
            model="ollama/qwen2.5:1.5b",
            api_base="http://localhost:11434",
        )
    )
    # Test the connection using correct DSPy API
    test_response = lm("test")
    assert test_response is not None
    return lm


@pytest.fixture
def test_config_manager():
    """Real ConfigManager for integration tests."""
    return create_default_config_manager()


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
        self,
        sample_search_results,
        real_dspy_lm,
        mock_ollama_server,
        test_config_manager,
    ):
        """Test SummarizerAgent with small model via real DSPy.LM"""
        with dspy.context(lm=real_dspy_lm):
            agent = SummarizerAgent(
                deps=SummarizerDeps(), config_manager=test_config_manager
            )

            from cogniverse_agents.summarizer_agent import SummaryRequest

            request = SummaryRequest(
                query="machine learning fundamentals",
                search_results=sample_search_results,
                summary_type="brief",
                include_visual_analysis=False,
            )

            result = await agent._summarize(request)

            assert result.summary is not None
            assert len(result.summary) > 10
            assert len(result.key_points) >= 1
            assert result.confidence_score > 0
            assert result.metadata["summary_type"] == "brief"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_summarizer_a2a_processing(
        self,
        sample_search_results,
        real_dspy_lm,
        mock_ollama_server,
        test_config_manager,
    ):
        """Test SummarizerAgent A2A processing with real DSPy.LM"""
        with dspy.context(lm=real_dspy_lm):
            agent = SummarizerAgent(
                deps=SummarizerDeps(), config_manager=test_config_manager
            )

            from cogniverse_agents.summarizer_agent import SummaryRequest

            request = SummaryRequest(
                query="summarize AI research",
                search_results=sample_search_results,
                summary_type="bullet_points",
                include_visual_analysis=False,
            )

            result = await agent._summarize(request)

            assert result.summary is not None
            assert len(result.summary) > 10
            assert result.confidence_score > 0


@pytest.mark.requires_ollama
class TestDetailedReportAgentDSPyIntegration:
    """Integration tests for DetailedReportAgent with DSPy.LM through Ollama"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_detailed_report_with_dspy(
        self,
        sample_search_results,
        real_dspy_lm,
        mock_ollama_server,
        test_config_manager,
    ):
        """Test DetailedReportAgent with real DSPy.LM"""
        with dspy.context(lm=real_dspy_lm):
            agent = DetailedReportAgent(
                deps=DetailedReportDeps(), config_manager=test_config_manager
            )

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

                result = await agent._generate_report(request)

                assert result.executive_summary is not None
                assert len(result.executive_summary) > 20
                assert len(result.detailed_findings) >= 3
                assert len(result.visual_analysis) > 0
                assert len(result.technical_details) >= 2
                assert len(result.recommendations) > 0
                assert result.confidence_assessment["overall"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_detailed_report_a2a_processing(
        self,
        sample_search_results,
        real_dspy_lm,
        mock_ollama_server,
        test_config_manager,
    ):
        """Test DetailedReportAgent A2A processing with real DSPy.LM"""
        with dspy.context(lm=real_dspy_lm):
            agent = DetailedReportAgent(
                deps=DetailedReportDeps(), config_manager=test_config_manager
            )

            from cogniverse_agents.detailed_report_agent import ReportRequest

            request = ReportRequest(
                query="detailed AI research report",
                search_results=sample_search_results,
                report_type="comprehensive",
                include_visual_analysis=True,
            )

            with patch.object(
                agent, "_perform_visual_analysis", new_callable=AsyncMock
            ) as mock_visual:
                mock_visual.return_value = {
                    "detailed_descriptions": ["Integration visual analysis"],
                    "technical_analysis": ["Integration technical finding"],
                    "quality_assessment": {"overall": 0.8},
                }

                result = await agent._generate_report(request)

                assert result.executive_summary is not None
                assert len(result.executive_summary) > 20
                assert len(result.detailed_findings) > 0
                assert len(result.recommendations) > 0


@pytest.mark.requires_ollama
class TestCrossAgentDSPyIntegration:
    """Integration tests across multiple agents with DSPy.LM"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_summarizer_to_detailed_report_workflow(
        self,
        sample_search_results,
        real_dspy_lm,
        mock_ollama_server,
        test_config_manager,
    ):
        """Test workflow from summarizer to detailed report using real DSPy.LM"""
        with dspy.context(lm=real_dspy_lm):
            summarizer = SummarizerAgent(
                deps=SummarizerDeps(), config_manager=test_config_manager
            )
            report_agent = DetailedReportAgent(
                deps=DetailedReportDeps(), config_manager=test_config_manager
            )

            from cogniverse_agents.summarizer_agent import SummaryRequest

            summary_request = SummaryRequest(
                query="AI research overview",
                search_results=sample_search_results,
                summary_type="comprehensive",
            )

            summary_result = await summarizer._summarize(summary_request)

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

                report_result = await report_agent._generate_report(report_request)

                assert summary_result.summary is not None
                assert len(summary_result.summary) > 20
                assert report_result.executive_summary is not None
                assert len(report_result.executive_summary) > 20
                assert len(report_result.detailed_findings) >= 3
                assert report_result.confidence_assessment["overall"] > 0.0


@pytest.mark.requires_ollama
class TestDSPyLMConfigurationIntegration:
    """Integration tests for DSPy.LM configuration and setup through Ollama"""

    def test_dspy_lm_configuration(self, test_config_manager):
        """Test DSPy.LM configuration from real config manager"""
        from cogniverse_foundation.config.utils import get_config

        config_helper = get_config(
            tenant_id="default", config_manager=test_config_manager
        )
        llm_config = config_helper.get_llm_config()
        assert llm_config.primary.model is not None
        assert len(llm_config.primary.model) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dspy_lm_model_switching(
        self,
        sample_search_results,
        real_dspy_lm,
        mock_ollama_server,
        test_config_manager,
    ):
        """Test switching between different models via real DSPy.LM"""
        with dspy.context(lm=real_dspy_lm):
            agent_small = SummarizerAgent(
                deps=SummarizerDeps(), config_manager=test_config_manager
            )
            agent_medium = SummarizerAgent(
                deps=SummarizerDeps(), config_manager=test_config_manager
            )

            from cogniverse_agents.summarizer_agent import SummaryRequest

            request = SummaryRequest(
                query="model comparison test",
                search_results=sample_search_results[:2],
                summary_type="brief",
            )

            result_small = await agent_small._summarize(request)
            result_medium = await agent_medium._summarize(request)

        assert result_small.summary is not None
        assert len(result_small.summary) > 10
        assert result_medium.summary is not None
        assert len(result_medium.summary) > 10
        assert result_small.confidence_score > 0
        assert result_medium.confidence_score > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_error_handling_with_bad_dspy_config(self, mock_ollama_server):
        """Test agent error handling when DSPy.LM configuration fails"""
        # Without config_manager, agent should raise ValueError
        with pytest.raises(ValueError, match="config_manager is required"):
            SummarizerAgent(deps=SummarizerDeps())


# Integration test configuration for DSPy.LM + Ollama
"""
To run these integration tests with real DSPy.LM + Ollama:

1. Install Ollama: https://ollama.ai
2. Pull models:
   ollama pull qwen2.5:1.5b  # Smallest model for tests
   ollama pull gemma3:4b      # Larger model for teacher/generation tests
3. Start Ollama server: ollama serve
4. Run tests with Ollama requirement:
   pytest -m requires_ollama tests/agents/integration/test_specialized_agents_integration.py
"""
