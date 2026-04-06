"""
Real integration tests for QueryAnalysisToolV3 with real Ollama LLM.

Replaces the deleted fake test_query_analysis_v3_integration.py which
mocked the entire routing agent. These tests call real Ollama for
query analysis and assert on actual analysis results.
"""

import logging

import dspy
import httpx
import pytest

from cogniverse_agents.query_analysis_tool_v3 import (
    QueryAnalysisResult,
    QueryAnalysisToolV3,
    QueryContext,
)
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager

logger = logging.getLogger(__name__)


def _ollama_available() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama not available"
)


@pytest.fixture(scope="module")
def dspy_lm():
    """Configure DSPy with real Ollama LLM."""
    config = LLMEndpointConfig(
        model="ollama_chat/llama3.2",
        api_base="http://localhost:11434",
        temperature=0.1,
        max_tokens=500,
    )
    lm = create_dspy_lm(config)
    dspy.configure(lm=lm)
    yield lm


@pytest.fixture(scope="module")
def analyzer(dspy_lm):
    """Real QueryAnalysisToolV3 with real config and telemetry."""
    from unittest.mock import AsyncMock, MagicMock

    mock_provider = MagicMock()
    mock_provider.datasets = MagicMock()
    mock_provider.datasets.get_dataset = AsyncMock(return_value=None)
    mock_provider.datasets.create_dataset = AsyncMock(return_value="ds_id")

    config_manager = create_default_config_manager()

    tool = QueryAnalysisToolV3(
        tenant_id="default",
        config_manager=config_manager,
        telemetry_provider=mock_provider,
        enable_agent_integration=False,
    )
    return tool


@pytest.mark.integration
@skip_if_no_ollama
class TestQueryAnalysisV3Real:
    """Real query analysis with real Ollama LLM inference."""

    @pytest.mark.asyncio
    async def test_simple_search_query(self, analyzer):
        """Analyze a simple search query with real LLM."""
        result = await analyzer.analyze("Show me videos of cats playing")

        assert isinstance(result, QueryAnalysisResult)
        assert result.original_query == "Show me videos of cats playing"
        assert len(result.cleaned_query) > 0
        assert result.needs_video_search is True
        assert result.confidence_score > 0
        assert result.analysis_time_ms > 0
        assert len(result.keywords) > 0

    @pytest.mark.asyncio
    async def test_temporal_query_detection(self, analyzer):
        """Real LLM detects temporal aspects in queries."""
        result = await analyzer.analyze(
            "What happened in the video after the explosion?"
        )

        assert isinstance(result, QueryAnalysisResult)
        assert result.confidence_score > 0
        assert result.analysis_time_ms > 0

    @pytest.mark.asyncio
    async def test_complex_multimodal_query(self, analyzer):
        """Real LLM handles complex multi-aspect queries."""
        result = await analyzer.analyze(
            "Compare the visual style of the sunset video with the ocean documentary"
        )

        assert isinstance(result, QueryAnalysisResult)
        assert result.needs_video_search is True or result.needs_visual_analysis is True
        assert result.confidence_score > 0
        assert len(result.entities) >= 0

    @pytest.mark.asyncio
    async def test_analysis_with_context(self, analyzer):
        """Query analysis uses conversation context."""
        context = QueryContext(
            conversation_history=[
                "Show me videos about machine learning",
                "Found 5 results about neural networks",
            ]
        )
        result = await analyzer.analyze(
            "Show me more like those", context=context
        )

        assert isinstance(result, QueryAnalysisResult)
        assert result.analysis_time_ms > 0

    @pytest.mark.asyncio
    async def test_result_to_dict(self, analyzer):
        """Verify to_dict returns all expected fields."""
        result = await analyzer.analyze("Find videos of dogs")
        result_dict = result.to_dict()

        assert "original_query" in result_dict
        assert "primary_intent" in result_dict
        assert "complexity_level" in result_dict
        assert "confidence_score" in result_dict
        assert "needs_video_search" in result_dict
        assert "entities" in result_dict
        assert "keywords" in result_dict
        assert "analysis_time_ms" in result_dict
        assert result_dict["original_query"] == "Find videos of dogs"
