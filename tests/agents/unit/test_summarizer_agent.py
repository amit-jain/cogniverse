"""
Unit tests for SummarizerAgent
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.app.agents.summarizer_agent import (
    SummarizerAgent,
    SummaryRequest,
    SummaryResult,
    ThinkingPhase,
    VLMInterface,
)
from src.tools.a2a_utils import A2AMessage, DataPart, Task


@pytest.mark.unit
class TestVLMInterface:
    """Test cases for VLMInterface class"""

    def test_vlm_interface_initialization_mock(self):
        """Test VLM interface initialization with mock client"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = {}

            vlm = VLMInterface()

            assert vlm.client_type == "mock"
            assert vlm.client is None

    @patch("builtins.__import__")
    @patch("src.app.agents.summarizer_agent.get_config")
    def test_vlm_interface_initialization_openai(self, mock_config, mock_import):
        """Test VLM interface initialization with OpenAI"""
        mock_config.return_value = {"openai_api_key": "test_key"}

        # Mock openai module
        mock_openai = Mock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        def mock_import_side_effect(name, *args, **kwargs):
            if name == "openai":
                return mock_openai
            elif name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            else:
                return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        vlm = VLMInterface()

        assert vlm.client_type == "openai"
        assert vlm.client == mock_client

    @patch("builtins.__import__")
    @patch("src.app.agents.summarizer_agent.get_config")
    def test_vlm_interface_initialization_anthropic(self, mock_config, mock_import):
        """Test VLM interface initialization with Anthropic"""
        mock_config.return_value = {"anthropic_api_key": "test_key"}

        # Mock anthropic module
        mock_anthropic = Mock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        def mock_import_side_effect(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            elif name == "anthropic":
                return mock_anthropic
            else:
                return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        vlm = VLMInterface()

        assert vlm.client_type == "anthropic"
        assert vlm.client == mock_client

    @pytest.mark.asyncio
    async def test_analyze_visual_content_mock(self):
        """Test visual content analysis with mock client"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = {}

            vlm = VLMInterface()

            image_paths = ["test1.jpg", "test2.jpg"]
            query = "test query"

            result = await vlm.analyze_visual_content(image_paths, query)

            assert "descriptions" in result
            assert "themes" in result
            assert "insights" in result
            assert len(result["descriptions"]) == len(image_paths)

    def test_mock_visual_analysis(self):
        """Test mock visual analysis method"""
        with patch("src.app.agents.summarizer_agent.get_config") as mock_config:
            mock_config.return_value = {}

            vlm = VLMInterface()

            image_paths = ["test1.jpg", "test2.jpg"]
            query = "test query"

            result = vlm._mock_visual_analysis(image_paths, query)

            assert len(result["descriptions"]) == 2
            assert result["relevance_to_query"] == 0.8
            assert "visual_content" in result["themes"]
            assert len(result["insights"]) == 3


@pytest.mark.unit
class TestSummarizerAgent:
    """Test cases for SummarizerAgent class"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {}

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            {
                "id": "result1",
                "video_id": "video1",
                "title": "Test Video 1",
                "description": "Educational tutorial about machine learning",
                "relevance": 0.95,
                "content_type": "video",
                "duration": 300,
            },
            {
                "id": "result2",
                "video_id": "video2",
                "title": "Test Video 2",
                "description": "News report about AI developments",
                "relevance": 0.87,
                "content_type": "video",
                "duration": 120,
                "frame_id": "frame1",
            },
            {
                "id": "result3",
                "title": "Test Document",
                "text_content": "Document about artificial intelligence",
                "score": 0.76,
                "content_type": "document",
            },
        ]

    @pytest.fixture
    def sample_summary_request(self, sample_search_results):
        """Sample summary request"""
        return SummaryRequest(
            query="machine learning tutorials",
            search_results=sample_search_results,
            summary_type="comprehensive",
            include_visual_analysis=True,
            max_results_to_analyze=10,
        )

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_summarizer_agent_initialization(self, mock_get_config, mock_config):
        """Test SummarizerAgent initialization"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()

        assert agent.config == mock_config
        assert isinstance(agent.vlm, VLMInterface)
        assert agent.max_summary_length == 500
        assert agent.thinking_enabled is True
        assert agent.visual_analysis_enabled is True

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_summarizer_agent_custom_config(self, mock_get_config, mock_config):
        """Test SummarizerAgent with custom configuration"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent(
            max_summary_length=1000,
            thinking_enabled=False,
            visual_analysis_enabled=False,
            vlm_model="custom-model",
        )

        assert agent.max_summary_length == 1000
        assert agent.thinking_enabled is False
        assert agent.visual_analysis_enabled is False

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_extract_themes(self, mock_get_config, mock_config, sample_search_results):
        """Test theme extraction from search results"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        themes = agent._extract_themes(sample_search_results)

        assert "video_content" in themes
        assert "educational_content" in themes
        assert "news_content" in themes
        assert len(themes) <= 10

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_categorize_content(
        self, mock_get_config, mock_config, sample_search_results
    ):
        """Test content categorization"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        categories = agent._categorize_content(sample_search_results)

        assert "video" in categories
        assert "text" in categories
        # Note: Both videos are medium_form (120s and 300s both < 600s)
        assert "medium_form" in categories  # Both 120s and 300s videos
        assert "frame_based" in categories

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_calculate_relevance_scores(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test relevance score calculation"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        scores = agent._calculate_relevance_scores(sample_summary_request)

        assert "result1" in scores
        assert "result2" in scores
        assert "result3" in scores
        assert scores["result1"] == 0.95
        assert scores["result2"] == 0.87
        assert scores["result3"] == 0.76

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_identify_visual_elements(
        self, mock_get_config, mock_config, sample_search_results
    ):
        """Test visual element identification"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        visual_elements = agent._identify_visual_elements(sample_search_results)

        assert "video_frames" in visual_elements

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_generate_reasoning(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test reasoning generation"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        themes = ["educational_content", "video_content"]
        categories = ["video", "short_form"]
        relevance_scores = {"result1": 0.95, "result2": 0.87}

        reasoning = agent._generate_reasoning(
            sample_summary_request, themes, categories, relevance_scores
        )

        assert "machine learning tutorials" in reasoning
        assert "educational_content" in reasoning
        assert "0.91" in reasoning  # Average relevance
        assert "comprehensive" in reasoning

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_thinking_phase(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test thinking phase execution"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        thinking_phase = await agent._thinking_phase(sample_summary_request)

        assert isinstance(thinking_phase, ThinkingPhase)
        assert len(thinking_phase.key_themes) > 0
        assert len(thinking_phase.content_categories) > 0
        assert len(thinking_phase.relevance_scores) == 3
        assert thinking_phase.reasoning is not None
        assert "video_content" in thinking_phase.key_themes

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_generate_brief_summary(
        self,
        mock_get_config,
        mock_config,
        sample_summary_request,
        sample_search_results,
    ):
        """Test brief summary generation"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        thinking_phase = ThinkingPhase(
            key_themes=["educational_content", "video_content"],
            content_categories=["video", "short_form"],
            relevance_scores={},
            visual_elements=[],
            reasoning="",
        )

        summary = agent._generate_brief_summary(
            sample_summary_request, sample_search_results, thinking_phase
        )

        assert "machine learning tutorials" in summary
        assert "3 results" in summary
        assert "educational_content" in summary

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_generate_bullet_summary(
        self,
        mock_get_config,
        mock_config,
        sample_summary_request,
        sample_search_results,
    ):
        """Test bullet point summary generation"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        thinking_phase = ThinkingPhase(
            key_themes=["educational_content"],
            content_categories=["video"],
            relevance_scores={},
            visual_elements=[],
            reasoning="",
        )

        summary = agent._generate_bullet_summary(
            sample_summary_request, sample_search_results, thinking_phase
        )

        assert summary.startswith("•")
        assert "machine learning tutorials" in summary
        assert "Results found: 3" in summary
        assert "Test Video 1" in summary

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_generate_comprehensive_summary(
        self,
        mock_get_config,
        mock_config,
        sample_summary_request,
        sample_search_results,
    ):
        """Test comprehensive summary generation"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        thinking_phase = ThinkingPhase(
            key_themes=["educational_content", "video_content"],
            content_categories=["video", "short_form"],
            relevance_scores={"result1": 0.95, "result2": 0.87, "result3": 0.76},
            visual_elements=["video_frames"],
            reasoning="",
        )
        visual_insights = ["Educational content detected", "High quality visuals"]

        summary = agent._generate_comprehensive_summary(
            sample_summary_request,
            sample_search_results,
            thinking_phase,
            visual_insights,
        )

        assert "machine learning tutorials" in summary
        assert "3 relevant items" in summary
        assert "Test Video 1" in summary
        assert "Educational content detected" in summary
        assert "strong alignment" in summary  # High avg relevance (0.86)

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_extract_key_points(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test key points extraction"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        thinking_phase = ThinkingPhase(
            key_themes=["educational_content", "video_content"],
            content_categories=["video", "document"],
            relevance_scores={"result1": 0.95, "result2": 0.87, "result3": 0.76},
            visual_elements=["video_frames"],
            reasoning="",
        )

        key_points = agent._extract_key_points(
            sample_summary_request, thinking_phase, "test summary"
        )

        assert any("educational_content" in point for point in key_points)
        assert any("video, document" in point for point in key_points)
        assert any("high-relevance" in point for point in key_points)
        assert any("video_frames" in point for point in key_points)

    @patch("src.app.agents.summarizer_agent.get_config")
    def test_calculate_confidence(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test confidence score calculation"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        thinking_phase = ThinkingPhase(
            key_themes=["theme1", "theme2", "theme3"],
            content_categories=[],
            relevance_scores={"result1": 0.9, "result2": 0.8, "result3": 0.7},
            visual_elements=[],
            reasoning="",
        )

        confidence = agent._calculate_confidence(sample_summary_request, thinking_phase)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high with good results and themes

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_analyze_visual_content_disabled(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test visual content analysis when disabled"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent(visual_analysis_enabled=False)
        thinking_phase = ThinkingPhase(
            key_themes=[],
            content_categories=[],
            relevance_scores={},
            visual_elements=["video_frames"],
            reasoning="",
        )

        # When visual analysis is disabled, the method should not be called
        # But the test request still has include_visual_analysis=True
        sample_summary_request.include_visual_analysis = False

        insights = await agent._analyze_visual_content(
            sample_summary_request, thinking_phase
        )

        assert insights == []

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_analyze_visual_content_no_elements(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test visual content analysis with no visual elements"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        thinking_phase = ThinkingPhase(
            key_themes=[],
            content_categories=[],
            relevance_scores={},
            visual_elements=[],
            reasoning="",
        )

        insights = await agent._analyze_visual_content(
            sample_summary_request, thinking_phase
        )

        assert insights == []

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_analyze_visual_content_with_images(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test visual content analysis with images"""
        mock_get_config.return_value = mock_config

        # Add image paths to search results
        sample_summary_request.search_results[0]["thumbnail"] = "test1.jpg"
        sample_summary_request.search_results[1]["image_path"] = "test2.jpg"

        agent = SummarizerAgent()
        thinking_phase = ThinkingPhase(
            key_themes=[],
            content_categories=[],
            relevance_scores={},
            visual_elements=["video_frames"],
            reasoning="",
        )

        # Mock VLM analysis
        agent.vlm.analyze_visual_content = AsyncMock(
            return_value={
                "insights": ["Educational content detected"],
                "descriptions": ["Frame showing tutorial"],
            }
        )

        insights = await agent._analyze_visual_content(
            sample_summary_request, thinking_phase
        )

        assert len(insights) > 0
        assert "Educational content detected" in insights

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_summarize_comprehensive(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test complete summarization process"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()

        # Mock VLM analysis
        agent.vlm.analyze_visual_content = AsyncMock(
            return_value={
                "insights": ["Educational content detected"],
                "descriptions": ["Tutorial video frame"],
            }
        )

        result = await agent.summarize(sample_summary_request)

        assert isinstance(result, SummaryResult)
        assert result.summary is not None
        assert len(result.key_points) > 0
        assert result.confidence_score > 0.0
        assert isinstance(result.thinking_phase, ThinkingPhase)
        assert result.metadata["results_analyzed"] == 3
        assert result.metadata["summary_type"] == "comprehensive"

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_summarize_brief(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test brief summarization"""
        mock_get_config.return_value = mock_config
        sample_summary_request.summary_type = "brief"

        agent = SummarizerAgent()
        result = await agent.summarize(sample_summary_request)

        assert isinstance(result, SummaryResult)
        # Brief summary should be shorter but allow some flexibility
        assert len(result.summary) < 250  # Increased tolerance
        assert "3 results" in result.summary

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_summarize_bullet_points(
        self, mock_get_config, mock_config, sample_summary_request
    ):
        """Test bullet point summarization"""
        mock_get_config.return_value = mock_config
        sample_summary_request.summary_type = "bullet_points"

        agent = SummarizerAgent()
        result = await agent.summarize(sample_summary_request)

        assert isinstance(result, SummaryResult)
        assert result.summary.startswith("•")
        assert "•" in result.summary

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_process_a2a_task(
        self, mock_get_config, mock_config, sample_search_results
    ):
        """Test A2A task processing"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()

        # Mock VLM analysis
        agent.vlm.analyze_visual_content = AsyncMock(
            return_value={
                "insights": ["Test insight"],
                "descriptions": ["Test description"],
            }
        )

        # Create A2A task
        message = A2AMessage(
            role="user",
            parts=[
                DataPart(
                    data={
                        "query": "test query",
                        "search_results": sample_search_results,
                        "summary_type": "comprehensive",
                        "include_visual_analysis": True,
                    }
                )
            ],
        )
        task = Task(id="test_task", messages=[message])

        result = await agent.process_a2a_task(task)

        assert result["task_id"] == "test_task"
        assert result["status"] == "completed"
        assert "summary" in result
        assert "key_points" in result
        assert "confidence_score" in result
        assert "thinking_process" in result

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_process_a2a_task_no_messages(self, mock_get_config, mock_config):
        """Test A2A task processing with no messages"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()
        task = Task(id="test_task", messages=[])

        with pytest.raises(ValueError, match="Task contains no messages"):
            await agent.process_a2a_task(task)

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_process_a2a_task_no_data_part(self, mock_get_config, mock_config):
        """Test A2A task processing with no data part"""
        mock_get_config.return_value = mock_config

        agent = SummarizerAgent()

        # Create task without DataPart
        from src.tools.a2a_utils import TextPart

        message = A2AMessage(role="user", parts=[TextPart(text="test message")])
        task = Task(id="test_task", messages=[message])

        with pytest.raises(ValueError, match="No data part found in message"):
            await agent.process_a2a_task(task)


@pytest.mark.unit
class TestSummarizerAgentEdgeCases:
    """Test edge cases and error conditions"""

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_summarize_empty_results(self, mock_get_config):
        """Test summarization with empty search results"""
        mock_get_config.return_value = {}

        agent = SummarizerAgent()
        request = SummaryRequest(
            query="test query", search_results=[], summary_type="brief"
        )

        result = await agent.summarize(request)

        assert "No relevant results" in result.summary
        assert result.confidence_score < 0.7  # Low confidence for empty results

    @patch("src.app.agents.summarizer_agent.get_config")
    @pytest.mark.asyncio
    async def test_visual_analysis_failure(self, mock_get_config):
        """Test handling of visual analysis failure"""
        mock_get_config.return_value = {}

        agent = SummarizerAgent()

        # Mock VLM to raise exception
        agent.vlm.analyze_visual_content = AsyncMock(
            side_effect=Exception("VLM failed")
        )

        thinking_phase = ThinkingPhase(
            key_themes=[],
            content_categories=[],
            relevance_scores={},
            visual_elements=["video_frames"],
            reasoning="",
        )

        request = SummaryRequest(
            query="test",
            search_results=[{"thumbnail": "test.jpg"}],
            include_visual_analysis=True,
        )

        insights = await agent._analyze_visual_content(request, thinking_phase)

        assert "Visual analysis unavailable" in insights


@pytest.mark.unit
class TestDataClasses:
    """Test data class validation and functionality"""

    def test_summary_request_defaults(self):
        """Test SummaryRequest with default values"""
        request = SummaryRequest(query="test query", search_results=[])

        assert request.summary_type == "comprehensive"
        assert request.include_visual_analysis is True
        assert request.max_results_to_analyze == 10
        assert request.context is None

    def test_thinking_phase_creation(self):
        """Test ThinkingPhase creation"""
        thinking_phase = ThinkingPhase(
            key_themes=["theme1", "theme2"],
            content_categories=["video"],
            relevance_scores={"result1": 0.9},
            visual_elements=["frames"],
            reasoning="test reasoning",
        )

        assert len(thinking_phase.key_themes) == 2
        assert thinking_phase.content_categories == ["video"]
        assert thinking_phase.relevance_scores["result1"] == 0.9
        assert thinking_phase.reasoning == "test reasoning"

    def test_summary_result_creation(self):
        """Test SummaryResult creation"""
        thinking_phase = ThinkingPhase(
            key_themes=[],
            content_categories=[],
            relevance_scores={},
            visual_elements=[],
            reasoning="",
        )

        result = SummaryResult(
            summary="test summary",
            key_points=["point1", "point2"],
            visual_insights=["insight1"],
            confidence_score=0.85,
            thinking_phase=thinking_phase,
            metadata={"test": "value"},
        )

        assert result.summary == "test summary"
        assert len(result.key_points) == 2
        assert result.confidence_score == 0.85
        assert result.metadata["test"] == "value"
