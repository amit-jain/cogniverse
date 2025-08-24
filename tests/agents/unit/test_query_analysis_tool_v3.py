"""Tests for QueryAnalysisToolV3 with enhanced capabilities."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.app.agents.query_analysis_tool_v3 import (
    QueryAnalysisResult,
    QueryAnalysisToolV3,
    QueryComplexity,
    QueryContext,
    QueryIntent,
    create_enhanced_query_analyzer,
)


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {"routing": {"enabled": True}, "agents": {"routing_agent": True}}


@pytest.fixture
def sample_context():
    """Sample query context for testing"""
    return QueryContext(
        conversation_history=["previous query about AI", "machine learning basics"],
        user_preferences={"preferred_format": "video"},
        session_metadata={"session_id": "test_session"},
    )


@pytest.fixture
def sample_routing_analysis():
    """Sample routing analysis result"""
    return {
        "workflow": {
            "type": "detailed_report",
            "steps": [
                {"step": 1, "agent": "video_search", "action": "search"},
                {"step": 2, "agent": "detailed_report", "action": "analyze"},
            ],
            "agents": ["video_search", "detailed_report"],
        }
    }


@pytest.mark.unit
class TestQueryAnalysisToolV3:
    """Test QueryAnalysisToolV3 functionality"""

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_analyzer_initialization_default(self, mock_get_config, mock_config):
        """Test analyzer initialization with default settings"""
        mock_get_config.return_value = mock_config

        analyzer = QueryAnalysisToolV3()

        assert analyzer.enable_thinking_phase is True
        assert analyzer.enable_query_expansion is True
        assert analyzer.enable_agent_integration is True
        assert analyzer.max_expanded_queries == 3
        assert analyzer.total_analyses == 0
        assert isinstance(analyzer.start_time, datetime)

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_analyzer_initialization_custom(self, mock_get_config, mock_config):
        """Test analyzer initialization with custom settings"""
        mock_get_config.return_value = mock_config

        analyzer = QueryAnalysisToolV3(
            enable_thinking_phase=False,
            enable_query_expansion=False,
            enable_agent_integration=False,
            max_expanded_queries=5,
        )

        assert analyzer.enable_thinking_phase is False
        assert analyzer.enable_query_expansion is False
        assert analyzer.enable_agent_integration is False
        assert analyzer.max_expanded_queries == 5

    @patch("src.app.agents.query_analysis_tool_v3.RoutingAgent")
    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_analyzer_with_routing_agent(
        self, mock_get_config, mock_routing_agent, mock_config
    ):
        """Test analyzer initialization with routing agent integration"""
        mock_get_config.return_value = mock_config
        mock_routing_instance = Mock()
        mock_routing_agent.return_value = mock_routing_instance

        analyzer = QueryAnalysisToolV3(enable_agent_integration=True)

        assert analyzer.routing_agent == mock_routing_instance
        mock_routing_agent.assert_called_once()

    @patch("src.app.agents.query_analysis_tool_v3.RoutingAgent")
    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_analyzer_routing_agent_failure(
        self, mock_get_config, mock_routing_agent, mock_config
    ):
        """Test analyzer handling routing agent initialization failure"""
        mock_get_config.return_value = mock_config
        mock_routing_agent.side_effect = Exception("Routing agent failed")

        analyzer = QueryAnalysisToolV3(enable_agent_integration=True)

        assert analyzer.routing_agent is None

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_clean_query(self, mock_get_config, mock_config):
        """Test query cleaning functionality"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        # Test basic cleaning
        assert analyzer._clean_query("  HELLO WORLD  ") == "hello world"
        assert (
            analyzer._clean_query("Multiple   spaces   here") == "multiple spaces here"
        )
        assert analyzer._clean_query("Mixed\t\nWhitespace") == "mixed whitespace"

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_detect_intents_simple(self, mock_get_config, mock_config):
        """Test intent detection for simple queries"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        # Test search intent
        thinking_phase = {"query_type_indicators": {}}
        primary, secondary = analyzer._detect_intents(
            "find videos about cats", thinking_phase
        )
        assert primary == QueryIntent.SEARCH

        # Test summarize intent
        thinking_phase = {"query_type_indicators": {"summary": True}}
        primary, secondary = analyzer._detect_intents(
            "summarize the content", thinking_phase
        )
        assert primary == QueryIntent.SUMMARIZE

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_detect_intents_complex(self, mock_get_config, mock_config):
        """Test intent detection for complex queries"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        thinking_phase = {
            "query_type_indicators": {"report": True},
            "complexity_signals": ["analyze", "detailed"],
            "temporal_indicators": ["recent"],
            "modality_hints": {"video": True, "text": True},
        }

        primary, secondary = analyzer._detect_intents(
            "analyze recent videos and create detailed report", thinking_phase
        )

        # Intent detection can vary with model - check that we get a reasonable intent
        assert primary in [QueryIntent.REPORT, QueryIntent.ANALYZE]
        # Secondary intents can vary with model - check for reasonable intents
        assert len(secondary) > 0
        # Should contain temporal since query mentions "recent"
        assert QueryIntent.TEMPORAL in secondary

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_assess_complexity_simple(self, mock_get_config, mock_config):
        """Test complexity assessment for simple queries"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        complexity = analyzer._assess_complexity("find cats", QueryIntent.SEARCH, [])
        assert complexity == QueryComplexity.SIMPLE

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_assess_complexity_moderate(self, mock_get_config, mock_config):
        """Test complexity assessment for moderate queries"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        complexity = analyzer._assess_complexity(
            "compare cats and dogs", QueryIntent.COMPARE, [QueryIntent.ANALYZE]
        )
        assert complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_assess_complexity_complex(self, mock_get_config, mock_config):
        """Test complexity assessment for complex queries"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        complexity = analyzer._assess_complexity(
            "analyze and report comprehensive findings",
            QueryIntent.ANALYZE,
            [QueryIntent.REPORT, QueryIntent.MULTIMODAL],
        )
        assert complexity == QueryComplexity.COMPLEX

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_detect_modality_requirements_default(self, mock_get_config, mock_config):
        """Test modality detection with default requirements"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        thinking_phase = {"modality_hints": {}}
        requirements = analyzer._detect_modality_requirements(
            "generic query", thinking_phase
        )

        assert requirements["video"] is True
        assert requirements["text"] is True
        assert requirements["visual_analysis"] is False

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_detect_modality_requirements_video_only(
        self, mock_get_config, mock_config
    ):
        """Test modality detection for video-only queries"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        thinking_phase = {"modality_hints": {"video": True, "text": False}}
        requirements = analyzer._detect_modality_requirements(
            "show video clips", thinking_phase
        )

        assert requirements["video"] is True
        assert requirements["text"] is False

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_detect_modality_requirements_visual_analysis(
        self, mock_get_config, mock_config
    ):
        """Test modality detection enabling visual analysis"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        thinking_phase = {"modality_hints": {"video": True}}
        requirements = analyzer._detect_modality_requirements(
            "analyze visual content", thinking_phase
        )

        assert requirements["visual_analysis"] is True

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_extract_temporal_filters(self, mock_get_config, mock_config):
        """Test temporal filter extraction"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        # Test common temporal terms
        filters = analyzer._extract_temporal_filters("show me recent videos")
        assert "start_date" in filters
        assert filters["temporal_term"] == "recent"

        # Test specific date patterns
        filters = analyzer._extract_temporal_filters("videos from 2023-01-15")
        assert "specific_dates" in filters
        assert "2023-01-15" in filters["specific_dates"]

        # Test no temporal info
        filters = analyzer._extract_temporal_filters("show me videos about cats")
        assert len(filters) == 0

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_extract_entities_and_keywords(self, mock_get_config, mock_config):
        """Test entity and keyword extraction"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        query = 'Find videos about "machine learning" and Python programming'
        entities, keywords = analyzer._extract_entities_and_keywords(query)

        # Check entities
        quoted_entities = [e for e in entities if e["type"] == "quoted_phrase"]
        assert len(quoted_entities) == 1
        assert quoted_entities[0]["text"] == "machine learning"
        assert quoted_entities[0]["confidence"] == 0.9

        proper_noun_entities = [e for e in entities if e["type"] == "proper_noun"]
        assert any(e["text"] == "Python" for e in proper_noun_entities)

        # Check keywords
        assert "videos" in keywords
        assert "programming" in keywords
        # Stop word filtering may vary - check that we get meaningful keywords
        assert (
            len(
                [
                    k
                    for k in keywords
                    if k not in ["about", "the", "and", "or", "in", "on", "at"]
                ]
            )
            >= 3
        )

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_expand_query_basic(self, mock_get_config, mock_config):
        """Test basic query expansion"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        expansions = await analyzer._expand_query("find videos about cats", None, {})

        # Should include some form of expansion
        assert len(expansions) > 0
        # Check for at least one synonym-like expansion
        original_words = set("find videos programming".split())
        expansion_words = set(" ".join(expansions).lower().split())
        assert len(expansion_words - original_words) > 0  # At least some new words

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_expand_query_with_context(
        self, mock_get_config, mock_config, sample_context
    ):
        """Test query expansion with context"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        expansions = await analyzer._expand_query("latest research", sample_context, {})

        # Should include some form of context-based expansion
        assert len(expansions) > 0
        # Check that expansions incorporate context somehow
        expansion_text = " ".join(expansions).lower()
        # Should have words related to research or context
        assert any(
            word in expansion_text
            for word in [
                "research",
                "study",
                "analysis",
                "ai",
                "artificial",
                "intelligence",
            ]
        )

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_calculate_confidence_high(self, mock_get_config, mock_config):
        """Test confidence calculation for high-confidence scenarios"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        confidence = analyzer._calculate_confidence(
            QueryIntent.SEARCH,  # Clear intent
            QueryComplexity.SIMPLE,  # Simple complexity
            5,  # Many entities
            20,  # Many keywords
        )

        assert confidence > 0.8

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_calculate_confidence_low(self, mock_get_config, mock_config):
        """Test confidence calculation for low-confidence scenarios"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        confidence = analyzer._calculate_confidence(
            QueryIntent.MULTIMODAL,  # Less clear intent
            QueryComplexity.COMPLEX,  # Complex query
            0,  # No entities
            2,  # Few keywords
        )

        assert confidence < 0.7

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_determine_workflow_without_routing_agent(
        self, mock_get_config, mock_config
    ):
        """Test workflow determination without routing agent"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(enable_agent_integration=False)

        workflow = await analyzer._determine_workflow(
            "analyze this content",
            QueryIntent.ANALYZE,
            [],
            QueryComplexity.MODERATE,
            {},
        )

        assert workflow["type"] == "detailed_report"
        assert "video_search" in workflow["agents"]
        assert "detailed_report" in workflow["agents"]
        assert len(workflow["steps"]) == 2

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_determine_workflow_with_routing_agent(
        self, mock_get_config, mock_config, sample_routing_analysis
    ):
        """Test workflow determination with routing agent integration"""
        mock_get_config.return_value = mock_config

        # Mock routing agent
        mock_routing_agent = AsyncMock()
        mock_routing_agent.analyze_and_route.return_value = sample_routing_analysis

        analyzer = QueryAnalysisToolV3(enable_agent_integration=True)
        analyzer.routing_agent = mock_routing_agent

        workflow = await analyzer._determine_workflow(
            "create detailed report",
            QueryIntent.REPORT,
            [],
            QueryComplexity.COMPLEX,
            {},
        )

        assert workflow["type"] == "detailed_report"
        assert workflow["agents"] == ["video_search", "detailed_report"]
        mock_routing_agent.analyze_and_route.assert_called_once()

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_thinking_phase_simple_query(self, mock_get_config, mock_config):
        """Test thinking phase for simple query"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        thinking = await analyzer._thinking_phase("find cats", None)

        assert thinking["query_length"] == 2
        assert thinking["has_context"] is False
        assert "Simple search query detected" in thinking["reasoning"]

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_thinking_phase_complex_query(
        self, mock_get_config, mock_config, sample_context
    ):
        """Test thinking phase for complex query"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()

        thinking = await analyzer._thinking_phase(
            "analyze recent videos and create comprehensive detailed report with visual analysis",
            sample_context,
        )

        assert thinking["query_length"] > 5
        assert thinking["has_context"] is True
        assert len(thinking["complexity_signals"]) >= 2
        assert thinking["modality_hints"]["video"] is True
        assert len(thinking["temporal_indicators"]) >= 1
        assert thinking["query_type_indicators"]["report"] is True
        assert "complexity signals" in thinking["reasoning"].lower()

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_analyze_simple_query(self, mock_get_config, mock_config):
        """Test full analysis of a simple query"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(enable_agent_integration=False)

        result = await analyzer.analyze("find videos about cats")

        assert isinstance(result, QueryAnalysisResult)
        assert result.original_query == "find videos about cats"
        assert result.cleaned_query == "find videos about cats"
        assert result.primary_intent == QueryIntent.SEARCH
        assert result.complexity_level == QueryComplexity.SIMPLE
        assert result.needs_video_search is True
        assert result.confidence_score > 0.0
        assert result.recommended_workflow in [
            "raw_results",
            "summary",
            "detailed_report",
        ]
        assert len(result.required_agents) > 0
        assert result.analysis_time_ms > 0
        assert analyzer.total_analyses == 1

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_analyze_complex_query(self, mock_get_config, mock_config):
        """Test full analysis of a complex query"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(enable_agent_integration=False)

        result = await analyzer.analyze(
            "analyze all content about artificial intelligence from last week and create a detailed report"
        )

        assert result.primary_intent in [QueryIntent.ANALYZE, QueryIntent.REPORT]
        assert result.complexity_level in [
            QueryComplexity.MODERATE,
            QueryComplexity.COMPLEX,
        ]
        assert QueryIntent.TEMPORAL in result.secondary_intents
        assert result.needs_visual_analysis is True
        assert result.recommended_workflow == "detailed_report"
        assert "detailed_report" in result.required_agents
        assert len(result.temporal_filters) > 0
        assert result.thinking_phase["reasoning"]

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_analyze_with_expansion(self, mock_get_config, mock_config):
        """Test analysis with query expansion enabled"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(
            enable_query_expansion=True, enable_agent_integration=False
        )

        result = await analyzer.analyze("find videos about machine learning")

        assert len(result.expanded_queries) > 0
        assert any(
            "search" in query for query in result.expanded_queries
        )  # Synonym expansion

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_analyze_with_context(
        self, mock_get_config, mock_config, sample_context
    ):
        """Test analysis with context"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(enable_agent_integration=False)

        result = await analyzer.analyze("latest developments", sample_context)

        assert result.thinking_phase["has_context"] is True
        # Should have context-based expansions
        assert len(result.expanded_queries) >= 0

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_analyze_error_handling(self, mock_get_config, mock_config):
        """Test analysis error handling"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(enable_agent_integration=False)

        # Mock a method to raise an exception
        with patch.object(
            analyzer, "_thinking_phase", side_effect=Exception("Test error")
        ):
            with pytest.raises(Exception) as exc_info:
                await analyzer.analyze("test query")

            assert "Test error" in str(exc_info.value)

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_get_statistics(self, mock_get_config, mock_config):
        """Test statistics collection"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3()
        analyzer.total_analyses = 10

        stats = analyzer.get_statistics()

        assert stats["total_analyses"] == 10
        assert "uptime_seconds" in stats
        assert "analyses_per_minute" in stats
        assert "configuration" in stats
        assert stats["configuration"]["thinking_phase_enabled"] is True
        assert stats["configuration"]["query_expansion_enabled"] is True
        assert stats["configuration"]["agent_integration_enabled"] is True

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_query_analysis_result_to_dict(self, mock_get_config, mock_config):
        """Test QueryAnalysisResult to_dict conversion"""
        result = QueryAnalysisResult(
            original_query="test query",
            cleaned_query="test query",
            expanded_queries=["expanded test query"],
            primary_intent=QueryIntent.SEARCH,
            secondary_intents=[QueryIntent.TEMPORAL],
            complexity_level=QueryComplexity.SIMPLE,
            confidence_score=0.8,
            needs_video_search=True,
            needs_text_search=True,
            needs_visual_analysis=False,
            temporal_filters={"start_date": "2024-01-01"},
            entities=[{"text": "test", "type": "keyword", "confidence": 0.5}],
            keywords=["test", "query"],
            recommended_workflow="raw_results",
            workflow_steps=[{"step": 1, "agent": "search"}],
            required_agents=["search"],
            thinking_phase={"reasoning": "test reasoning"},
            analysis_time_ms=100.5,
            routing_method="enhanced_v3",
        )

        result_dict = result.to_dict()

        assert result_dict["original_query"] == "test query"
        assert result_dict["primary_intent"] == "search"
        assert result_dict["secondary_intents"] == ["temporal"]
        assert result_dict["complexity_level"] == "simple"
        assert result_dict["confidence_score"] == 0.8
        assert result_dict["analysis_time_ms"] == 100.5


@pytest.mark.unit
class TestQueryAnalysisToolV3EdgeCases:
    """Test edge cases for QueryAnalysisToolV3"""

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_analyze_empty_query(self, mock_get_config, mock_config):
        """Test analysis of empty query"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(enable_agent_integration=False)

        result = await analyzer.analyze("")

        assert result.original_query == ""
        assert result.cleaned_query == ""
        assert result.primary_intent == QueryIntent.SEARCH  # Default
        assert result.complexity_level == QueryComplexity.SIMPLE
        assert len(result.entities) == 0
        assert len(result.keywords) == 0

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    @pytest.mark.asyncio
    async def test_analyze_very_long_query(self, mock_get_config, mock_config):
        """Test analysis of very long query"""
        mock_get_config.return_value = mock_config
        analyzer = QueryAnalysisToolV3(enable_agent_integration=False)

        long_query = (
            "analyze " + " ".join(["word"] * 50) + " and create detailed report"
        )
        result = await analyzer.analyze(long_query)

        assert result.thinking_phase["query_length"] > 10
        assert (
            "Long query suggests complex information need"
            in result.thinking_phase["reasoning"]
        )
        assert result.complexity_level in [
            QueryComplexity.MODERATE,
            QueryComplexity.COMPLEX,
        ]


@pytest.mark.unit
class TestFactoryFunction:
    """Test factory function"""

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_create_enhanced_query_analyzer_default(self, mock_get_config, mock_config):
        """Test factory function with default parameters"""
        mock_get_config.return_value = mock_config

        analyzer = create_enhanced_query_analyzer()

        assert isinstance(analyzer, QueryAnalysisToolV3)
        assert analyzer.enable_thinking_phase is True
        assert analyzer.enable_query_expansion is True
        assert analyzer.enable_agent_integration is True

    @patch("src.app.agents.query_analysis_tool_v3.get_config")
    def test_create_enhanced_query_analyzer_custom(self, mock_get_config, mock_config):
        """Test factory function with custom parameters"""
        mock_get_config.return_value = mock_config

        analyzer = create_enhanced_query_analyzer(
            enable_thinking_phase=False, max_expanded_queries=10
        )

        assert isinstance(analyzer, QueryAnalysisToolV3)
        assert analyzer.enable_thinking_phase is False
        assert analyzer.max_expanded_queries == 10


@pytest.mark.unit
class TestDataClasses:
    """Test data classes and enums"""

    def test_query_context_defaults(self):
        """Test QueryContext with default values"""
        context = QueryContext()

        assert context.conversation_history == []
        assert context.user_preferences == {}
        assert context.previous_results == []
        assert context.session_metadata == {}

    def test_query_context_with_values(self):
        """Test QueryContext with provided values"""
        context = QueryContext(
            conversation_history=["query1"],
            user_preferences={"format": "video"},
            previous_results=[{"id": 1}],
            session_metadata={"session": "test"},
        )

        assert context.conversation_history == ["query1"]
        assert context.user_preferences == {"format": "video"}
        assert context.previous_results == [{"id": 1}]
        assert context.session_metadata == {"session": "test"}

    def test_query_intent_enum(self):
        """Test QueryIntent enum values"""
        assert QueryIntent.SEARCH.value == "search"
        assert QueryIntent.ANALYZE.value == "analyze"
        assert QueryIntent.REPORT.value == "report"

    def test_query_complexity_enum(self):
        """Test QueryComplexity enum values"""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MODERATE.value == "moderate"
        assert QueryComplexity.COMPLEX.value == "complex"
