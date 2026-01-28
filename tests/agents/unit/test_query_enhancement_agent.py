"""Unit tests for QueryEnhancementAgent"""

from unittest.mock import Mock, patch

import dspy
import pytest

from cogniverse_agents.query_enhancement_agent import (
    QueryEnhancementAgent,
    QueryEnhancementDeps,
    QueryEnhancementInput,
    QueryEnhancementModule,
    QueryEnhancementResult,
)


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model"""
    lm = Mock()
    lm.return_value = dspy.Prediction(
        enhanced_query="machine learning tutorials and comprehensive guides",
        expansion_terms="deep learning, neural networks, AI fundamentals",
        synonyms="ML, artificial intelligence, AI",
        context="beginner, introduction, basics",
        confidence="0.85",
        reasoning="Expanded ML acronym, added related AI terms and beginner context",
    )
    return lm


@pytest.fixture
def query_agent():
    """Create QueryEnhancementAgent for testing"""
    with patch("dspy.ChainOfThought"):
        deps = QueryEnhancementDeps(tenant_id="test_tenant")
        agent = QueryEnhancementAgent(deps=deps, port=8012)
        return agent


class TestQueryEnhancementModule:
    """Test DSPy module for query enhancement"""

    def test_module_initialization(self):
        """Test QueryEnhancementModule initializes correctly"""
        with patch("dspy.ChainOfThought") as mock_cot:
            module = QueryEnhancementModule()
            assert module.enhancer is not None
            mock_cot.assert_called_once()

    def test_forward_success(self, mock_dspy_lm):
        """Test successful query enhancement"""
        module = QueryEnhancementModule()
        module.enhancer = mock_dspy_lm

        result = module.forward(query="ML tutorials")

        assert (
            result.enhanced_query
            == "machine learning tutorials and comprehensive guides"
        )
        assert "deep learning" in result.expansion_terms
        assert result.confidence == "0.85"

    def test_forward_fallback(self):
        """Test fallback when DSPy fails"""
        module = QueryEnhancementModule()
        module.enhancer = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(query="Show me ML videos")

        # Fallback should add basic expansions
        assert result.enhanced_query == "Show me ML videos"  # Keeps original
        assert "machine learning" in result.expansion_terms.lower()
        assert result.confidence == "0.5"

    def test_fallback_ai_expansion(self):
        """Test fallback expands AI acronym"""
        module = QueryEnhancementModule()
        module.enhancer = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(query="AI tutorials")

        assert "artificial intelligence" in result.expansion_terms.lower()

    def test_fallback_video_context(self):
        """Test fallback adds video-related context"""
        module = QueryEnhancementModule()
        module.enhancer = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(query="Show videos about Python")

        expansions_lower = result.expansion_terms.lower()
        assert any(
            term in expansions_lower for term in ["tutorial", "guide", "demonstration"]
        )


class TestQueryEnhancementAgent:
    """Test QueryEnhancementAgent core functionality"""

    def test_agent_initialization(self, query_agent):
        """Test agent initializes with correct configuration"""
        assert query_agent.agent_name == "query_enhancement_agent"
        assert query_agent.tenant_id == "test_tenant"
        assert "query_enhancement" in query_agent.capabilities

    @pytest.mark.asyncio
    async def test_process_with_query(self, query_agent):
        """Test processing query for enhancement"""
        query_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                enhanced_query="machine learning tutorials and guides",
                expansion_terms="deep learning, neural networks, AI",
                synonyms="ML, artificial intelligence",
                context="beginner, fundamentals",
                confidence="0.85",
                reasoning="Expanded ML and added related terms",
            )
        )

        result = await query_agent._process_impl(
            QueryEnhancementInput(query="ML tutorials")
        )

        assert isinstance(result, QueryEnhancementResult)
        assert result.original_query == "ML tutorials"
        assert result.enhanced_query == "machine learning tutorials and guides"
        assert len(result.expansion_terms) == 3
        assert "deep learning" in result.expansion_terms
        assert len(result.synonyms) == 2
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_process_empty_query(self, query_agent):
        """Test processing empty query"""
        result = await query_agent._process_impl(QueryEnhancementInput(query=""))

        assert result.original_query == ""
        assert result.enhanced_query == ""
        assert len(result.expansion_terms) == 0
        assert result.confidence == 0.0
        assert "no enhancement" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_process_missing_query(self, query_agent):
        """Test processing with missing query field (empty string equivalent)"""
        result = await query_agent._process_impl(QueryEnhancementInput(query=""))

        assert result.original_query == ""
        assert result.enhanced_query == ""

    @pytest.mark.asyncio
    async def test_process_invalid_confidence(self, query_agent):
        """Test processing with invalid confidence value"""
        query_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                enhanced_query="test query",
                expansion_terms="",
                synonyms="",
                context="",
                confidence="invalid",  # Invalid confidence
                reasoning="Test",
            )
        )

        result = await query_agent._process_impl(QueryEnhancementInput(query="test"))

        assert result.confidence == 0.7  # Default fallback

    @pytest.mark.asyncio
    async def test_process_empty_expansions(self, query_agent):
        """Test processing with no expansions"""
        query_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                enhanced_query="simple query",
                expansion_terms="",
                synonyms="",
                context="",
                confidence="0.6",
                reasoning="No expansions needed",
            )
        )

        result = await query_agent._process_impl(
            QueryEnhancementInput(query="simple query")
        )

        assert len(result.expansion_terms) == 0
        assert len(result.synonyms) == 0
        assert len(result.context_additions) == 0

    @pytest.mark.asyncio
    async def test_process_with_context(self, query_agent):
        """Test processing adds contextual terms"""
        query_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                enhanced_query="Python programming tutorials for beginners",
                expansion_terms="coding, development",
                synonyms="programming, scripting",
                context="beginner, introduction, basics",
                confidence="0.9",
                reasoning="Added beginner context and programming synonyms",
            )
        )

        result = await query_agent._process_impl(
            QueryEnhancementInput(query="Python tutorials")
        )

        assert len(result.context_additions) == 3
        assert "beginner" in result.context_additions
        assert "introduction" in result.context_additions

    def test_dspy_to_a2a_output(self, query_agent):
        """Test conversion to A2A output format"""
        result = QueryEnhancementResult(
            original_query="ML videos",
            enhanced_query="machine learning video tutorials",
            expansion_terms=["deep learning", "neural networks"],
            synonyms=["ML", "AI"],
            context_additions=["beginner", "fundamentals"],
            confidence=0.85,
            reasoning="Expanded ML and added context",
        )

        a2a_output = query_agent._dspy_to_a2a_output(result)

        assert a2a_output["status"] == "success"
        assert a2a_output["agent"] == "query_enhancement_agent"
        assert a2a_output["original_query"] == "ML videos"
        assert a2a_output["enhanced_query"] == "machine learning video tutorials"
        assert len(a2a_output["expansion_terms"]) == 2
        assert len(a2a_output["synonyms"]) == 2
        assert a2a_output["confidence"] == 0.85

    def test_get_agent_skills(self, query_agent):
        """Test agent skills definition"""
        skills = query_agent._get_agent_skills()

        assert len(skills) == 1
        assert skills[0]["name"] == "enhance_query"
        assert "query" in skills[0]["input_schema"]
        assert "enhanced_query" in skills[0]["output_schema"]
        assert "expansion_terms" in skills[0]["output_schema"]
        assert len(skills[0]["examples"]) > 0
