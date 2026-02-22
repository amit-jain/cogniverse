"""Unit tests for ProfileSelectionAgent"""

from unittest.mock import Mock, patch

import dspy
import pytest

from cogniverse_agents.profile_selection_agent import (
    ProfileCandidate,
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
    ProfileSelectionModule,
    ProfileSelectionOutput,
)


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model"""
    lm = Mock()
    lm.return_value = dspy.Prediction(
        selected_profile="video_colpali_base",
        confidence="0.9",
        reasoning="Query requests video content with machine learning topic, best matched by video_colpali_base profile",
        query_intent="video_search",
        modality="video",
        complexity="medium",
    )
    return lm


@pytest.fixture
def profile_agent():
    """Create ProfileSelectionAgent for testing"""
    with patch("dspy.ChainOfThought"):
        deps = ProfileSelectionDeps(
            tenant_id="test_tenant",
            available_profiles=[
                "video_colpali_base",
                "video_colpali_large",
                "image_colpali_base",
            ],
        )
        agent = ProfileSelectionAgent(deps=deps, port=8011)
        return agent


class TestProfileSelectionModule:
    """Test DSPy module for profile selection"""

    def test_module_initialization(self):
        """Test ProfileSelectionModule initializes correctly"""
        with patch("dspy.ChainOfThought") as mock_cot:
            module = ProfileSelectionModule()
            assert module.selector is not None
            mock_cot.assert_called_once()

    def test_forward_success(self, mock_dspy_lm):
        """Test successful profile selection"""
        module = ProfileSelectionModule()
        module.selector = mock_dspy_lm

        result = module.forward(
            query="Show me machine learning videos",
            available_profiles="video_colpali_base, video_colpali_large",
        )

        assert result.selected_profile == "video_colpali_base"
        assert result.confidence == "0.9"
        assert result.modality == "video"

    def test_forward_fallback(self):
        """Test fallback when DSPy fails"""
        module = ProfileSelectionModule()
        module.selector = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(
            query="Show me videos about cats",
            available_profiles="video_colpali_base, text_bge_base",
        )

        # Fallback should detect video modality
        assert result.modality == "video"
        assert result.query_intent == "video_search"
        assert result.selected_profile in ["video_colpali_base", "text_bge_base"]

    def test_fallback_image_detection(self):
        """Test fallback detects image queries"""
        module = ProfileSelectionModule()
        module.selector = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(
            query="Show me pictures of mountains",
            available_profiles="image_colpali_base, video_colpali_base",
        )

        assert result.modality == "image"
        assert result.query_intent == "image_search"

    def test_fallback_complexity_simple(self):
        """Test fallback detects simple query"""
        module = ProfileSelectionModule()
        module.selector = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(
            query="cat videos", available_profiles="video_colpali_base"
        )

        assert result.complexity == "simple"

    def test_fallback_complexity_complex(self):
        """Test fallback detects complex query"""
        module = ProfileSelectionModule()
        module.selector = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(
            query="Show me detailed tutorials about advanced machine learning techniques with neural networks",
            available_profiles="video_colpali_base",
        )

        assert result.complexity == "complex"


class TestProfileSelectionAgent:
    """Test ProfileSelectionAgent core functionality"""

    def test_agent_initialization(self, profile_agent):
        """Test agent initializes with correct configuration"""
        assert profile_agent.agent_name == "profile_selection_agent"
        assert "profile_selection" in profile_agent.capabilities
        assert len(profile_agent.deps.available_profiles) == 3

    @pytest.mark.asyncio
    async def test_process_with_query(self, profile_agent):
        """Test processing query for profile selection"""
        profile_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                selected_profile="video_colpali_base",
                confidence="0.9",
                reasoning="Best match for video search",
                query_intent="video_search",
                modality="video",
                complexity="medium",
            )
        )

        result = await profile_agent._process_impl(
            ProfileSelectionInput(query="Show me machine learning videos")
        )

        assert isinstance(result, ProfileSelectionOutput)
        assert result.query == "Show me machine learning videos"
        assert result.selected_profile == "video_colpali_base"
        assert result.confidence == 0.9
        assert result.query_intent == "video_search"
        assert result.modality == "video"
        assert result.complexity == "medium"

    @pytest.mark.asyncio
    async def test_process_empty_query(self, profile_agent):
        """Test processing empty query"""
        result = await profile_agent._process_impl(ProfileSelectionInput(query=""))

        assert result.query == ""
        assert result.confidence == 0.0
        assert result.reasoning == "Empty query, using default profile"
        assert result.selected_profile == profile_agent.deps.available_profiles[0]

    @pytest.mark.asyncio
    async def test_process_custom_profiles(self, profile_agent):
        """Test processing with custom available profiles"""
        profile_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                selected_profile="custom_profile_1",
                confidence="0.85",
                reasoning="Custom profile match",
                query_intent="text_search",
                modality="text",
                complexity="simple",
            )
        )

        result = await profile_agent._process_impl(
            ProfileSelectionInput(
                query="test query",
                available_profiles=["custom_profile_1", "custom_profile_2"],
            )
        )

        assert result.selected_profile == "custom_profile_1"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_process_invalid_confidence(self, profile_agent):
        """Test processing with invalid confidence value"""
        profile_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                selected_profile="video_colpali_base",
                confidence="invalid",  # Invalid confidence
                reasoning="Test",
                query_intent="video_search",
                modality="video",
                complexity="medium",
            )
        )

        result = await profile_agent._process_impl(ProfileSelectionInput(query="test"))

        assert result.confidence == 0.5  # Default fallback

    def test_generate_alternatives(self, profile_agent):
        """Test alternative profile generation"""
        profiles = ["video_colpali_base", "video_colpali_large", "image_colpali_base"]
        selected = "video_colpali_base"
        modality = "video"

        alternatives = profile_agent._generate_alternatives(
            query="test query", profiles=profiles, selected=selected, modality=modality
        )

        # Should not include selected profile
        assert all(alt.profile_name != selected for alt in alternatives)

        # Video profiles should rank higher
        if alternatives:
            video_alts = [
                alt for alt in alternatives if "video" in alt.profile_name.lower()
            ]
            if video_alts:
                assert video_alts[0].score > 0.3

    def test_generate_alternatives_string_input(self, profile_agent):
        """Test alternative generation with string input"""
        profiles_str = "video_colpali_base, video_colpali_large"
        selected = "video_colpali_base"
        modality = "video"

        alternatives = profile_agent._generate_alternatives(
            query="test", profiles=profiles_str, selected=selected, modality=modality
        )

        assert isinstance(alternatives, list)

    def test_generate_alternatives_max_three(self, profile_agent):
        """Test alternatives limited to top 3"""
        profiles = [f"profile_{i}" for i in range(10)]
        selected = "profile_0"
        modality = "video"

        alternatives = profile_agent._generate_alternatives(
            query="test", profiles=profiles, selected=selected, modality=modality
        )

        assert len(alternatives) <= 3

    def test_dspy_to_a2a_output(self, profile_agent):
        """Test conversion to A2A output format"""
        result = ProfileSelectionOutput(
            query="test query",
            selected_profile="video_colpali_base",
            confidence=0.9,
            reasoning="Best match for video",
            query_intent="video_search",
            modality="video",
            complexity="medium",
            alternatives=[
                ProfileCandidate(
                    profile_name="video_colpali_large",
                    score=0.7,
                    reasoning="Alternative video profile",
                )
            ],
        )

        a2a_output = profile_agent._dspy_to_a2a_output(result)

        assert a2a_output["status"] == "success"
        assert a2a_output["agent"] == "profile_selection_agent"
        assert a2a_output["selected_profile"] == "video_colpali_base"
        assert a2a_output["confidence"] == 0.9
        assert a2a_output["query_intent"] == "video_search"
        assert len(a2a_output["alternatives"]) == 1

    def test_get_agent_skills(self, profile_agent):
        """Test agent skills definition"""
        skills = profile_agent._get_agent_skills()

        assert len(skills) == 1
        assert skills[0]["name"] == "select_profile"
        assert "query" in skills[0]["input_schema"]
        assert "selected_profile" in skills[0]["output_schema"]
        assert len(skills[0]["examples"]) > 0
