"""
Unit tests for QueryExpander

Tests query expansion functionality including visual-to-text conversion,
text-to-visual conversion, temporal extraction, and modality detection.
"""

import pytest

from cogniverse_agents.routing.query_expansion import QueryExpander


@pytest.mark.unit
class TestQueryExpander:
    """Test query expansion functionality"""

    @pytest.fixture
    def expander(self):
        """Create query expander instance"""
        return QueryExpander()

    @pytest.mark.asyncio
    async def test_visual_to_text_expansion(self, expander):
        """Test visual query expansion to text alternatives"""
        query = "show me machine learning"
        expansions = await expander.expand_visual_to_text(query)

        assert len(expansions) > 0
        assert "machine learning" in expansions
        assert any("video" in exp.lower() for exp in expansions)
        assert any("image" in exp.lower() for exp in expansions)

    @pytest.mark.asyncio
    async def test_text_to_visual_expansion(self, expander):
        """Test text query expansion to visual keywords"""
        query = "explain neural networks"
        result = await expander.expand_text_to_visual(query)

        assert "video_keywords" in result
        assert "image_keywords" in result
        assert len(result["video_keywords"]) > 0
        assert len(result["image_keywords"]) > 0

        # Check for expected patterns
        video_keywords = " ".join(result["video_keywords"])
        assert "tutorial" in video_keywords or "demonstration" in video_keywords

        image_keywords = " ".join(result["image_keywords"])
        assert "diagram" in image_keywords or "chart" in image_keywords

    @pytest.mark.asyncio
    async def test_temporal_expansion_year(self, expander):
        """Test temporal expansion with year"""
        query = "events in 2023"
        result = await expander.expand_temporal(query)

        assert result["requires_temporal_search"] is True
        assert result["temporal_type"] == "year"
        assert "2023" in result["temporal_keywords"]
        assert result["time_range"] is not None

        start, end = result["time_range"]
        assert start.year == 2023
        assert end.year == 2023

    @pytest.mark.asyncio
    async def test_temporal_expansion_relative(self, expander):
        """Test temporal expansion with relative time"""
        query = "videos from last week"
        result = await expander.expand_temporal(query)

        assert result["requires_temporal_search"] is True
        assert result["temporal_type"] == "relative"
        assert len(result["temporal_keywords"]) > 0

    @pytest.mark.asyncio
    async def test_modality_expansion(self, expander):
        """Test modality-specific expansion"""
        query = "machine learning"

        # Test video expansion
        video_exp = await expander.expand_for_modality(query, "video")
        assert any(
            "video" in exp.lower() or "tutorial" in exp.lower() for exp in video_exp
        )

        # Test image expansion
        image_exp = await expander.expand_for_modality(query, "image")
        assert any(
            "image" in exp.lower() or "diagram" in exp.lower() for exp in image_exp
        )

        # Test audio expansion
        audio_exp = await expander.expand_for_modality(query, "audio")
        assert any(
            "audio" in exp.lower() or "podcast" in exp.lower() for exp in audio_exp
        )

    def test_modality_intent_detection(self, expander):
        """Test modality intent detection"""
        # Video intent
        assert "video" in expander.detect_modality_intent(
            "show me a video about robotics"
        )

        # Image intent
        assert "image" in expander.detect_modality_intent("find images of cats")

        # Audio intent
        assert "audio" in expander.detect_modality_intent("listen to podcasts about AI")

        # Document intent
        assert "document" in expander.detect_modality_intent(
            "read papers about quantum computing"
        )

    @pytest.mark.asyncio
    async def test_comprehensive_expansion(self, expander):
        """Test comprehensive query expansion"""
        query = "show me tutorials from 2023"
        result = await expander.expand_query(query)

        assert "original_query" in result
        assert result["original_query"] == query
        assert "modality_intent" in result
        assert "temporal" in result
        assert "expansions" in result

        # Should have temporal info
        assert result["temporal"]["requires_temporal_search"] is True

        # Should have text alternatives (visual query)
        assert "text_alternatives" in result["expansions"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
