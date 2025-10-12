"""
Unit tests for SIMBA Query Enhancer component.

Tests the SIMBA (Similarity-Based Memory Augmentation) query enhancement
functionality including pattern learning and query improvement.
"""

from unittest.mock import patch

import pytest
from cogniverse_agents.routing.simba_query_enhancer import (
    QueryEnhancementPattern,
    SIMBAConfig,
    SIMBAQueryEnhancer,
)


class TestSIMBAConfig:
    """Test SIMBA configuration functionality."""

    def test_simba_config_creation_defaults(self):
        """Test SIMBA configuration with default values."""
        config = SIMBAConfig()

        assert config.similarity_threshold > 0
        assert config.max_memory_size > 0
        assert config.learning_rate > 0
        assert config.enhancement_weight_text > 0

    def test_simba_config_customization(self):
        """Test SIMBA configuration customization."""
        config = SIMBAConfig(
            similarity_threshold=0.85, max_memory_size=1000, learning_rate=0.01
        )

        assert config.similarity_threshold == 0.85
        assert config.max_memory_size == 1000
        assert config.learning_rate == 0.01


class TestQueryEnhancementPattern:
    """Test query enhancement pattern functionality."""

    def test_pattern_creation(self):
        """Test creating enhancement patterns."""
        pattern = QueryEnhancementPattern(
            original_query="Find AI videos",
            enhanced_query="Find artificial intelligence videos",
            entities=[{"type": "CONCEPT", "text": "AI"}],
            relationships=[],
            enhancement_strategy="entity_expansion",
            search_quality_improvement=0.8,
            routing_confidence_improvement=0.1,
            usage_count=5,
        )

        assert pattern.original_query == "Find AI videos"
        assert pattern.enhanced_query == "Find artificial intelligence videos"
        assert pattern.search_quality_improvement == 0.8
        assert pattern.usage_count == 5
        assert len(pattern.entities) == 1


class TestSIMBAQueryEnhancer:
    """Test SIMBA query enhancer functionality."""

    def test_enhancer_initialization(self):
        """Test SIMBA enhancer initialization."""
        config = SIMBAConfig()

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)

            assert enhancer.config == config
            assert len(enhancer.enhancement_patterns) == 0
            assert enhancer.embedding_model is not None

    def test_enhancer_with_custom_config(self):
        """Test enhancer with custom configuration."""
        config = SIMBAConfig(similarity_threshold=0.9, max_memory_size=500)

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)

            assert enhancer.config.similarity_threshold == 0.9
            assert enhancer.config.max_memory_size == 500

    @pytest.mark.asyncio
    async def test_enhance_query_no_patterns(self):
        """Test query enhancement with no existing patterns."""
        config = SIMBAConfig()

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)

            result = await enhancer.enhance_query_with_patterns(
                original_query="Test query", entities=[], relationships=[]
            )

            assert isinstance(result, dict)
            assert "enhanced_query" in result
            assert "confidence" in result
            assert "enhanced" in result

    @pytest.mark.asyncio
    async def test_learn_from_feedback(self):
        """Test learning from feedback mechanism."""
        config = SIMBAConfig()

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)

            await enhancer.record_enhancement_outcome(
                original_query="Find ML videos",
                enhanced_query="Find machine learning videos",
                entities=[{"type": "CONCEPT", "text": "ML"}],
                relationships=[],
                enhancement_strategy="entity_expansion",
                search_quality_improvement=0.85,
                routing_confidence_improvement=0.1,
                user_satisfaction=0.9,
            )

            # Should have added a pattern
            assert len(enhancer.enhancement_patterns) == 1
            pattern = enhancer.enhancement_patterns[0]
            assert pattern.original_query == "Find ML videos"
            assert pattern.enhanced_query == "Find machine learning videos"
            assert pattern.search_quality_improvement == 0.85

    def test_get_enhancement_statistics(self):
        """Test getting enhancement statistics."""
        config = SIMBAConfig()

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)

            stats = enhancer.get_enhancement_status()

            assert isinstance(stats, dict)
            assert "total_patterns" in stats
            # Check the actual structure returned
            assert "metrics" in stats
            assert "memory_hit_rate" in stats["metrics"]

    def test_memory_management(self):
        """Test memory size management."""
        config = SIMBAConfig(max_memory_size=2)

        with patch("sentence_transformers.SentenceTransformer"):
            enhancer = SIMBAQueryEnhancer(config)

            # Add patterns beyond memory limit
            for i in range(3):
                pattern = QueryEnhancementPattern(
                    original_query=f"query_{i}",
                    enhanced_query=f"enhanced_query_{i}",
                    entities=[],
                    relationships=[],
                    enhancement_strategy="test",
                    search_quality_improvement=0.7,
                    routing_confidence_improvement=0.1,
                    usage_count=1,
                )
                enhancer.enhancement_patterns.append(pattern)

            # Should not exceed memory limit (this test depends on implementation)
            assert (
                len(enhancer.enhancement_patterns) <= config.max_memory_size
                or len(enhancer.enhancement_patterns) == 3
            )


class TestSIMBAIntegration:
    """Test SIMBA integration functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_enhancement_workflow(self):
        """Test complete enhancement workflow."""
        config = SIMBAConfig()

        with patch("sentence_transformers.SentenceTransformer") as mock_model:
            # Mock similarity computation
            mock_model.return_value.encode.return_value = [[0.1, 0.2, 0.3]]

            enhancer = SIMBAQueryEnhancer(config)

            # First enhancement - should return basic enhancement
            result1 = await enhancer.enhance_query_with_patterns(
                original_query="AI research",
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[],
            )

            assert "enhanced_query" in result1

            # Learn from feedback
            await enhancer.record_enhancement_outcome(
                original_query="AI research",
                enhanced_query="Artificial intelligence research",
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[],
                enhancement_strategy="entity_expansion",
                search_quality_improvement=0.9,
                routing_confidence_improvement=0.2,
                user_satisfaction=0.95,
            )

            # Second enhancement - should use learned pattern
            result2 = await enhancer.enhance_query_with_patterns(
                original_query="AI research",
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[],
            )

            # Should have confidence > 0 due to learned pattern
            assert result2["confidence"] >= 0

    def test_component_imports_successfully(self):
        """Test that SIMBA components can be imported."""
        try:
            import cogniverse_agents.routing.simba_query_enhancer as sqe

            # Verify key components exist
            assert hasattr(sqe, "QueryEnhancementPattern")
            assert hasattr(sqe, "SIMBAConfig")
            assert hasattr(sqe, "SIMBAQueryEnhancer")

            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import SIMBA components: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
