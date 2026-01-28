"""
Unit tests for ContextualAnalyzer

Tests contextual analysis functionality including conversation tracking,
modality preferences, topic tracking, and transition pattern analysis.
"""

import pytest

from cogniverse_agents.routing.contextual_analyzer import ContextualAnalyzer


@pytest.mark.unit
class TestContextualAnalyzer:
    """Test contextual analysis functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create contextual analyzer instance"""
        return ContextualAnalyzer()

    def test_context_update(self, analyzer):
        """Test context tracking"""
        analyzer.update_context(
            query="machine learning tutorials",
            detected_modalities=["video"],
            result_count=5,
        )

        assert analyzer.total_queries == 1
        assert analyzer.successful_queries == 1
        assert len(analyzer.conversation_history) == 1
        assert "video" in analyzer.modality_preferences

    def test_modality_preferences(self, analyzer):
        """Test modality preference learning"""
        # Add multiple queries with video preference
        for i in range(5):
            analyzer.update_context(
                query=f"query {i}",
                detected_modalities=["video"],
                result_count=3,
            )

        # Add some image queries
        for i in range(2):
            analyzer.update_context(
                query=f"image query {i}",
                detected_modalities=["image"],
                result_count=3,
            )

        top_modalities = analyzer._get_top_modalities()

        assert len(top_modalities) > 0
        assert top_modalities[0]["modality"] == "video"
        assert top_modalities[0]["count"] == 5

    def test_topic_tracking(self, analyzer):
        """Test topic tracking"""
        analyzer.update_context(
            query="machine learning algorithms",
            detected_modalities=["text"],
            result_count=5,
        )

        analyzer.update_context(
            query="deep learning neural networks",
            detected_modalities=["text"],
            result_count=4,
        )

        assert "machine" in analyzer.topic_tracking
        assert "learning" in analyzer.topic_tracking

    def test_contextual_hints(self, analyzer):
        """Test contextual hints generation"""
        # Build some history
        analyzer.update_context(
            query="machine learning videos",
            detected_modalities=["video"],
            result_count=5,
        )

        analyzer.update_context(
            query="neural networks tutorial",
            detected_modalities=["video"],
            result_count=3,
        )

        hints = analyzer.get_contextual_hints("deep learning")

        assert "preferred_modalities" in hints
        assert "related_topics" in hints
        assert "temporal_context" in hints
        assert "conversation_context" in hints
        assert "session_metrics" in hints

    def test_conversation_context(self, analyzer):
        """Test conversation context analysis"""
        # Add queries with modality shifts
        analyzer.update_context(
            query="video tutorials",
            detected_modalities=["video"],
            result_count=5,
        )

        analyzer.update_context(
            query="image diagrams",
            detected_modalities=["image"],
            result_count=3,
        )

        analyzer.update_context(
            query="more videos",
            detected_modalities=["video"],
            result_count=4,
        )

        context = analyzer._get_conversation_context()

        assert context["conversation_depth"] == 3
        assert context["modality_shifts"] > 0

    def test_modality_transitions(self, analyzer):
        """Test modality transition pattern analysis"""
        # Create transition pattern: video -> image -> document
        analyzer.update_context("q1", ["video"], result_count=1)
        analyzer.update_context("q2", ["image"], result_count=1)
        analyzer.update_context("q3", ["document"], result_count=1)

        transitions = analyzer.get_modality_transition_patterns()

        assert "video" in transitions
        assert "image" in transitions["video"]

    def test_next_modality_suggestion(self, analyzer):
        """Test next modality suggestion"""
        # Create pattern: video often followed by image
        for _ in range(3):
            analyzer.update_context("video query", ["video"], result_count=1)
            analyzer.update_context("image query", ["image"], result_count=1)

        suggestion = analyzer.suggest_next_modality(["video"])
        assert suggestion == "image"

    def test_context_export(self, analyzer):
        """Test context export"""
        analyzer.update_context(
            query="test query",
            detected_modalities=["video"],
            result_count=1,
        )

        exported = analyzer.export_context()

        assert "conversation_history" in exported
        assert "modality_preferences" in exported
        assert "topic_tracking" in exported
        assert len(exported["conversation_history"]) == 1

    def test_clear_context(self, analyzer):
        """Test context clearing"""
        analyzer.update_context("test", ["video"], result_count=1)
        assert analyzer.total_queries == 1

        analyzer.clear_context()

        assert analyzer.total_queries == 0
        assert len(analyzer.conversation_history) == 0
        assert len(analyzer.modality_preferences) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
