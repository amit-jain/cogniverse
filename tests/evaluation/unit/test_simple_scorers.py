"""
Unit tests for simple scorers.
"""

from unittest.mock import Mock

import pytest
from cogniverse_core.evaluation.core.simple_scorers import (
    get_configured_scorers,
    simple_relevance_scorer,
)


class TestSimpleScorers:
    """Test simple scorer functions."""

    @pytest.mark.unit
    def test_get_configured_scorers_default(self):
        """Test getting configured scorers with default config."""
        config = {}
        scorers = get_configured_scorers(config)

        assert isinstance(scorers, list)
        assert len(scorers) == 1
        # Should return simple_relevance_scorer

    @pytest.mark.unit
    def test_get_configured_scorers_with_config(self):
        """Test getting configured scorers with custom config."""
        config = {"scorers": ["simple_relevance"], "scorer_config": {"threshold": 0.7}}
        scorers = get_configured_scorers(config)

        assert isinstance(scorers, list)
        assert len(scorers) >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_relevance_scorer_creation(self):
        """Test simple relevance scorer creation."""
        scorer_func = simple_relevance_scorer()

        assert scorer_func is not None
        assert callable(scorer_func)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_relevance_scorer_execution(self):
        """Test simple relevance scorer execution."""
        scorer_func = simple_relevance_scorer()

        # Create mock state and target
        state = Mock()
        state.output = "test output"
        target = "test target"

        # Execute scorer
        score = await scorer_func(state, target)

        assert score is not None
        assert score.value == 0.5
        assert (
            "testing" in score.explanation.lower()
            or "simplified" in score.explanation.lower()
        )
        assert score.metadata["scorer"] == "simple_relevance"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_relevance_scorer_with_empty_input(self):
        """Test scorer with empty input."""
        scorer_func = simple_relevance_scorer()

        state = Mock()
        state.output = ""
        target = ""

        score = await scorer_func(state, target)

        # Should still return a score
        assert score.value == 0.5
        assert score.metadata is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_relevance_scorer_with_none_input(self):
        """Test scorer with None input."""
        scorer_func = simple_relevance_scorer()

        state = None
        target = None

        score = await scorer_func(state, target)

        # Should handle None gracefully
        assert score.value == 0.5
        assert score.explanation is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scorer_metadata(self):
        """Test scorer metadata."""
        scorer_func = simple_relevance_scorer()

        state = Mock()
        state.output = "test"
        state.metadata = {"query": "test query"}
        target = "expected"

        score = await scorer_func(state, target)

        assert "scorer" in score.metadata
        assert score.metadata["scorer"] == "simple_relevance"

    @pytest.mark.unit
    def test_scorer_decorator_usage(self):
        """Test that scorer decorator is properly applied."""
        # The simple_relevance_scorer should be decorated
        assert hasattr(simple_relevance_scorer, "__name__")
        assert simple_relevance_scorer.__name__ == "simple_relevance_scorer"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scorer_consistency(self):
        """Test scorer returns consistent results."""
        scorer_func = simple_relevance_scorer()

        state = Mock()
        state.output = "test"
        target = "test"

        # Run multiple times
        scores = []
        for _ in range(3):
            score = await scorer_func(state, target)
            scores.append(score.value)

        # Should return same score
        assert all(s == 0.5 for s in scores)

    @pytest.mark.unit
    def test_get_configured_scorers_empty_list(self):
        """Test getting scorers with empty config."""
        config = {"scorers": []}
        scorers = get_configured_scorers(config)

        # Should still return default scorers
        assert len(scorers) >= 1

    @pytest.mark.unit
    def test_get_configured_scorers_invalid_config(self):
        """Test getting scorers with invalid config."""
        config = {"scorers": "invalid"}  # Should be a list
        scorers = get_configured_scorers(config)

        # Should handle gracefully and return defaults
        assert isinstance(scorers, list)
        assert len(scorers) >= 1
