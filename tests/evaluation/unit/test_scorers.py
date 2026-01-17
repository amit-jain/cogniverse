"""
Unit tests for evaluation scorers.
"""

from unittest.mock import Mock, patch

import pytest
from cogniverse_evaluation.core.scorers import (
    _calculate_keyword_relevance,
    diversity_scorer,
    get_configured_scorers,
    precision_scorer,
    recall_scorer,
    relevance_scorer,
    schema_aware_temporal_scorer,
)


class TestConfiguredScorers:
    """Test scorer configuration."""

    @pytest.mark.unit
    def test_configured_scorers_requires_config(self):
        """Test that get_configured_scorers requires config."""
        with pytest.raises(ValueError, match="config is required"):
            get_configured_scorers(None)

    @pytest.mark.unit
    def test_default_scorer_configuration(self):
        """Test default scorer configuration."""
        config = {
            "use_relevance": True,
            "use_diversity": True,
            "use_temporal": False,
            "use_precision_recall": False,
            "enable_llm_evaluators": False,
            "enable_quality_evaluators": False,
        }
        scorers = get_configured_scorers(config)
        assert len(scorers) == 2  # relevance and diversity

    @pytest.mark.unit
    def test_precision_recall_configuration(self):
        """Test configuring precision/recall scorers."""
        config = {
            "use_relevance": False,
            "use_diversity": False,
            "use_precision_recall": True,
            "enable_llm_evaluators": False,
            "enable_quality_evaluators": False,
        }
        scorers = get_configured_scorers(config)
        assert len(scorers) == 2  # precision and recall

    @pytest.mark.unit
    def test_visual_evaluator_configuration(self):
        """Test configuring visual evaluators."""
        config = {
            "use_relevance": False,
            "use_diversity": False,
            "use_precision_recall": False,
            "enable_llm_evaluators": True,
            "enable_quality_evaluators": True,
            "evaluator_name": "test_judge",
        }

        with patch(
            "cogniverse_evaluation.plugins.visual_evaluator.get_visual_scorers"
        ) as mock_get:
            mock_get.return_value = [Mock(), Mock()]  # Two mock scorers
            scorers = get_configured_scorers(config)
            assert len(scorers) == 2  # visual judge and quality scorer
            mock_get.assert_called_once_with(config)

    @pytest.mark.unit
    def test_temporal_scorer_configuration(self):
        """Test temporal scorer configuration."""
        config = {"use_relevance": False, "use_diversity": False, "use_temporal": True}
        scorers = get_configured_scorers(config)
        assert len(scorers) == 1


class TestRelevanceScorer:
    """Test relevance scorer."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_relevance_scorer_with_results(self):
        """Test relevance scorer with results."""
        state = Mock()
        state.input = {"query": "test query"}
        state.outputs = {
            "config1": {
                "success": True,
                "results": [
                    {"content": "test query results here"},
                    {"text": "another test result"},
                    {"description": "query related content"},
                ],
            }
        }

        scorer = relevance_scorer()
        score = await scorer(state)

        assert score is not None
        assert 0.0 <= score.value <= 1.0
        assert "Relevance scores" in score.explanation
        assert "individual_scores" in score.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_relevance_scorer_no_results(self):
        """Test relevance scorer with no results."""
        state = Mock()
        state.input = {"query": "test"}
        state.outputs = {"config1": {"success": False}}

        scorer = relevance_scorer()
        score = await scorer(state)

        assert score.value == 0.0
        assert "config1=0.000" in score.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_relevance_scorer_empty_content(self):
        """Test relevance scorer with empty content."""
        state = Mock()
        state.input = {"query": "test"}
        state.outputs = {
            "config1": {
                "success": True,
                "results": [{"content": ""}, {"content": None}],
            }
        }

        scorer = relevance_scorer()
        score = await scorer(state)

        assert score.value == 0.0


class TestDiversityScorer:
    """Test diversity scorer."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_diversity_scorer_with_analyzer(self):
        """Test diversity scorer using schema analyzer."""
        # Mock the analyzer
        with patch(
            "cogniverse_evaluation.core.scorers.get_schema_analyzer"
        ) as mock_get_analyzer:
            mock_analyzer = Mock()
            mock_analyzer.extract_item_id.side_effect = ["item1", "item2", "item1"]
            mock_get_analyzer.return_value = mock_analyzer

            state = Mock()
            state.metadata = {"schema_name": "test", "schema_fields": {}}
            state.outputs = {
                "config1": {
                    "success": True,
                    "results": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
                }
            }

            scorer = diversity_scorer()
            score = await scorer(state)

            assert score is not None
            # 2 unique items out of 3 results = 0.666...
            assert score.value == pytest.approx(2 / 3, rel=1e-3)
            assert "Result diversity" in score.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_diversity_scorer_all_unique(self):
        """Test diversity scorer with all unique results."""
        with patch(
            "cogniverse_evaluation.core.scorers.get_schema_analyzer"
        ) as mock_get_analyzer:
            mock_analyzer = Mock()
            mock_analyzer.extract_item_id.side_effect = ["item1", "item2", "item3"]
            mock_get_analyzer.return_value = mock_analyzer

            state = Mock()
            state.metadata = {"schema_name": "test", "schema_fields": {}}
            state.outputs = {
                "config1": {
                    "success": True,
                    "results": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
                }
            }

            scorer = diversity_scorer()
            score = await scorer(state)
            assert score.value == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_diversity_scorer_no_results(self):
        """Test diversity scorer with no results."""
        state = Mock()
        state.metadata = {}
        state.outputs = {"config1": {"success": True, "results": []}}

        scorer = diversity_scorer()
        score = await scorer(state)
        assert score.value == 0.0


class TestTemporalScorer:
    """Test schema-aware temporal scorer."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temporal_scorer_with_temporal_schema(self):
        """Test temporal scorer with temporal schema."""
        state = Mock()
        state.input = {"query": "what happened after the meeting"}
        state.metadata = {
            "schema_fields": {
                "temporal_fields": ["timestamp", "start_time", "end_time"]
            }
        }
        state.outputs = {
            "config1": {
                "success": True,
                "results": [{"timestamp": 100}, {"timestamp": 200}, {"timestamp": 300}],
            }
        }

        scorer = schema_aware_temporal_scorer()
        score = await scorer(state)

        assert score is not None
        assert score.value == 1.0  # Properly ordered

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temporal_scorer_unordered(self):
        """Test temporal scorer with unordered results."""
        state = Mock()
        state.input = {"query": "when did this happen"}
        state.metadata = {"schema_fields": {"temporal_fields": ["timestamp"]}}
        state.outputs = {
            "config1": {
                "success": True,
                "results": [{"timestamp": 300}, {"timestamp": 100}, {"timestamp": 200}],
            }
        }

        scorer = schema_aware_temporal_scorer()
        score = await scorer(state)
        assert score.value == 0.0  # Not ordered

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temporal_scorer_non_temporal_schema(self):
        """Test temporal scorer with non-temporal schema."""
        state = Mock()
        state.input = {"query": "when did this happen"}
        state.metadata = {
            "schema_fields": {"temporal_fields": []}  # No temporal fields
        }
        state.outputs = {"config1": {"success": True, "results": []}}

        scorer = schema_aware_temporal_scorer()
        score = await scorer(state)

        assert score.value == 1.0  # N/A returns perfect score
        assert "Not a temporal schema" in score.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temporal_scorer_non_temporal_query(self):
        """Test temporal scorer with non-temporal query."""
        state = Mock()
        state.input = {"query": "red car"}
        state.metadata = {"schema_fields": {"temporal_fields": ["timestamp"]}}
        state.outputs = {"config1": {"success": True, "results": []}}

        scorer = schema_aware_temporal_scorer()
        score = await scorer(state)

        assert score.value == 1.0  # N/A returns perfect score
        assert "Not a temporal query" in score.explanation


class TestPrecisionRecallScorers:
    """Test precision and recall scorers."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_precision_scorer_with_ground_truth(self):
        """Test precision scorer with ground truth."""
        with patch(
            "cogniverse_evaluation.core.scorers.get_schema_analyzer"
        ) as mock_get_analyzer:
            mock_analyzer = Mock()
            mock_analyzer.extract_item_id.side_effect = ["item1", "item2", "item3"]
            mock_get_analyzer.return_value = mock_analyzer

            state = Mock()
            state.output = {"expected_items": ["item1", "item3", "item5"]}
            state.metadata = {"schema_name": "test", "schema_fields": {}}
            state.outputs = {
                "config1": {
                    "success": True,
                    "results": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
                }
            }

            scorer = precision_scorer()
            score = await scorer(state)

            assert score is not None
            # 2 out of 3 retrieved are relevant
            assert score.value == pytest.approx(2 / 3, rel=1e-3)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_recall_scorer_with_ground_truth(self):
        """Test recall scorer with ground truth."""
        with patch(
            "cogniverse_evaluation.core.scorers.get_schema_analyzer"
        ) as mock_get_analyzer:
            mock_analyzer = Mock()
            mock_analyzer.extract_item_id.side_effect = ["item1", "item2", "item3"]
            mock_get_analyzer.return_value = mock_analyzer

            state = Mock()
            state.output = {"expected_items": ["item1", "item3", "item5"]}
            state.metadata = {"schema_name": "test", "schema_fields": {}}
            state.outputs = {
                "config1": {
                    "success": True,
                    "results": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
                }
            }

            scorer = recall_scorer()
            score = await scorer(state)

            assert score is not None
            # 2 out of 3 expected were retrieved
            assert score.value == pytest.approx(2 / 3, rel=1e-3)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_precision_scorer_no_ground_truth(self):
        """Test precision scorer without ground truth."""
        state = Mock(spec=["metadata", "outputs"])  # Specify attributes explicitly
        # No output attribute means no ground truth
        state.metadata = {}
        state.outputs = {"config1": {"success": True, "results": []}}

        scorer = precision_scorer()
        score = await scorer(state)

        assert score.value == 0.0
        assert "No ground truth available" in score.explanation


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.unit
    def test_calculate_keyword_relevance(self):
        """Test keyword relevance calculation."""
        query = "red car park"
        contexts = [
            "red car in the park",  # 3/3 words match
            "blue car",  # 1/3 words match
            "something else",  # 0/3 words match
        ]

        relevancy = _calculate_keyword_relevance(query, contexts)
        # (1.0 + 0.333 + 0) / 3 ≈ 0.444
        assert relevancy == pytest.approx(0.444, rel=0.01)

    @pytest.mark.unit
    def test_calculate_keyword_relevance_empty(self):
        """Test keyword relevance with empty contexts."""
        assert _calculate_keyword_relevance("query", []) == 0.0
        assert _calculate_keyword_relevance("", ["context"]) == 0.0
