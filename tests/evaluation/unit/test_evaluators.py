"""
Unit tests for evaluation evaluators modules.
"""

from unittest.mock import AsyncMock

import pytest
from phoenix.experiments.types import EvaluationResult

from cogniverse_core.evaluation.evaluators.base_evaluator import NoSpanEvaluator, no_span
from cogniverse_core.evaluation.evaluators.reference_free import (
    CompositeEvaluator,
    LLMRelevanceEvaluator,
    QueryResultRelevanceEvaluator,
    ResultDiversityEvaluator,
    RetrievalContext,
    TemporalCoverageEvaluator,
    create_reference_free_evaluators,
)


class TestBaseEvaluator:
    """Test base evaluator functionality."""

    @pytest.mark.unit
    def test_no_span_evaluator_with_evaluate_method(self):
        """Test NoSpanEvaluator with _evaluate method."""

        class TestEvaluator(NoSpanEvaluator):
            def _evaluate(self, **kwargs):
                return {"result": "test", "kwargs": kwargs}

        evaluator = TestEvaluator()
        result = evaluator.evaluate(test_param="test_value")

        assert result["result"] == "test"
        assert result["kwargs"]["test_param"] == "test_value"

    @pytest.mark.unit
    def test_no_span_evaluator_without_evaluate_method(self):
        """Test NoSpanEvaluator without _evaluate method raises NotImplementedError."""
        evaluator = NoSpanEvaluator()

        with pytest.raises(
            NotImplementedError, match="Subclass must implement _evaluate method"
        ):
            evaluator.evaluate()

    @pytest.mark.unit
    def test_no_span_decorator(self):
        """Test no_span decorator functionality."""

        @no_span
        def test_function(x, y=None):
            return f"result: {x}, {y}"

        result = test_function("test", y="value")
        assert result == "result: test, value"
        assert hasattr(test_function, "_skip_instrumentation")
        assert test_function._skip_instrumentation is True


class TestRetrievalContext:
    """Test RetrievalContext dataclass."""

    @pytest.mark.unit
    def test_retrieval_context_creation(self):
        """Test creating RetrievalContext with required fields."""
        context = RetrievalContext(
            query="test query", results=[{"id": "1", "score": 0.9}]
        )

        assert context.query == "test query"
        assert len(context.results) == 1
        assert context.metadata is None

    @pytest.mark.unit
    def test_retrieval_context_with_metadata(self):
        """Test creating RetrievalContext with metadata."""
        metadata = {"profile": "test", "strategy": "test"}
        context = RetrievalContext(query="test query", results=[], metadata=metadata)

        assert context.metadata == metadata


class TestQueryResultRelevanceEvaluator:
    """Test QueryResultRelevanceEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return QueryResultRelevanceEvaluator(min_score_threshold=0.5)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_no_results(self, evaluator):
        """Test evaluation with no results."""
        result = await evaluator.evaluate("test query", [])

        assert result.score == 0.0
        assert result.label == "no_results"
        assert "No results returned" in result.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_high_relevance(self, evaluator):
        """Test evaluation with high relevance scores."""
        output = [{"score": 0.9}, {"score": 0.85}, {"score": 0.8}]

        result = await evaluator.evaluate("test query", output)

        assert result.score >= 0.8
        assert result.label == "highly_relevant"
        assert "Average relevance score" in result.explanation
        assert result.metadata["num_results"] == 3
        assert len(result.metadata["top_scores"]) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_moderate_relevance(self, evaluator):
        """Test evaluation with moderate relevance scores."""
        output = [{"score": 0.6}, {"score": 0.55}, {"score": 0.5}]

        result = await evaluator.evaluate("test query", output)

        assert 0.5 <= result.score < 0.8
        assert result.label == "relevant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_low_relevance(self, evaluator):
        """Test evaluation with low relevance scores."""
        output = [{"score": 0.3}, {"score": 0.2}, {"score": 0.1}]

        result = await evaluator.evaluate("test query", output)

        assert result.score < 0.5
        assert result.label == "low_relevance"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_alternative_score_field(self, evaluator):
        """Test evaluation with relevance_score field."""
        output = [{"relevance_score": 0.9}, {"relevance_score": 0.8}]

        result = await evaluator.evaluate("test query", output)

        assert result.score >= 0.8
        assert result.label == "highly_relevant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_more_than_five_results(self, evaluator):
        """Test evaluation with more than 5 results (should only use top 5)."""
        output = [{"score": 0.9}] * 10  # 10 results

        result = await evaluator.evaluate("test query", output)

        # Should only consider top 5
        assert len(result.metadata["top_scores"]) == 5


class TestResultDiversityEvaluator:
    """Test ResultDiversityEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return ResultDiversityEvaluator()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_insufficient_results(self, evaluator):
        """Test evaluation with insufficient results."""
        result = await evaluator.evaluate("test query", [{"id": "1"}])

        assert result.score == 0.0
        assert result.label == "insufficient_results"
        assert "Need at least 2 results" in result.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_high_diversity(self, evaluator):
        """Test evaluation with high diversity."""
        output = [
            {"source_id": "video1"},
            {"source_id": "video2"},
            {"source_id": "video3"},
            {"source_id": "video4"},
        ]

        result = await evaluator.evaluate("test query", output)

        assert result.score == 1.0  # 4 unique / 4 total
        assert result.label == "high_diversity"
        assert result.metadata["unique_videos"] == 4
        assert result.metadata["total_results"] == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_moderate_diversity(self, evaluator):
        """Test evaluation with moderate diversity."""
        output = [
            {"source_id": "video1"},
            {"source_id": "video2"},
            {"source_id": "video1"},  # Duplicate
            {"source_id": "video3"},
        ]

        result = await evaluator.evaluate("test query", output)

        assert result.score == 0.75  # 3 unique / 4 total
        assert result.label == "moderate_diversity"
        assert result.metadata["unique_videos"] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_low_diversity(self, evaluator):
        """Test evaluation with low diversity."""
        output = [
            {"source_id": "video1"},
            {"source_id": "video1"},
            {"source_id": "video1"},
            {"source_id": "video2"},
        ]

        result = await evaluator.evaluate("test query", output)

        assert result.score == 0.5  # 2 unique / 4 total
        assert result.label == "moderate_diversity"  # 0.5 is still moderate

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_video_id_field(self, evaluator):
        """Test evaluation using video_id field instead of source_id."""
        output = [{"video_id": "video1"}, {"video_id": "video2"}]

        result = await evaluator.evaluate("test query", output)

        assert result.score == 1.0
        assert result.metadata["unique_videos"] == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_missing_id_fields(self, evaluator):
        """Test evaluation with missing ID fields."""
        output = [{"score": 0.9}, {"score": 0.8}]

        result = await evaluator.evaluate("test query", output)

        assert result.score == 0.0  # No IDs found
        assert result.metadata["unique_videos"] == 0


class TestTemporalCoverageEvaluator:
    """Test TemporalCoverageEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return TemporalCoverageEvaluator()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_no_results(self, evaluator):
        """Test evaluation with no results."""
        result = await evaluator.evaluate("test query", [])

        assert result.score == 0.0
        assert result.label == "no_results"
        assert "No results to evaluate" in result.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_no_temporal_info(self, evaluator):
        """Test evaluation with no temporal information."""
        output = [{"score": 0.9}, {"score": 0.8}]

        result = await evaluator.evaluate("test query", output)

        assert result.score == 0.0
        assert result.label == "no_temporal_info"
        assert "No temporal information" in result.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_good_coverage(self, evaluator):
        """Test evaluation with good temporal coverage."""
        output = [
            {"temporal_info": {"start_time": 0, "end_time": 10}},
            {"temporal_info": {"start_time": 10, "end_time": 20}},
            {"temporal_info": {"start_time": 20, "end_time": 30}},
            {"temporal_info": {"start_time": 30, "end_time": 40}},
            {"temporal_info": {"start_time": 40, "end_time": 50}},
            {"temporal_info": {"start_time": 50, "end_time": 60}},
            {"temporal_info": {"start_time": 60, "end_time": 70}},
            {"temporal_info": {"start_time": 70, "end_time": 80}},
        ]

        result = await evaluator.evaluate("test query", output)

        assert result.score >= 0.7
        assert result.label == "good_coverage"
        assert result.metadata["unique_segments"] == 8
        assert result.metadata["total_duration"] == 80.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_moderate_coverage(self, evaluator):
        """Test evaluation with moderate temporal coverage."""
        output = [
            {"temporal_info": {"start_time": 0, "end_time": 10}},
            {"temporal_info": {"start_time": 10, "end_time": 20}},
            {"temporal_info": {"start_time": 20, "end_time": 30}},
        ]

        result = await evaluator.evaluate("test query", output)

        assert 0.3 <= result.score < 0.7
        assert result.label == "moderate_coverage"
        assert result.metadata["unique_segments"] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_poor_coverage(self, evaluator):
        """Test evaluation with poor temporal coverage."""
        output = [
            {"temporal_info": {"start_time": 0, "end_time": 10}},
            {"temporal_info": {"start_time": 5, "end_time": 15}},  # Duplicate segment
        ]

        result = await evaluator.evaluate("test query", output)

        assert result.score < 0.3
        assert result.label == "poor_coverage"
        assert result.metadata["unique_segments"] == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_start_time_only(self, evaluator):
        """Test evaluation with only start_time provided."""
        output = [
            {"temporal_info": {"start_time": 0}},
            {"temporal_info": {"start_time": 10}},
        ]

        result = await evaluator.evaluate("test query", output)

        # Should use start_time as both start and end
        assert result.score > 0
        assert result.metadata["unique_segments"] == 2
        assert result.metadata["total_duration"] == 0.0  # No duration


class TestLLMRelevanceEvaluator:
    """Test LLMRelevanceEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return LLMRelevanceEvaluator(model_name="gpt-4")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_placeholder(self, evaluator):
        """Test placeholder LLM evaluation."""
        output = [{"score": 0.9}, {"score": 0.8}]

        result = await evaluator.evaluate("test query", output)

        assert result.score == 0.75
        assert result.label == "llm_evaluated"
        assert "LLM evaluation placeholder" in result.explanation
        assert result.metadata["model"] == "gpt-4"
        assert result.metadata["evaluation_type"] == "relevance"

    @pytest.mark.unit
    def test_init_default_model(self):
        """Test initialization with default model."""
        evaluator = LLMRelevanceEvaluator()
        assert evaluator.model_name == "gpt-4"

    @pytest.mark.unit
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        evaluator = LLMRelevanceEvaluator(model_name="custom-model")
        assert evaluator.model_name == "custom-model"


class TestCompositeEvaluator:
    """Test CompositeEvaluator."""

    @pytest.fixture
    def mock_evaluators(self):
        """Create mock evaluators for testing."""
        evaluator1 = AsyncMock()
        evaluator1.__class__.__name__ = "MockEvaluator1"
        evaluator1.evaluate.return_value = EvaluationResult(
            score=0.8, label="good", explanation="test1"
        )

        evaluator2 = AsyncMock()
        evaluator2.__class__.__name__ = "MockEvaluator2"
        evaluator2.evaluate.return_value = EvaluationResult(
            score=0.6, label="ok", explanation="test2"
        )

        return [evaluator1, evaluator2]

    @pytest.mark.unit
    def test_init_default_weights(self, mock_evaluators):
        """Test initialization with default weights."""
        evaluator = CompositeEvaluator(mock_evaluators)

        assert len(evaluator.evaluators) == 2
        assert evaluator.weights == [0.5, 0.5]  # Normalized

    @pytest.mark.unit
    def test_init_custom_weights(self, mock_evaluators):
        """Test initialization with custom weights."""
        evaluator = CompositeEvaluator(mock_evaluators, weights=[3.0, 1.0])

        assert evaluator.weights == [0.75, 0.25]  # Normalized

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_composite(self, mock_evaluators):
        """Test composite evaluation."""
        evaluator = CompositeEvaluator(mock_evaluators, weights=[0.6, 0.4])

        output = [{"score": 0.9}]
        result = await evaluator.evaluate("test query", output)

        # Weighted score: 0.8 * 0.6 + 0.6 * 0.4 = 0.48 + 0.24 = 0.72
        assert result.score == 0.72
        assert result.label == "composite_2_evaluators"
        assert "MockEvaluator1: 0.800 (good)" in result.explanation
        assert "MockEvaluator2: 0.600 (ok)" in result.explanation

        assert "MockEvaluator1" in result.metadata["component_scores"]
        assert "MockEvaluator2" in result.metadata["component_scores"]
        assert result.metadata["component_scores"]["MockEvaluator1"] == 0.8
        assert result.metadata["component_scores"]["MockEvaluator2"] == 0.6

        assert result.metadata["component_labels"]["MockEvaluator1"] == "good"
        assert result.metadata["component_labels"]["MockEvaluator2"] == "ok"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_calls_all_evaluators(self, mock_evaluators):
        """Test that composite evaluator calls all sub-evaluators."""
        evaluator = CompositeEvaluator(mock_evaluators)

        output = [{"score": 0.9}]
        await evaluator.evaluate("test query", output)

        for mock_eval in mock_evaluators:
            mock_eval.evaluate.assert_called_once_with("test query", output)


class TestReferenceFreeFunctions:
    """Test reference-free evaluator utility functions."""

    @pytest.mark.unit
    def test_create_reference_free_evaluators(self):
        """Test creating reference-free evaluators."""
        evaluators = create_reference_free_evaluators()

        assert len(evaluators) == 5
        assert "relevance" in evaluators
        assert "diversity" in evaluators
        assert "temporal_coverage" in evaluators
        assert "llm_relevance" in evaluators
        assert "composite" in evaluators

        assert isinstance(evaluators["relevance"], QueryResultRelevanceEvaluator)
        assert isinstance(evaluators["diversity"], ResultDiversityEvaluator)
        assert isinstance(evaluators["temporal_coverage"], TemporalCoverageEvaluator)
        assert isinstance(evaluators["llm_relevance"], LLMRelevanceEvaluator)
        assert isinstance(evaluators["composite"], CompositeEvaluator)

        # Check composite evaluator has 3 component evaluators
        composite = evaluators["composite"]
        assert len(composite.evaluators) == 3
        assert len(composite.weights) == 3
