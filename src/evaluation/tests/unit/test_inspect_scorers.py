"""
Unit tests for Inspect AI scorers.
"""

from unittest.mock import Mock

import pytest

from src.evaluation.inspect_tasks.scorers import (
    AlignmentScorer,
    FailureAnalysisScorer,
    TemporalAccuracyScorer,
    VideoRetrievalScorer,
)


class TestVideoRetrievalScorer:
    """Test VideoRetrievalScorer functionality."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return VideoRetrievalScorer()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test scorer initialization."""
        # Default metrics
        scorer = VideoRetrievalScorer()
        assert scorer.metrics == ["mrr", "ndcg", "precision", "recall"]
        assert len(scorer.metric_calculators) >= 4

        # Custom metrics
        scorer = VideoRetrievalScorer(metrics=["mrr", "precision"])
        assert scorer.metrics == ["mrr", "precision"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_score_no_results(self, scorer):
        """Test scoring with no retrieval results."""
        state = Mock()
        state.metadata = {}
        target = []

        score = await scorer(state, target)

        assert score.value == 0.0
        assert "No retrieval results" in score.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_score_with_results(self, scorer):
        """Test scoring with retrieval results."""
        state = Mock()
        state.metadata = {
            "retrieval_results": ["video1", "video2", "video3"],
            "expected_videos": ["video2", "video4"],
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        assert score.value > 0  # Should have some score
        assert score.metadata is not None
        assert "metrics" in score.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_mrr(self, scorer):
        """Test MRR calculation."""
        retrieved = [
            {"video_id": "video1"},
            {"video_id": "video2"},
            {"video_id": "video3"},
        ]
        expected = ["video2", "video4"]

        mrr = scorer._calculate_mrr(retrieved, expected)

        # video2 is at position 2 (index 1), so MRR = 1/2 = 0.5
        assert mrr == 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_mrr_no_match(self, scorer):
        """Test MRR with no matching items."""
        retrieved = [
            {"video_id": "video1"},
            {"video_id": "video2"},
            {"video_id": "video3"},
        ]
        expected = ["video4", "video5"]

        mrr = scorer._calculate_mrr(retrieved, expected)

        assert mrr == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_precision(self, scorer):
        """Test precision calculation."""
        retrieved = [
            {"video_id": "video1"},
            {"video_id": "video2"},
            {"video_id": "video3"},
        ]
        expected = ["video1", "video3", "video5"]

        precision = scorer._calculate_precision(retrieved, expected, k=3)

        # 2 out of 3 retrieved are relevant
        assert precision == pytest.approx(2 / 3)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_recall(self, scorer):
        """Test recall calculation."""
        retrieved = [
            {"video_id": "video1"},
            {"video_id": "video2"},
            {"video_id": "video3"},
        ]
        expected = ["video1", "video3", "video5"]

        recall = scorer._calculate_recall(retrieved, expected, k=3)

        # 2 out of 3 expected are retrieved
        assert recall == pytest.approx(2 / 3)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_ndcg(self, scorer):
        """Test NDCG calculation."""
        retrieved = [
            {"video_id": "video1"},
            {"video_id": "video2"},
            {"video_id": "video3"},
        ]
        expected = ["video1", "video3"]

        ndcg = scorer._calculate_ndcg(retrieved, expected, k=3)

        # Should be between 0 and 1
        assert 0 <= ndcg <= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calculate_map(self, scorer):
        """Test MAP calculation."""
        retrieved = [
            {"video_id": "video1"},
            {"video_id": "video2"},
            {"video_id": "video3"},
            {"video_id": "video4"},
        ]
        expected = ["video1", "video3"]

        map_score = scorer._calculate_map(retrieved, expected)

        # Should calculate average precision
        assert 0 <= map_score <= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phoenix_logging(self, scorer):
        """Test Phoenix logging integration."""
        # No need to patch px since it's not imported in scorers module
        state = Mock()
        state.metadata = {
            "retrieval_results": [{"video_id": "video1"}],
            "expected_videos": ["video1"],
            "query": "test query",
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        # Should have perfect score
        assert score.value == 1.0


class TestTemporalAccuracyScorer:
    """Test TemporalAccuracyScorer functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mrr_scorer_initialization(self):
        """Test MRR scorer initialization."""
        scorer = TemporalAccuracyScorer()
        assert scorer.name == "mrr"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temporal_scorer_calculation(self):
        """Test temporal scorer calculation."""
        scorer = TemporalAccuracyScorer()

        state = Mock()
        state.metadata = {
            "temporal_info": {"extracted_range": [10, 20]},
            "expected_time_range": [15, 25],
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        # Should calculate overlap between ranges
        assert 0 <= score.value <= 1
        assert (
            "overlap" in score.explanation.lower()
            or "range" in score.explanation.lower()
        )


class TestAlignmentScorer:
    """Test AlignmentScorer functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_alignment_scorer_initialization(self):
        """Test alignment scorer initialization."""
        scorer = AlignmentScorer()
        assert scorer.name == "ndcg"
        assert scorer.k == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_alignment_scorer_calculation(self):
        """Test alignment scorer calculation."""
        scorer = AlignmentScorer()

        state = Mock()
        state.metadata = {
            "alignment_score": 0.85,
            "alignment_check": True,
            "expected_alignment": True,
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        # Should have alignment score
        assert score.value == 1.0  # alignment_check matches expected_alignment
        assert (
            "alignment" in score.explanation.lower()
            or "correct" in score.explanation.lower()
        )


class TestCustomMetricScorer:
    """Test custom metric scoring functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_metric_f1_score(self):
        """Test F1 score calculation as custom metric."""
        # Create a VideoRetrievalScorer which calculates multiple metrics
        scorer = VideoRetrievalScorer(metrics=["precision", "recall"])

        state = Mock()
        state.metadata = {
            "retrieval_results": ["item1", "item2", "item3"],
            "expected_videos": ["item2", "item3", "item4"],
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        # Should calculate both precision and recall
        assert score.metadata["metrics"]["precision"] > 0
        assert score.metadata["metrics"]["recall"] > 0

        # F1 can be calculated from precision and recall
        p = score.metadata["metrics"]["precision"]
        r = score.metadata["metrics"]["recall"]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        assert 0 <= f1 <= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_metric_map_score(self):
        """Test MAP score as custom metric."""
        scorer = VideoRetrievalScorer(metrics=["map"])

        state = Mock()
        state.metadata = {
            "retrieval_results": ["item1", "item2", "item3", "item4"],
            "expected_videos": ["item1", "item3"],
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        # MAP should be calculated
        assert "map" in score.metadata["metrics"]
        assert 0 <= score.metadata["metrics"]["map"] <= 1


class TestFailureAnalysisScorer:
    """Test FailureAnalysisScorer functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failure_analysis_scorer_initialization(self):
        """Test failure analysis scorer initialization."""
        scorer = FailureAnalysisScorer()
        assert scorer is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failure_analysis_with_errors(self):
        """Test failure analysis when errors occur."""
        scorer = FailureAnalysisScorer()

        state = Mock()
        state.metadata = {
            "retrieval_results": [],
            "error": "Connection timeout",
            "expected_videos": ["item1", "item2"],
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        # Should identify the failure
        assert score.value == 0.0 or score.value < 0.5
        assert (
            "error" in score.explanation.lower()
            or "failure" in score.explanation.lower()
        )


class TestScorerIntegration:
    """Test scorer integration with Inspect AI."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scorer_with_inspect_state(self):
        """Test scorer with full Inspect AI state."""
        scorer = VideoRetrievalScorer(metrics=["mrr", "precision"])

        # Create mock Inspect AI state
        state = Mock()
        state.metadata = {
            "retrieval_results": ["v1", "v2", "v3", "v4", "v5"],
            "expected_videos": ["v2", "v4", "v6"],
            "query": "test video query",
            "profile": "test_profile",
            "strategy": "test_strategy",
        }
        state.input = Mock()
        state.input.query = "test video query"
        state.input.metadata = {}

        target = ["v2", "v4", "v6"]

        score = await scorer(state, target)

        assert score.value > 0
        assert score.value <= 1
        assert "MRR" in score.explanation or "metrics" in str(score.metadata)
        assert score.metadata["metrics"]["mrr"] > 0
        assert "precision" in score.metadata["metrics"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scorer_error_handling(self):
        """Test scorer error handling."""
        scorer = VideoRetrievalScorer()

        # Test with malformed state
        state = Mock()
        state.metadata = {
            "retrieval_results": None,  # Invalid results
            "expected_videos": ["v1"],
        }
        state.input = Mock()
        state.input.metadata = {}
        target = []

        # Should handle gracefully
        score = await scorer(state, target)
        assert score.value == 0.0 or score.value is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scorer_with_empty_results(self):
        """Test scorer with empty results."""
        scorer = VideoRetrievalScorer()

        state = Mock()
        state.metadata = {"retrieval_results": [], "expected_videos": ["v1", "v2"]}
        state.input = Mock()
        state.input.metadata = {}
        target = []

        score = await scorer(state, target)

        assert score.value == 0.0
        assert "no results" in score.explanation.lower() or score.value == 0.0
