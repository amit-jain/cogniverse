"""
Unit tests for evaluation metrics.
"""

import pytest
from cogniverse_core.evaluation.metrics.custom import (
    calculate_map,
    calculate_metrics_suite,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


class TestReferenceBasedMetrics:
    """Test reference-based metrics."""

    @pytest.mark.unit
    def test_calculate_mrr(self):
        """Test Mean Reciprocal Rank calculation."""
        # First relevant item at position 1 (index 0)
        retrieved = ["video1", "video2", "video3"]
        relevant = ["video1", "video4"]
        assert calculate_mrr(retrieved, relevant) == 1.0

        # First relevant item at position 2 (index 1)
        retrieved = ["video2", "video1", "video3"]
        relevant = ["video1", "video4"]
        assert calculate_mrr(retrieved, relevant) == 0.5

        # No relevant items
        retrieved = ["video2", "video3", "video5"]
        relevant = ["video1", "video4"]
        assert calculate_mrr(retrieved, relevant) == 0.0

    @pytest.mark.unit
    def test_calculate_mrr_empty(self):
        """Test MRR with empty inputs."""
        assert calculate_mrr([], ["video1"]) == 0.0
        assert calculate_mrr(["video1"], []) == 0.0
        assert calculate_mrr([], []) == 0.0

    @pytest.mark.unit
    def test_calculate_recall_at_k(self):
        """Test Recall@K calculation."""
        retrieved = ["video1", "video2", "video3", "video4", "video5"]
        relevant = ["video1", "video3", "video6"]

        # Recall@1: 1 out of 3 relevant items found
        assert calculate_recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)

        # Recall@3: 2 out of 3 relevant items found
        assert calculate_recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

        # Recall@5: 2 out of 3 relevant items found (video6 not retrieved)
        assert calculate_recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)

    @pytest.mark.unit
    def test_calculate_precision_at_k(self):
        """Test Precision@K calculation."""
        retrieved = ["video1", "video2", "video3", "video4", "video5"]
        relevant = ["video1", "video3", "video6"]

        # Precision@1: 1 out of 1 retrieved is relevant
        assert calculate_precision_at_k(retrieved, relevant, k=1) == 1.0

        # Precision@3: 2 out of 3 retrieved are relevant
        assert calculate_precision_at_k(retrieved, relevant, k=3) == pytest.approx(
            2 / 3
        )

        # Precision@5: 2 out of 5 retrieved are relevant
        assert calculate_precision_at_k(retrieved, relevant, k=5) == pytest.approx(
            2 / 5
        )

    @pytest.mark.unit
    def test_calculate_precision_at_k_no_relevant(self):
        """Test Precision@K with no relevant items."""
        retrieved = ["video1", "video2"]
        relevant = ["video3", "video4"]

        assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.0

    @pytest.mark.unit
    def test_calculate_ndcg(self):
        """Test NDCG calculation."""
        retrieved = ["video1", "video2", "video3", "video4"]
        relevant = ["video1", "video3"]

        ndcg = calculate_ndcg(retrieved, relevant, k=4)

        # NDCG should be between 0 and 1
        assert 0 <= ndcg <= 1

        # Perfect ranking should give NDCG = 1
        perfect_retrieved = ["video1", "video3", "video2", "video4"]
        perfect_ndcg = calculate_ndcg(perfect_retrieved, relevant, k=2)
        assert perfect_ndcg == 1.0

    @pytest.mark.unit
    def test_calculate_ndcg_no_relevant(self):
        """Test NDCG with no relevant items."""
        retrieved = ["video1", "video2"]
        relevant = ["video3", "video4"]

        ndcg = calculate_ndcg(retrieved, relevant, k=2)
        assert ndcg == 0.0

    @pytest.mark.unit
    def test_calculate_map(self):
        """Test Mean Average Precision calculation."""
        # MAP expects lists of lists
        retrieved_list = [["video1", "video2", "video3", "video4", "video5"]]
        relevant_list = [["video1", "video3", "video5"]]

        map_score = calculate_map(retrieved_list, relevant_list)

        # MAP should be between 0 and 1
        assert 0 <= map_score <= 1

    @pytest.mark.unit
    def test_calculate_metrics_suite(self):
        """Test complete metrics suite calculation."""
        results = ["video1", "video2", "video3", "video4", "video5"]
        expected = ["video1", "video3", "video6"]

        metrics = calculate_metrics_suite(results, expected)

        assert "mrr" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "precision@1" in metrics
        assert "precision@5" in metrics

        # Check metric values are in valid range
        for _key, value in metrics.items():
            assert 0 <= value <= 1


class TestReferenceFreeMetrics:
    """Test reference-free metrics."""

    @pytest.mark.unit
    def test_placeholder_for_future_reference_free_metrics(self):
        """Placeholder test for future reference-free metrics implementation."""
        # These will be implemented when we add RAGAS-based reference-free evaluation
        # For now, we're using RAGAS through the scorers in src/evaluation/core/scorers.py
        assert True
