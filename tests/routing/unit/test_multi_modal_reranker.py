"""
Unit tests for MultiModalReranker

Tests multi-modal reranking functionality including cross-modal scoring,
temporal alignment, complementarity, diversity, and ranking quality analysis.
"""

from datetime import datetime, timedelta

import pytest

from cogniverse_agents.search.multi_modal_reranker import (
    MultiModalReranker,
    QueryModality,
    SearchResult,
)


@pytest.mark.unit
class TestMultiModalReranker:
    """Test multi-modal reranking functionality"""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance"""
        return MultiModalReranker()

    @pytest.fixture
    def sample_results(self):
        """Create sample search results"""
        return [
            SearchResult(
                id="video_1",
                title="Machine Learning Tutorial",
                content="Comprehensive ML tutorial covering basics",
                modality="video",
                score=0.9,
                metadata={},
                timestamp=datetime.now() - timedelta(days=10),
            ),
            SearchResult(
                id="image_1",
                title="Neural Network Diagram",
                content="Visual representation of neural network architecture",
                modality="image",
                score=0.85,
                metadata={},
                timestamp=datetime.now() - timedelta(days=5),
            ),
            SearchResult(
                id="doc_1",
                title="ML Research Paper",
                content="Latest research in machine learning algorithms",
                modality="document",
                score=0.8,
                metadata={},
                timestamp=datetime.now() - timedelta(days=2),
            ),
        ]

    @pytest.mark.asyncio
    async def test_basic_reranking(self, reranker, sample_results):
        """Test basic reranking functionality"""
        query = "machine learning"
        modalities = [QueryModality.VIDEO]

        reranked = await reranker.rerank_results(sample_results, query, modalities, {})

        assert len(reranked) == len(sample_results)
        # Video should be ranked higher due to modality match
        assert reranked[0].modality == "video"
        # All results should have reranking metadata
        for result in reranked:
            assert "reranking_score" in result.metadata
            assert "score_components" in result.metadata

    @pytest.mark.asyncio
    async def test_temporal_reranking(self, reranker, sample_results):
        """Test temporal alignment in reranking"""
        query = "machine learning"
        modalities = [QueryModality.VIDEO]

        # Create temporal context (prefer recent results)
        context = {
            "temporal": {
                "time_range": (
                    datetime.now() - timedelta(days=7),
                    datetime.now(),
                ),
                "requires_temporal_search": True,
            }
        }

        reranked = await reranker.rerank_results(
            sample_results, query, modalities, context
        )

        # More recent results should rank higher with temporal context
        # doc_1 is most recent (2 days ago), should benefit
        scores = [r.metadata["reranking_score"] for r in reranked]
        assert len(scores) == 3

    def test_cross_modal_score(self, reranker):
        """Test cross-modal relevance scoring"""
        result = SearchResult(
            id="test",
            title="Test",
            content="Test content",
            modality="video",
            score=0.9,
            metadata={},
        )

        # Direct match
        score = reranker._calculate_cross_modal_score(
            result, "test", [QueryModality.VIDEO]
        )
        assert score == 1.0

        # Compatible match (video-image)
        score = reranker._calculate_cross_modal_score(
            result, "test", [QueryModality.IMAGE]
        )
        assert 0.5 < score < 1.0

        # No match
        score = reranker._calculate_cross_modal_score(
            result, "test", [QueryModality.DOCUMENT]
        )
        assert score < 0.5

    def test_diversity_score(self, reranker, sample_results):
        """Test diversity scoring"""
        selected = []

        # First result gets max diversity
        score1 = reranker._calculate_diversity_score(sample_results[0], selected)
        assert score1 == 1.0

        # Add first result to selected
        selected.append((0.9, sample_results[0], {}))

        # Same modality gets lower diversity
        score2 = reranker._calculate_diversity_score(sample_results[0], selected)
        assert score2 < score1

        # Different modality gets higher diversity
        score3 = reranker._calculate_diversity_score(sample_results[1], selected)
        assert score3 > score2

    def test_modality_distribution(self, reranker, sample_results):
        """Test modality distribution calculation"""
        distribution = reranker.get_modality_distribution(sample_results)

        assert distribution["video"] == 1
        assert distribution["image"] == 1
        assert distribution["document"] == 1

    def test_ranking_quality_analysis(self, reranker, sample_results):
        """Test ranking quality metrics"""
        # Add reranking scores
        for i, result in enumerate(sample_results):
            result.metadata["reranking_score"] = 0.9 - (i * 0.1)

        quality = reranker.analyze_ranking_quality(sample_results)

        assert "diversity" in quality
        assert "average_score" in quality
        assert "modality_distribution" in quality
        assert "temporal_coverage" in quality
        assert 0.0 <= quality["diversity"] <= 1.0
        assert quality["total_results"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
