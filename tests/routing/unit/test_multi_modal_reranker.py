"""
Unit tests for MultiModalReranker

Tests multi-modal reranking functionality including cross-modal scoring,
temporal alignment, complementarity, diversity, and ranking quality analysis.
"""

from datetime import datetime, timedelta

import pytest

from cogniverse_agents.search.multi_modal_reranker import MultiModalReranker
from cogniverse_agents.search.types import QueryModality, RerankerSearchResult


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
            RerankerSearchResult(
                id="video_1",
                title="Machine Learning Tutorial",
                content="Comprehensive ML tutorial covering basics",
                modality="video",
                score=0.9,
                metadata={},
                timestamp=datetime.now() - timedelta(days=10),
            ),
            RerankerSearchResult(
                id="image_1",
                title="Neural Network Diagram",
                content="Visual representation of neural network architecture",
                modality="image",
                score=0.85,
                metadata={},
                timestamp=datetime.now() - timedelta(days=5),
            ),
            RerankerSearchResult(
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

    def test_temporal_score_rewards_in_range_and_centered(self, reranker):
        """_calculate_temporal_score: in-range > edge > out-of-range, with a
        perfectly centered result scoring 1.0."""
        now = datetime.now()
        ctx = {"temporal": {"time_range": (now - timedelta(days=10), now)}}

        def _at(ts):
            return RerankerSearchResult(
                id="x",
                title="t",
                content="c",
                modality="video",
                score=1.0,
                metadata={},
                timestamp=ts,
            )

        centered = reranker._calculate_temporal_score(_at(now - timedelta(days=5)), ctx)
        edge = reranker._calculate_temporal_score(_at(now - timedelta(days=10)), ctx)
        outside = reranker._calculate_temporal_score(
            _at(now - timedelta(days=100)), ctx
        )

        assert centered == pytest.approx(1.0)
        assert edge == pytest.approx(0.7)
        assert centered > edge > outside

    def test_temporal_score_handles_naive_aware_timezone_mix(self, reranker):
        """A naive/aware datetime mix must not raise TypeError and abort the
        rerank loop; naive datetimes are read as UTC."""
        from datetime import timezone

        def _res(ts):
            return RerankerSearchResult(
                id="x",
                title="t",
                content="c",
                modality="video",
                score=1.0,
                metadata={},
                timestamp=ts,
            )

        naive_range = {
            "temporal": {"time_range": (datetime(2026, 5, 1), datetime(2026, 7, 1))}
        }
        aware_mid = datetime(2026, 6, 1, tzinfo=timezone.utc)
        # In-range, near-centered → high score; the point is no TypeError.
        aware_vs_naive = reranker._calculate_temporal_score(
            _res(aware_mid), naive_range
        )
        assert 0.9 < aware_vs_naive <= 1.0

        aware_range = {
            "temporal": {
                "time_range": (
                    datetime(2026, 5, 1, tzinfo=timezone.utc),
                    datetime(2026, 7, 1, tzinfo=timezone.utc),
                )
            }
        }
        naive_vs_aware = reranker._calculate_temporal_score(
            _res(datetime(2026, 6, 1)), aware_range
        )
        assert 0.9 < naive_vs_aware <= 1.0

    def test_temporal_score_neutral_without_context(self, reranker):
        now = datetime.now()
        r = RerankerSearchResult(
            id="x",
            title="t",
            content="c",
            modality="video",
            score=1.0,
            metadata={},
            timestamp=now,
        )
        assert reranker._calculate_temporal_score(r, {}) == 0.5

    def test_cross_modal_score(self, reranker):
        """Test cross-modal relevance scoring"""
        result = RerankerSearchResult(
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


class TestRankingDiversity:
    """analyze_ranking_quality diversity is normalized Shannon entropy."""

    @staticmethod
    def _r(modality: str) -> RerankerSearchResult:
        return RerankerSearchResult(
            id="x",
            title="t",
            content="c",
            modality=modality,
            score=1.0,
            metadata={"reranking_score": 1.0},
        )

    def test_two_equal_modalities_max_diversity(self):
        rr = object.__new__(MultiModalReranker)
        q = rr.analyze_ranking_quality([self._r("video"), self._r("text")])
        assert q["diversity"] == pytest.approx(1.0)

    def test_single_modality_zero_diversity(self):
        rr = object.__new__(MultiModalReranker)
        q = rr.analyze_ranking_quality([self._r("video"), self._r("video")])
        assert q["diversity"] == 0.0
