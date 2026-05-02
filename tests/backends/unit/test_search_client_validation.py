"""Pure-logic unit tests for VespaVideoSearchClient input validation
and RankingStrategy recommendation — covers what the dormant
tests/test_search_client.py used to exercise as part of its end-to-end
script. The integration coverage moved to
tests/runtime/integration/test_ranking_strategies_real.py; the unit-
level pieces live here.
"""

from __future__ import annotations

import pytest

from cogniverse_vespa.vespa_search_client import (
    RankingStrategy,
    VespaVideoSearchClient,
)


class TestRecommendStrategy:
    """RankingStrategy.recommend_strategy maps query characteristics to enum."""

    def test_text_only_query_recommends_bm25(self):
        assert (
            RankingStrategy.recommend_strategy(
                has_visual_component=False, has_text_component=True
            )
            is RankingStrategy.BM25_ONLY
        )

    def test_visual_only_query_default_recommends_float_float(self):
        assert (
            RankingStrategy.recommend_strategy(
                has_visual_component=True, has_text_component=False
            )
            is RankingStrategy.FLOAT_FLOAT
        )

    def test_visual_only_query_speed_priority_recommends_binary(self):
        assert (
            RankingStrategy.recommend_strategy(
                has_visual_component=True,
                has_text_component=False,
                speed_priority=True,
            )
            is RankingStrategy.BINARY_BINARY
        )

    def test_hybrid_query_default_recommends_hybrid_float_bm25(self):
        assert (
            RankingStrategy.recommend_strategy(
                has_visual_component=True, has_text_component=True
            )
            is RankingStrategy.HYBRID_FLOAT_BM25
        )

    def test_hybrid_query_speed_priority_recommends_hybrid_binary_bm25(self):
        assert (
            RankingStrategy.recommend_strategy(
                has_visual_component=True,
                has_text_component=True,
                speed_priority=True,
            )
            is RankingStrategy.HYBRID_BINARY_BM25
        )

    def test_no_components_falls_back_to_bm25(self):
        assert (
            RankingStrategy.recommend_strategy(
                has_visual_component=False, has_text_component=False
            )
            is RankingStrategy.BM25_ONLY
        )


class TestValidateStrategyInputs:
    """VespaVideoSearchClient.validate_strategy_inputs catches missing
    text queries and missing embeddings before they hit Vespa."""

    @pytest.fixture
    def client(self):
        # Bypass __init__ entirely — validate_strategy_inputs uses no
        # instance state and we don't want to require a live Vespa to
        # construct the client just to test pure validation logic.
        return VespaVideoSearchClient.__new__(VespaVideoSearchClient)

    def test_unknown_strategy_returns_error(self, client):
        errors = client.validate_strategy_inputs("nonexistent_strategy", "q", None)
        assert errors and "Unknown strategy" in errors[0]

    @pytest.mark.parametrize(
        "strategy",
        [
            RankingStrategy.BM25_ONLY.value,
            RankingStrategy.BM25_NO_DESCRIPTION.value,
            RankingStrategy.HYBRID_FLOAT_BM25.value,
            RankingStrategy.HYBRID_BINARY_BM25.value,
            RankingStrategy.HYBRID_BM25_BINARY.value,
            RankingStrategy.HYBRID_BM25_FLOAT.value,
        ],
    )
    def test_text_required_strategies_reject_empty_query(self, client, strategy):
        import numpy as np

        embeddings = np.zeros((10, 128), dtype=np.float32)
        errors = client.validate_strategy_inputs(strategy, "", embeddings)
        assert errors, f"{strategy} must reject empty query"

    @pytest.mark.parametrize(
        "strategy",
        [
            RankingStrategy.FLOAT_FLOAT.value,
            RankingStrategy.BINARY_BINARY.value,
            RankingStrategy.FLOAT_BINARY.value,
            RankingStrategy.PHASED.value,
            RankingStrategy.HYBRID_FLOAT_BM25.value,
            RankingStrategy.HYBRID_BINARY_BM25.value,
        ],
    )
    def test_visual_strategies_reject_missing_embeddings(self, client, strategy):
        errors = client.validate_strategy_inputs(strategy, "q", None)
        assert errors, f"{strategy} must reject missing embeddings"

    def test_text_only_strategy_accepts_no_embeddings(self, client):
        errors = client.validate_strategy_inputs(
            RankingStrategy.BM25_ONLY.value, "ocean waves", None
        )
        assert errors == []

    def test_visual_strategy_with_embeddings_passes(self, client):
        import numpy as np

        embeddings = np.zeros((10, 128), dtype=np.float32)
        errors = client.validate_strategy_inputs(
            RankingStrategy.FLOAT_FLOAT.value, "", embeddings
        )
        assert errors == []
