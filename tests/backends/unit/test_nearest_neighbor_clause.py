"""The nearestNeighbor YQL term every served vector search uses."""

from __future__ import annotations

import pytest

from cogniverse_vespa._yql import (
    NEAREST_NEIGHBOR_EXPLORE_ADDITIONAL_HITS,
    nearest_neighbor,
)

pytestmark = pytest.mark.unit


def test_approximate_search_widens_the_hnsw_beam():
    assert NEAREST_NEIGHBOR_EXPLORE_ADDITIONAL_HITS == 200
    assert nearest_neighbor("embedding", "qt", 4) == (
        "{targetHits: 4, hnsw.exploreAdditionalHits: 200}nearestNeighbor(embedding, qt)"
    )


def test_exact_search_scans_without_a_beam():
    assert nearest_neighbor(
        "acoustic_embedding", "acoustic_query", 10, approximate=False
    ) == (
        "{targetHits: 10, approximate: false}"
        "nearestNeighbor(acoustic_embedding, acoustic_query)"
    )
