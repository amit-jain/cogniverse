"""``_build_query`` must bind every declared float query input.

Besides ``qt``/``qtb``/``q``, the audio ``acoustic_similarity`` and
``hybrid_acoustic_bm25`` profiles declare an ``acoustic_query`` float
tensor. If the binding loop skips it, the nearestNeighbor operator
references an unbound tensor and ranking collapses.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_vespa.search_backend import VespaSearchBackend


@pytest.fixture
def backend() -> VespaSearchBackend:
    # _build_query only uses _build_filter_conditions (empty for {} filters);
    # no Vespa connection needed.
    return object.__new__(VespaSearchBackend)


def test_acoustic_query_float_input_is_bound(backend: VespaSearchBackend) -> None:
    rank_config = {
        "use_nearestneighbor": True,
        "nearestneighbor_field": "acoustic_embedding",
        "nearestneighbor_tensor": "acoustic_query",
        "inputs": {"acoustic_query": "tensor<float>(v[512])"},
    }
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0

    params = backend._build_query(
        query_text="dog barking",
        query_embeddings=vec,
        rank_config=rank_config,
        ranking_profile="acoustic_similarity",
        schema_name="audio_content",
        limit=10,
        filters={},
        correlation_id="t",
    )

    assert params["input.query(acoustic_query)"] == vec.tolist()
    assert "nearestNeighbor(acoustic_embedding, acoustic_query)" in params["yql"]
    assert params["ranking"] == "acoustic_similarity"


def test_hybrid_acoustic_binds_acoustic_query_and_text(
    backend: VespaSearchBackend,
) -> None:
    rank_config = {
        "use_nearestneighbor": True,
        "needs_text_query": True,
        "nearestneighbor_field": "acoustic_embedding",
        "nearestneighbor_tensor": "acoustic_query",
        "inputs": {"acoustic_query": "tensor<float>(v[512])"},
    }
    vec = np.zeros(512, dtype=np.float32)
    vec[3] = 1.0

    params = backend._build_query(
        query_text="ocean waves",
        query_embeddings=vec,
        rank_config=rank_config,
        ranking_profile="hybrid_acoustic_bm25",
        schema_name="audio_content",
        limit=5,
        filters={},
        correlation_id="t",
    )

    assert params["input.query(acoustic_query)"] == vec.tolist()
    assert params["userQuery"] == "ocean waves"
    assert "nearestNeighbor(acoustic_embedding, acoustic_query)" in params["yql"]
