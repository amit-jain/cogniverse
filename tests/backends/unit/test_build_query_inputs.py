"""``_build_query`` must bind every declared float query input.

Besides ``qt``/``qtb``/``q``, the audio ``acoustic_similarity`` and
``hybrid_acoustic_bm25`` profiles declare an ``acoustic_query`` float
tensor. If the binding loop skips it, the nearestNeighbor operator
references an unbound tensor and ranking collapses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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


def test_generic_q_input_flattens_single_row_2d(
    backend: VespaSearchBackend,
) -> None:
    """A single-vector encoder returning (1, dim) bound to the generic ``q``
    input must emit a flat dim-length list, not a nested [[...]] the
    tensor<float>(x[dim]) input rejects."""
    rank_config = {"inputs": {"q": "tensor<float>(v[128])"}}
    vec = np.zeros((1, 128), dtype=np.float32)
    vec[0, 0] = 1.0

    params = backend._build_query(
        query_text="hello",
        query_embeddings=vec,
        rank_config=rank_config,
        ranking_profile="default",
        schema_name="video_colpali",
        limit=10,
        filters={},
        correlation_id="t",
    )

    bound = params["input.query(q)"]
    assert len(bound) == 128
    assert not isinstance(bound[0], list)
    assert bound == vec[0].tolist()


def test_strategy_timeout_forwarded_to_query(backend: VespaSearchBackend) -> None:
    """A per-strategy timeout must reach Vespa so a hung query can't drain the
    connection pool."""
    rank_config = {"needs_text_query": True, "timeout": 2.0}
    params = backend._build_query(
        query_text="cats",
        query_embeddings=None,
        rank_config=rank_config,
        ranking_profile="bm25_only",
        schema_name="video_frame",
        limit=10,
        filters={},
        correlation_id="t",
    )
    assert params["timeout"] == "2.0s"


def test_top_k_above_default_hit_cap_raises_native_limits(
    backend: VespaSearchBackend,
) -> None:
    """The default query profile caps hits at 400; every built query must
    raise maxHits/maxOffset to the requested limit or Vespa rejects any
    top_k > 400 as an illegal query."""
    rank_config = {"needs_text_query": True}
    params = backend._build_query(
        query_text="needle",
        query_embeddings=None,
        rank_config=rank_config,
        ranking_profile="bm25_only",
        schema_name="video_frame",
        limit=1000,
        filters={},
        correlation_id="t",
    )
    assert params["hits"] == 1000
    assert params["maxHits"] == 1000
    assert params["maxOffset"] == 1000


def test_no_timeout_key_when_strategy_omits_it(backend: VespaSearchBackend) -> None:
    rank_config = {"needs_text_query": True}
    params = backend._build_query(
        query_text="cats",
        query_embeddings=None,
        rank_config=rank_config,
        ranking_profile="bm25_only",
        schema_name="video_frame",
        limit=10,
        filters={},
        correlation_id="t",
    )
    assert "timeout" not in params


def _nearest_neighbor_query(
    backend: VespaSearchBackend,
    *,
    nearest_neighbor_approximate: bool = True,
) -> dict:
    return backend._build_query(
        query_text="Marie Curie discovered radium",
        query_embeddings=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        rank_config={
            "use_nearestneighbor": True,
            "nearestneighbor_field": "embedding",
            "nearestneighbor_tensor": "qt",
            "inputs": {"qt": "tensor<float>(v[3])"},
        },
        ranking_profile="semantic_search",
        schema_name="agent_memories_acme_acme",
        limit=1,
        filters={"user_id": "scientists"},
        correlation_id="memory-query-contract",
        nearest_neighbor_approximate=nearest_neighbor_approximate,
    )


def test_exact_nearest_neighbor_emits_canonical_vespa_annotation(
    backend: VespaSearchBackend,
) -> None:
    params = _nearest_neighbor_query(
        backend,
        nearest_neighbor_approximate=False,
    )

    assert (
        params["yql"] == "select * from agent_memories_acme_acme where "
        "{targetHits: 1, approximate: false}nearestNeighbor(embedding, qt) "
        'AND user_id contains "scientists"'
    )


def test_default_nearest_neighbor_remains_approximate(
    backend: VespaSearchBackend,
) -> None:
    params = _nearest_neighbor_query(backend)

    assert (
        params["yql"] == "select * from agent_memories_acme_acme where "
        "{targetHits: 1}nearestNeighbor(embedding, qt) "
        'AND user_id contains "scientists"'
    )
    assert "approximate: false" not in params["yql"]


def test_nearest_neighbor_approximate_rejects_non_boolean(
    backend: VespaSearchBackend,
) -> None:
    with pytest.raises(
        ValueError,
        match="nearest_neighbor_approximate must be a bool",
    ):
        _nearest_neighbor_query(
            backend,
            nearest_neighbor_approximate="false",
        )


def test_concurrent_exact_and_approximate_queries_do_not_share_mode(
    backend: VespaSearchBackend,
) -> None:
    modes = [False, True] * 20

    with ThreadPoolExecutor(max_workers=8) as executor:
        queries = list(
            executor.map(
                lambda mode: _nearest_neighbor_query(
                    backend,
                    nearest_neighbor_approximate=mode,
                )["yql"],
                modes,
            )
        )

    for mode, yql in zip(modes, queries):
        assert ("approximate: false" in yql) is (mode is False)


def _hybrid_query(
    backend: VespaSearchBackend,
    *,
    query_text: str,
    first_phase_embedding_field: str | None,
    filters: dict | None = None,
    text_candidates_only: bool = False,
) -> dict:
    return backend._build_query(
        query_text=query_text,
        query_embeddings=None,
        rank_config={
            "needs_text_query": True,
            "first_phase_embedding_field": first_phase_embedding_field,
            "text_candidates_only": text_candidates_only,
        },
        ranking_profile="hybrid_float_bm25",
        schema_name="video_frame_acme_acme",
        limit=10,
        filters=filters or {},
        correlation_id="t",
    )


def test_embedding_first_hybrid_matches_every_document(
    backend: VespaSearchBackend,
) -> None:
    """A hybrid whose first phase scores every document by its embedding must
    not narrow the match set to text matches; the text terms only rank."""
    params = _hybrid_query(
        backend,
        query_text="man shoveling snow",
        first_phase_embedding_field="embedding",
    )

    assert params["yql"] == (
        "select * from video_frame_acme_acme where "
        'rank(true, {grammar: "any"}userInput(@userQuery))'
    )
    assert params["userQuery"] == "man shoveling snow"


def test_embedding_first_hybrid_keeps_filters(backend: VespaSearchBackend) -> None:
    params = _hybrid_query(
        backend,
        query_text="man shoveling snow",
        first_phase_embedding_field="embedding",
        filters={"video_id": "v1"},
    )

    assert params["yql"] == (
        "select * from video_frame_acme_acme where "
        'rank(true, {grammar: "any"}userInput(@userQuery)) '
        'AND video_id contains "v1"'
    )


def test_embedding_first_hybrid_without_text_ranks_every_document(
    backend: VespaSearchBackend,
) -> None:
    params = _hybrid_query(
        backend, query_text="", first_phase_embedding_field="embedding"
    )

    assert params["yql"] == "select * from video_frame_acme_acme where true"
    assert "userQuery" not in params


def test_text_first_hybrid_matches_text(backend: VespaSearchBackend) -> None:
    params = _hybrid_query(
        backend, query_text="man shoveling snow", first_phase_embedding_field=None
    )

    assert params["yql"] == (
        "select * from video_frame_acme_acme where userInput(@userQuery)"
    )
    assert params["userQuery"] == "man shoveling snow"


def test_text_matches_only_hybrid_matches_text_though_it_scores_an_embedding(
    backend: VespaSearchBackend,
) -> None:
    """A hybrid that ranks text matches only matches the query text even
    though its first phase scores the embedding, with or without filters,
    and finds nothing without text."""
    params = _hybrid_query(
        backend,
        query_text="man shoveling snow",
        first_phase_embedding_field="embedding",
        text_candidates_only=True,
    )
    filtered = _hybrid_query(
        backend,
        query_text="man shoveling snow",
        first_phase_embedding_field="embedding",
        text_candidates_only=True,
        filters={"video_id": "v1"},
    )
    empty = _hybrid_query(
        backend,
        query_text="",
        first_phase_embedding_field="embedding",
        text_candidates_only=True,
    )

    assert params["yql"] == (
        "select * from video_frame_acme_acme where userInput(@userQuery)"
    )
    assert params["userQuery"] == "man shoveling snow"
    assert filtered["yql"] == (
        "select * from video_frame_acme_acme where userInput(@userQuery) "
        'AND video_id contains "v1"'
    )
    assert (empty["yql"], empty["userQuery"]) == (
        "select * from video_frame_acme_acme where userInput(@userQuery)",
        "",
    )


def _nearest_neighbor_hybrid_query(
    backend: VespaSearchBackend, *, filters: dict | None = None
) -> dict:
    return backend._build_query(
        query_text="man shoveling snow",
        query_embeddings=None,
        rank_config={
            "use_nearestneighbor": True,
            "needs_text_query": True,
            "first_phase_embedding_field": "embedding",
            "nearestneighbor_field": "embedding",
            "nearestneighbor_tensor": "qt",
        },
        ranking_profile="hybrid_float_bm25",
        schema_name="video_xclip_acme_acme",
        limit=10,
        filters=filters or {},
        correlation_id="t",
    )


def test_nearest_neighbor_hybrid_ors_the_query_terms(
    backend: VespaSearchBackend,
) -> None:
    """The nearest neighbours and every document holding any query term are
    candidates, each with its full text features."""
    params = _nearest_neighbor_hybrid_query(backend)

    assert params["yql"] == (
        "select * from video_xclip_acme_acme where "
        '({grammar: "any"}userInput(@userQuery)) OR '
        "({targetHits: 10}nearestNeighbor(embedding, qt))"
    )
    assert params["userQuery"] == "man shoveling snow"


def test_nearest_neighbor_hybrid_keeps_filters(backend: VespaSearchBackend) -> None:
    params = _nearest_neighbor_hybrid_query(backend, filters={"video_id": "v1"})

    assert params["yql"] == (
        "select * from video_xclip_acme_acme where "
        '(({grammar: "any"}userInput(@userQuery)) OR '
        "({targetHits: 10}nearestNeighbor(embedding, qt))) "
        'AND video_id contains "v1"'
    )
