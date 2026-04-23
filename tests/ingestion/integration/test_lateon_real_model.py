"""Real-model smoke test for LateOn.

Validates the foundational claim of the lateon_mv profile plan: PyLate can
load ``lightonai/LateOn`` (with its custom residual-MLP projection head) and
produces per-token embeddings of shape ``(N, 128)``. If this fails, the
lateon_mv schema's ``v[128]`` tensor size is wrong and the sidecar/Vespa
wiring won't work.

Marked ``requires_models`` + ``slow`` because it downloads ~520MB on first
run. Run explicitly via: ``uv run pytest -m requires_models -v``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_models, pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def lateon_model():
    pylate_models = pytest.importorskip("pylate.models")
    return pylate_models.ColBERT("lightonai/LateOn", device="cpu")


def test_lateon_encodes_document_to_128_dim_per_token(lateon_model):
    doc = (
        "Vespa is a platform for low-latency computation over large, evolving "
        "datasets, supporting structured and vector search together."
    )
    result = lateon_model.encode([doc], is_query=False)
    assert len(result) == 1
    tokens = np.asarray(result[0])
    assert tokens.ndim == 2, f"expected 2D, got {tokens.shape}"
    assert tokens.shape[1] == 128, (
        f"LateOn output dim must be 128 (lateon_mv schema uses v[128]), got {tokens.shape[1]}"
    )
    assert tokens.shape[0] > 0
    assert tokens.shape[0] <= 300, (
        f"LateOn doc max is 299 tokens; got {tokens.shape[0]}"
    )


def test_lateon_encodes_query_to_128_dim_per_token(lateon_model):
    query = "what is a vector database"
    result = lateon_model.encode([query], is_query=True)
    tokens = np.asarray(result[0])
    assert tokens.shape[1] == 128
    assert tokens.shape[0] > 0
    # LateOn query max is 32 tokens; small queries usually pad or stay short.
    assert tokens.shape[0] <= 32, (
        f"LateOn query max is 32 tokens; got {tokens.shape[0]}"
    )


def test_lateon_is_query_flag_changes_token_count(lateon_model):
    """Query-side encoding uses different markers + max length than doc-side.
    Exactly the semantics the sidecar's is_query flag must preserve."""
    text = "vector retrieval for code search"
    doc_tokens = np.asarray(lateon_model.encode([text], is_query=False)[0])
    query_tokens = np.asarray(lateon_model.encode([text], is_query=True)[0])
    assert doc_tokens.shape[1] == 128
    assert query_tokens.shape[1] == 128
    # The actual token counts differ because query/doc have different prompting.
    assert doc_tokens.shape[0] != query_tokens.shape[0] or not np.allclose(
        doc_tokens, query_tokens
    ), "is_query must change either token count or embedding values"


def test_lateon_related_pair_scores_higher_than_unrelated(lateon_model):
    """End-to-end sanity: MaxSim between related query/doc > unrelated pair.
    Not strict correctness — guards against catastrophic model misload
    (e.g., projection head loaded wrong produces near-random vectors)."""
    query = lateon_model.encode(["what is a vector database"], is_query=True)[0]
    related = lateon_model.encode(
        ["Vespa is a vector database for low-latency search over billions of vectors."],
        is_query=False,
    )[0]
    unrelated = lateon_model.encode(
        ["The Sphinx cat is a rare breed with minimal fur and high body temperature."],
        is_query=False,
    )[0]

    def max_sim(q: np.ndarray, d: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)
        sims = q @ d.T  # (Nq, Nd) — no norm; LateOn outputs are unit-scaled by training
        return float(sims.max(axis=1).sum())

    related_score = max_sim(query, related)
    unrelated_score = max_sim(query, unrelated)
    assert related_score > unrelated_score, (
        f"related MaxSim ({related_score:.3f}) should exceed unrelated "
        f"({unrelated_score:.3f}); if close, the custom projection head may "
        f"be loading incorrectly"
    )
