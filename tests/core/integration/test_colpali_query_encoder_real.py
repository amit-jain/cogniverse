"""Real-model shape contract for ColPaliQueryEncoder.

ImageSearchAgent._search_vespa assumes the query encoder returns a 2D
multi-vector array ``(num_tokens, 128)`` — it builds the ``query(qt)``
mapped tensor as ``{str(i): row.tolist() for i, row in enumerate(emb)}`` and
the deployed ``image_colpali_mv`` schema declares ``v[128]``. This pins that
contract against the real model so an encoder-output drift (rank, dim, or a
wrapped object) is caught before it reaches Vespa.

Marked ``requires_models`` + ``slow``: downloads/loads colsmol-500m. Run via
``uv run pytest -m requires_models``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_core.query.encoders import ColPaliQueryEncoder

pytestmark = [pytest.mark.requires_models, pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def colpali_encoder():
    return ColPaliQueryEncoder("vidore/colsmol-500m")


def test_encode_returns_2d_128dim_multivector(colpali_encoder):
    emb = colpali_encoder.encode("a sunset over the mountains")

    assert isinstance(emb, np.ndarray)
    # The exact contract _search_vespa relies on for its ndim==2 qt branch.
    assert emb.ndim == 2, f"expected (tokens, dim), got {emb.shape}"
    assert emb.shape[0] >= 1, "no query tokens produced"
    assert emb.shape[1] == 128
    assert emb.shape[1] == colpali_encoder.embedding_dim
    assert np.issubdtype(emb.dtype, np.floating)


def test_encode_output_maps_to_per_token_128_vectors(colpali_encoder):
    emb = colpali_encoder.encode("query tokens here")

    # Mirror _search_vespa's qt construction: every token row must be a
    # 128-length float vector the v[128] schema accepts.
    qt = {str(i): row.tolist() for i, row in enumerate(emb)}
    assert len(qt) == emb.shape[0]
    assert all(len(vec) == 128 for vec in qt.values())
