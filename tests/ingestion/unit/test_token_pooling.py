import subprocess
import sys

import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.embedding_generator.token_pooling import (
    pool_document_tokens,
)


def test_pool_factor_3_reduces_tokens_and_keeps_dim_and_norm():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, 320)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = pool_document_tokens(emb, pool_factor=3)
    assert out.shape[1] == 320
    assert out.shape[0] == max(30 // 3, 1)  # 10 clusters
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-2)


def test_pool_factor_1_or_none_is_identity():
    emb = np.ones((5, 320), dtype=np.float32)
    np.testing.assert_array_equal(pool_document_tokens(emb, pool_factor=1), emb)
    np.testing.assert_array_equal(pool_document_tokens(emb, pool_factor=None), emb)


def test_single_token_passthrough():
    emb = np.ones((1, 320), dtype=np.float32)
    np.testing.assert_array_equal(pool_document_tokens(emb, pool_factor=3), emb)


def test_one_dimensional_input_passthrough():
    # A 1D (dim,) vector has shape[0]==dim (>1), so it slipped past the
    # single-token guard and crashed the pooler; it must return unchanged.
    emb = np.ones(320, dtype=np.float32)
    np.testing.assert_array_equal(pool_document_tokens(emb, pool_factor=3), emb)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("pool_factor", [2, 3, 4])
def test_matches_colpali_engine_hierarchical_pooling(seed, pool_factor):
    """Same clusters and, up to float32 rounding, the same vectors as
    colpali_engine's HierarchicalTokenPooler, which needs torch."""
    import torch
    from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((12, 320))
    emb = centres[rng.integers(0, 12, 120)] + 0.4 * rng.standard_normal((120, 320))
    emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)

    expected = (
        HierarchicalTokenPooler()
        .pool_embeddings([torch.from_numpy(emb)], pool_factor=pool_factor)[0]
        .numpy()
    )
    got = pool_document_tokens(emb, pool_factor=pool_factor)

    assert got.dtype == np.float32
    assert got.shape == expected.shape == (120 // pool_factor, 320)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_pooling_does_not_import_torch():
    """The runtime image ships no torch; pooling runs there."""
    script = (
        "import sys\n"
        "import numpy as np\n"
        "from cogniverse_runtime.ingestion.processors.embedding_generator."
        "token_pooling import pool_document_tokens\n"
        "rng = np.random.default_rng(0)\n"
        "emb = rng.standard_normal((30, 320)).astype(np.float32)\n"
        "out = pool_document_tokens(emb, pool_factor=3)\n"
        "print(out.shape, 'torch' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert done.stdout.strip() == "(10, 320) False"
