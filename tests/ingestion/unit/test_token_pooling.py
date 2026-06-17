import numpy as np

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
