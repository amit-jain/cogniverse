"""Post-hoc multi-vector token pooling for document-side embeddings.

Wraps colpali_engine's HierarchicalTokenPooler (parameter-free agglomerative
clustering + mean-pool + L2 renormalize). Applied to document/frame/chunk
multi-vectors before Vespa feed; never to queries.
"""

from __future__ import annotations

import numpy as np


def pool_document_tokens(embeddings: np.ndarray, pool_factor: int | None) -> np.ndarray:
    """Pool an ``(n_tokens, dim)`` document embedding to ``(~n_tokens/pool_factor, dim)``.

    ``pool_factor`` None/<=1, or a single-token input, returns the input
    unchanged. Output rows are L2-normalized (MaxSim-compatible).
    """
    if (
        pool_factor is None
        or pool_factor <= 1
        or embeddings.ndim < 2
        or embeddings.shape[0] <= 1
    ):
        return embeddings
    import torch
    from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

    pooler = HierarchicalTokenPooler()
    pooled = pooler.pool_embeddings(
        [torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))],
        pool_factor=pool_factor,
    )
    return pooled[0].cpu().numpy().astype(np.float32)
