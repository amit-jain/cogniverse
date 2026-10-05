"""Post-hoc multi-vector token pooling for document-side embeddings.

Hierarchical token pooling, the method of colpali_engine's
``HierarchicalTokenPooler``, in numpy and scipy (the runtime image ships no
torch): Ward-linkage clustering of the tokens, cut at
``max(n_tokens // pool_factor, 1)`` clusters, each cluster mean-pooled and
L2-renormalized. Applied to document/frame/chunk multi-vectors before Vespa
feed; never to queries.
"""

from __future__ import annotations

import numpy as np


def pool_document_tokens(embeddings: np.ndarray, pool_factor: int | None) -> np.ndarray:
    """Pool an ``(n_tokens, dim)`` document embedding to ``(~n_tokens/pool_factor, dim)``.

    ``pool_factor`` None/<=1, or a single-token input, returns the input
    unchanged. Output rows are L2-normalized (MaxSim-compatible), float32.
    """
    if (
        pool_factor is None
        or pool_factor <= 1
        or embeddings.ndim < 2
        or embeddings.shape[0] <= 1
    ):
        return embeddings
    from scipy.cluster.hierarchy import fcluster, linkage

    tokens = np.ascontiguousarray(embeddings, dtype=np.float32)
    # Each token's row of cosine distances to every token is its point; Ward
    # linkage over those rows groups tokens that sit alike against the rest.
    distances = 1 - tokens @ tokens.T
    tree = linkage(distances, metric="euclidean", method="ward")
    max_clusters = max(tokens.shape[0] // pool_factor, 1)
    labels = fcluster(tree, t=max_clusters, criterion="maxclust") - 1

    pooled = []
    for cluster_id in range(max_clusters):
        members = tokens[labels == cluster_id]
        if len(members):
            mean = members.mean(axis=0)
            pooled.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
    return np.stack(pooled).astype(np.float32)
