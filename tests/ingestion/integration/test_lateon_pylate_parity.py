"""Real PyLate-service LateOn parity — served per-token embeddings vs the
in-process pylate oracle.

Resolves the ``colbert_pylate`` service through the shared inference
fixture (locally: the ``deploy/pylate`` image built and run by the
fixture) and drives text through ``RemoteColBERTLoader``. Asserts:

- the per-token matrix is 2-D, 128-dim (LateOn stays 128), L2-normalized,
- PARITY against the in-process ``pylate.models.ColBERT`` oracle at the
  same pinned revision: cosine ≥ 0.99 per token for both ``is_query=True``
  and ``is_query=False``, and
- MaxSim ranks a relevant document above a distractor through the remote
  path.

Both sides run pylate's canonical encode — the service inside the
container, the oracle in-process — so any drift in the served revision,
the ``/pooling`` request contract, or the response parsing breaks the
cosine check. The stock vLLM ``/pooling`` route cannot pass this test:
its request schema carries no attention mask, so PyLate's query expansion
(mask padding excluded from attention) is unreproducible there and
query-side per-token cosine tops out near 0.88.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

pytestmark = [
    pytest.mark.requires_docker,
    pytest.mark.requires_models,
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.requires_inference("colbert_pylate"),
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not installed",
    ),
]

LATEON_MODEL = "lightonai/LateOn"
LATEON_REVISION = "c01907b70557ee5c7753680d4819a5cce1674b83"
EMBED_DIM = 128


@pytest.fixture(scope="module")
def remote_lateon(resolved_inference_endpoints):
    endpoint = resolved_inference_endpoints["colbert_pylate"]
    loader = RemoteColBERTLoader(
        model_name=LATEON_MODEL,
        config={"remote_inference_url": endpoint.base_url},
        logger=logging.getLogger("test"),
        _resolved_headers=dict(endpoint.headers),
    )
    model, _ = loader.load_model()
    yield model
    model._close()


@pytest.fixture(scope="module")
def pylate_oracle():
    """The in-process reference at the exact served revision, loaded from
    the writable test-owned cache the service containers also use."""
    pylate_models = pytest.importorskip("pylate.models")

    from tests.utils.vllm_sidecar import writable_test_hf_cache

    return pylate_models.ColBERT(
        LATEON_MODEL,
        device="cpu",
        revision=LATEON_REVISION,
        cache_folder=str(Path(writable_test_hf_cache()) / "hub"),
    )


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _per_token_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(_l2(a) * _l2(b), axis=1)


def test_remote_lateon_is_128d_l2_normalized(remote_lateon):
    tokens = np.asarray(
        remote_lateon.encode(
            ["Vespa is a vector database for low-latency retrieval."],
            is_query=False,
        )[0],
        dtype=np.float32,
    )

    assert tokens.ndim == 2, f"expected 2-D per-token matrix, got {tokens.shape}"
    assert tokens.shape[1] == EMBED_DIM, (
        f"LateOn stays {EMBED_DIM}-dim; got {tokens.shape[1]}"
    )
    assert tokens.shape[0] > 0

    row_norms = np.linalg.norm(tokens, axis=1)
    np.testing.assert_allclose(
        row_norms,
        np.ones_like(row_norms),
        atol=1e-2,
        err_msg=f"LateOn token rows must be L2-normalized; got {row_norms}",
    )


@pytest.mark.parametrize("is_query", [True, False])
def test_remote_lateon_matches_pylate_oracle(remote_lateon, pylate_oracle, is_query):
    text = (
        "what is a vector database"
        if is_query
        else ("Vespa stores token embeddings as tensor<bfloat16>(token{}, v[128]).")
    )

    remote_tokens = np.asarray(
        remote_lateon.encode([text], is_query=is_query)[0], dtype=np.float32
    )
    oracle_tokens = np.asarray(
        pylate_oracle.encode([text], is_query=is_query)[0], dtype=np.float32
    )

    assert remote_tokens.shape == oracle_tokens.shape, (
        f"is_query={is_query}: remote shape {remote_tokens.shape} must match "
        f"pylate oracle shape {oracle_tokens.shape} — both sides run pylate's "
        f"canonical encode (query expansion, document skiplist), so any shape "
        f"drift means the service is not serving the pinned PyLate contract"
    )
    assert remote_tokens.shape[1] == EMBED_DIM

    cosines = _per_token_cosine(remote_tokens, oracle_tokens)
    assert float(cosines.min()) >= 0.99, (
        f"is_query={is_query}: every token must match the pylate oracle at "
        f"cosine ≥ 0.99; got min {float(cosines.min()):.4f} "
        f"(per-token cosines: {np.round(cosines, 4).tolist()})"
    )


def _maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    sims = _l2(query_tokens) @ _l2(doc_tokens).T
    return float(sims.max(axis=1).sum())


def test_remote_lateon_maxsim_ranks_relevant_above_distractor(remote_lateon):
    """Encode query + relevant/distractor docs ALL via the remote path and
    assert the relevant doc out-scores the distractor under MaxSim. Proves
    the served encode preserves retrieval quality, not just shape parity.
    """
    query = "how does Vespa store token embeddings"
    relevant = "Vespa stores token embeddings as tensor<bfloat16>(token{}, v[128])."
    distractor = "The chef seasoned the soup with fresh basil and a pinch of salt."

    q = np.asarray(remote_lateon.encode([query], is_query=True)[0], dtype=np.float32)
    rel = np.asarray(
        remote_lateon.encode([relevant], is_query=False)[0], dtype=np.float32
    )
    dist = np.asarray(
        remote_lateon.encode([distractor], is_query=False)[0], dtype=np.float32
    )

    rel_score = _maxsim(q, rel)
    dist_score = _maxsim(q, dist)
    assert rel_score > dist_score, (
        f"relevant doc MaxSim {rel_score:.4f} must exceed distractor "
        f"{dist_score:.4f} via the PyLate service path"
    )
