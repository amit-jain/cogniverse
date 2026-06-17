"""Real vLLM LateOn parity — vLLM-served per-token embeddings vs pylate oracle.

Spawns ``vllm/vllm-openai-cpu`` serving ``lightonai/LateOn`` (forced to the
``ColBERTModernBertModel`` architecture via ``--hf-overrides``) and drives text
through ``RemoteColBERTLoader``. Asserts:

- the per-token matrix is 2-D, 128-dim (LateOn stays 128), L2-normalized, and
- PARITY against the in-process ``pylate.models.ColBERT`` oracle: cosine ≥ 0.99
  per token for both ``is_query=True`` and ``is_query=False``.

The vLLM wrapper reproduces pylate's full per-token contract client-side: it
prepends ``[Q] ``/``[D] `` (each a single vocabulary token, identical to pylate's
marker insertion) and, for documents, drops the same punctuation tokens pylate
removes via ``ColBERT.skiplist_mask``. The pylate oracle applies the same marker
and skiplist, so remote and oracle stay like-for-like under both ``is_query``
flags. A drift in the served projection head, the client prefix, the document
skiplist, or the /pooling response parsing breaks the cosine check.
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pytest

from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

pytestmark = [
    pytest.mark.requires_docker,
    pytest.mark.requires_models,
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker CLI not installed",
    ),
]

LATEON_MODEL = "lightonai/LateOn"
EMBED_DIM = 128


@pytest.fixture(scope="module")
def lateon_url(vllm_sidecar):
    return vllm_sidecar.spawn(
        model=LATEON_MODEL,
        extra_args=[
            "--hf-overrides",
            '{"architectures": ["ColBERTModernBertModel"]}',
        ],
    )


@pytest.fixture(scope="module")
def remote_lateon(lateon_url):
    loader = RemoteColBERTLoader(
        model_name=LATEON_MODEL,
        config={"remote_inference_url": lateon_url},
        logger=logging.getLogger("test"),
    )
    model, _ = loader.load_model()
    return model


@pytest.fixture(scope="module")
def pylate_oracle():
    pylate_models = pytest.importorskip("pylate.models")
    return pylate_models.ColBERT(LATEON_MODEL, device="cpu")


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
        f"pylate oracle shape {oracle_tokens.shape} for like-for-like token "
        f"cosine comparison (both apply the same [Q]/[D] marker, and for "
        f"documents both drop the punctuation skiplist tokens)"
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
    """Encode query + relevant/distractor docs ALL via the vLLM path and assert
    the relevant doc out-scores the distractor under MaxSim. Proves the
    document-side skiplist masking preserves retrieval quality (the punctuation
    rows pylate drops carry no signal), not just pylate shape parity.
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
        f"{dist_score:.4f} via the vLLM path"
    )
