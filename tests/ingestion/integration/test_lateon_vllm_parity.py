"""Real vLLM LateOn parity — vLLM-served per-token embeddings vs pylate oracle.

Spawns ``vllm/vllm-openai-cpu`` serving ``lightonai/LateOn`` (forced to the
``ColBERTModernBertModel`` architecture via ``--hf-overrides``) and drives text
through ``RemoteColBERTLoader``. Asserts:

- the per-token matrix is 2-D, 128-dim (LateOn stays 128), L2-normalized, and
- PARITY against the in-process ``pylate.models.ColBERT`` oracle: cosine ≥ 0.99
  per token for both ``is_query=True`` and ``is_query=False``.

The vLLM wrapper prepends ``[Q] ``/``[D] `` client-side and the pylate oracle
applies its own query/document prefix, so we compare like-for-like (oracle and
remote both encode with the same ``is_query`` flag). A drift in the served
projection head, the client prefix, or the /pooling response parsing breaks the
cosine check.
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
        f"cosine comparison (both apply the same [Q]/[D] prefix)"
    )
    assert remote_tokens.shape[1] == EMBED_DIM

    cosines = _per_token_cosine(remote_tokens, oracle_tokens)
    assert float(cosines.min()) >= 0.99, (
        f"is_query={is_query}: every token must match the pylate oracle at "
        f"cosine ≥ 0.99; got min {float(cosines.min()):.4f} "
        f"(per-token cosines: {np.round(cosines, 4).tolist()})"
    )
