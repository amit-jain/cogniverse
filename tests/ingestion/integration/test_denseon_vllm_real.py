"""Real vLLM DenseOn parity — production embedder vs sentence-transformers oracle.

Spawns ``vllm/vllm-openai-cpu`` serving ``lightonai/DenseOn`` and exercises:

- the raw OpenAI-compatible ``/v1/embeddings`` contract (200, single consistent
  dim, in-order indices), and
- PARITY of the *production* client path. ``RemoteOpenAIEmbedder`` applies the
  ``document: ``/``query: `` prompt and L2-normalizes (the behavior the pylate
  sidecar always applied, restored client-side by B0). We therefore test the
  production embedder, NOT a raw /v1/embeddings call with a manual prefix, and
  compare it against ``SentenceTransformer.encode([f"document: {text}"],
  normalize_embeddings=True)``. If someone reverts B0 (drops the prompt or the
  normalization) this cosine check fails.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import requests

from cogniverse_core.common.models.semantic_embedder import RemoteOpenAIEmbedder

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

DENSEON_MODEL = "lightonai/DenseOn"


@pytest.fixture(scope="module")
def denseon_url(vllm_sidecar):
    return vllm_sidecar.spawn(model=DENSEON_MODEL)


@pytest.fixture(scope="module")
def sentence_transformer_oracle():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    return sentence_transformers.SentenceTransformer(DENSEON_MODEL, device="cpu")


def test_raw_v1_embeddings_contract(denseon_url):
    resp = requests.post(
        f"{denseon_url}/v1/embeddings",
        json={"model": DENSEON_MODEL, "input": ["a", "b"]},
        timeout=120,
    )
    assert resp.status_code == 200, (
        f"/v1/embeddings must return 200; got {resp.status_code}: {resp.text[:500]}"
    )
    rows = resp.json()["data"]
    assert len(rows) == 2, f"two inputs must yield two rows; got {len(rows)}"
    assert [r["index"] for r in rows] == [0, 1], (
        f"rows must carry in-order indices [0, 1]; got {[r['index'] for r in rows]}"
    )
    dims = {len(r["embedding"]) for r in rows}
    assert len(dims) == 1, f"all rows must share one dim; got dims {dims}"
    assert next(iter(dims)) > 0


def test_production_embedder_matches_sentence_transformer_oracle(
    denseon_url, sentence_transformer_oracle
):
    text = "Vespa is a vector database for low-latency retrieval."

    embedder = RemoteOpenAIEmbedder(base_url=denseon_url, model=DENSEON_MODEL)
    remote_vec = np.asarray(embedder.encode([text], is_query=False), dtype=np.float32)

    # Production applies `document: ` + L2-normalize; mirror that in the oracle.
    local_vec = np.asarray(
        sentence_transformer_oracle.encode(
            [f"document: {text}"], normalize_embeddings=True
        ),
        dtype=np.float32,
    )

    assert remote_vec.shape == local_vec.shape, (
        f"remote shape {remote_vec.shape} must match oracle {local_vec.shape}"
    )

    remote_row = remote_vec[0]
    local_row = local_vec[0]
    # Production normalizes; the oracle was asked to normalize too — both unit.
    np.testing.assert_allclose(
        np.linalg.norm(remote_row),
        1.0,
        atol=1e-2,
        err_msg="RemoteOpenAIEmbedder must L2-normalize its output (B0)",
    )

    cosine = float(np.dot(remote_row, local_row))
    assert cosine >= 0.99, (
        f"production RemoteOpenAIEmbedder(is_query=False) must match "
        f"SentenceTransformer('document: ...', normalize=True) at cosine ≥ 0.99; "
        f"got {cosine:.4f}. A drop below this means the document prompt or the "
        f"normalization (B0) drifted."
    )
