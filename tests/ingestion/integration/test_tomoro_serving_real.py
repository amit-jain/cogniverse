"""Real vLLM Tomoro serving — 320-dim normalization + pooling-gate coverage.

Spawns ``vllm/vllm-openai-cpu`` serving ``TomoroAI/tomoro-colqwen3-embed-4b``
(320-dim multi-vector) and drives queries + images through
``RemoteColPaliLoader``. ``test_vllm_colpali_real_sidecar.py`` already covers
the bare 320-dim query/image round-trip shape; this file is the focused
*ranking + token-pooling eval gate*:

- query rows are L2-normalized (MaxSim-compatible),
- MaxSim ranks the matching image above a distractor, and
- ``pool_document_tokens(pool_factor=3)`` preserves both the 320 dim and the
  match > distractor ordering — the property the document-side pooling step
  must not break before Vespa feed.
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pytest
from PIL import Image

from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader
from cogniverse_runtime.ingestion.processors.embedding_generator.token_pooling import (
    pool_document_tokens,
)

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

TOMORO_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"
EMBED_DIM = 320


@pytest.fixture(scope="module")
def tomoro_url(vllm_sidecar):
    return vllm_sidecar.spawn(
        model=TOMORO_MODEL,
        extra_args=[
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.10",
        ],
    )


@pytest.fixture(scope="module")
def tomoro_client(tomoro_url):
    loader = RemoteColPaliLoader(
        model_name=TOMORO_MODEL,
        config={"remote_inference_url": tomoro_url},
        logger=logging.getLogger("test"),
    )
    client, processor = loader.load_model()
    assert client is processor
    return client


def _query_embedding(client, text: str) -> np.ndarray:
    result = client.process_queries([text], model_name=TOMORO_MODEL)
    return np.asarray(result["embeddings"], dtype=np.float32)


def _image_embedding(client, path) -> np.ndarray:
    result = client.process_images([path], model_name=TOMORO_MODEL)
    return np.asarray(result["embeddings"], dtype=np.float32)


def _max_sim(query: np.ndarray, doc: np.ndarray) -> float:
    """Sum over query tokens of the max dot product against any doc token."""
    sims = query @ doc.T  # (Nq, Nd)
    return float(sims.max(axis=1).sum())


def test_query_is_320d_l2_normalized(tomoro_client):
    query = _query_embedding(tomoro_client, "a lush green field")

    assert query.ndim == 2, (
        f"query embedding must be 2-D [num_query_tokens, dim]; got {query.shape}"
    )
    assert query.shape[1] == EMBED_DIM, (
        f"Tomoro serves {EMBED_DIM}-dim embeddings; got dim {query.shape[1]}"
    )
    assert query.shape[0] > 0, "must have at least one query token"

    row_norms = np.linalg.norm(query, axis=1)
    np.testing.assert_allclose(
        row_norms,
        np.ones_like(row_norms),
        atol=1e-2,
        err_msg=f"query rows must be L2-normalized; got norms {row_norms}",
    )


def test_maxsim_ranks_match_above_distractor(tomoro_client, tmp_path):
    green_path = tmp_path / "green.png"
    red_path = tmp_path / "red.png"
    Image.new("RGB", (224, 224), color=(0, 200, 0)).save(green_path)
    Image.new("RGB", (224, 224), color=(200, 0, 0)).save(red_path)

    query = _query_embedding(tomoro_client, "green")
    green_doc = _image_embedding(tomoro_client, green_path)
    red_doc = _image_embedding(tomoro_client, red_path)

    match_score = _max_sim(query, green_doc)
    distractor_score = _max_sim(query, red_doc)

    assert match_score > distractor_score, (
        f"MaxSim(green, green_img)={match_score:.4f} must exceed "
        f"MaxSim(green, red_img)={distractor_score:.4f}; if not, the served "
        f"model or per-token extraction is mis-wired"
    )


def test_pool_factor_3_preserves_ranking(tomoro_client, tmp_path):
    green_path = tmp_path / "green.png"
    red_path = tmp_path / "red.png"
    Image.new("RGB", (224, 224), color=(0, 200, 0)).save(green_path)
    Image.new("RGB", (224, 224), color=(200, 0, 0)).save(red_path)

    query = _query_embedding(tomoro_client, "green")
    green_doc = _image_embedding(tomoro_client, green_path)
    red_doc = _image_embedding(tomoro_client, red_path)

    pooled_green = pool_document_tokens(green_doc, pool_factor=3)
    pooled_red = pool_document_tokens(red_doc, pool_factor=3)

    assert pooled_green.shape[1] == EMBED_DIM, (
        f"pooling must preserve the {EMBED_DIM} dim; got {pooled_green.shape}"
    )
    assert pooled_red.shape[1] == EMBED_DIM, (
        f"pooling must preserve the {EMBED_DIM} dim; got {pooled_red.shape}"
    )
    assert pooled_green.shape[0] <= green_doc.shape[0], (
        "pooling must not increase token count"
    )

    match_score = _max_sim(query, pooled_green)
    distractor_score = _max_sim(query, pooled_red)
    assert match_score > distractor_score, (
        f"after pool_factor=3, MaxSim(green, pooled_green)={match_score:.4f} "
        f"must still exceed MaxSim(green, pooled_red)={distractor_score:.4f}; "
        f"pooling that flips ranking would degrade retrieval quality"
    )
