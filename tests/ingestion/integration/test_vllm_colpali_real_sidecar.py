"""Real vLLM ColPali sidecar — RemoteColPaliLoader end-to-end coverage.

Spawns ``vllm/vllm-openai-cpu`` serving ``TomoroAI/tomoro-colqwen3-embed-4b`` and
drives a real image through ``RemoteColPaliLoader`` to verify
multi-vector embeddings come back with the expected shape. Catches
vLLM /pooling contract drift, payload shape regressions, and per-token
embedding extraction bugs that mocks miss.
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pytest
from PIL import Image

from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader

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

COLPALI_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"


@pytest.fixture(scope="module")
def vllm_colpali_url(vllm_sidecar):
    return vllm_sidecar.spawn(
        model=COLPALI_MODEL,
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
def remote_colpali_client(vllm_colpali_url):
    loader = RemoteColPaliLoader(
        model_name=COLPALI_MODEL,
        config={"remote_inference_url": vllm_colpali_url},
        logger=logging.getLogger("test"),
    )
    client, processor = loader.load_model()
    assert client is processor, (
        "RemoteColPaliLoader returns the client as both model and processor"
    )
    return client


def test_remote_colpali_returns_multivector_embeddings(remote_colpali_client, tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (224, 224), color=(0, 128, 255)).save(image_path)

    result = remote_colpali_client.process_images(
        [image_path], model_name=COLPALI_MODEL
    )
    embeddings = np.asarray(result["embeddings"])

    assert embeddings.ndim == 2, (
        f"ColPali per-token embeddings must be 2-D [num_patches, dim]; "
        f"got shape {embeddings.shape}"
    )
    assert embeddings.shape[1] == 320, (
        f"Tomoro serves 320-dim embeddings; got dim {embeddings.shape[1]}"
    )
    assert embeddings.shape[0] > 0, "must have at least one patch token"


def test_remote_colpali_query_encoding_returns_multivector_embeddings(
    remote_colpali_client,
):
    """Exercise the process_queries_vllm path bound by RemoteColPaliLoader."""
    result = remote_colpali_client.process_queries(
        ["a doctor explaining medical procedures"],
        model_name=COLPALI_MODEL,
    )
    embeddings = np.asarray(result["embeddings"])

    assert embeddings.ndim == 2, (
        f"ColPali query embeddings must be 2-D [num_query_tokens, dim]; "
        f"got shape {embeddings.shape}"
    )
    assert embeddings.shape[1] == 320, (
        f"Tomoro serves 320-dim embeddings; got dim {embeddings.shape[1]}"
    )
    assert embeddings.shape[0] > 0, "must have at least one query token"


def test_single_frame_chunk_preserves_multivector(remote_colpali_client, tmp_path):
    """A single-frame remote chunk must keep its (T, D) multivector, not
    mean-pool over the token dim to (D,) which the mv chunk schema rejects."""
    import cv2

    from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (  # noqa: E501
        EmbeddingGeneratorImpl,
    )

    mp4 = tmp_path / "single.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (224, 224))
    try:
        writer.write(np.zeros((224, 224, 3), dtype=np.uint8))
    finally:
        writer.release()

    gen = EmbeddingGeneratorImpl(
        {
            "model_loader": "colqwen",
            "embedding_model": COLPALI_MODEL,
            "schema_name": "video_colqwen",
            "fps": 1.0,
        },
        logging.getLogger("test"),
    )
    gen.model = remote_colpali_client
    gen.processor = remote_colpali_client

    result = gen._generate_chunk_embeddings(mp4)

    assert result is not None
    assert result.ndim == 2, f"single-frame chunk collapsed to {result.shape}"
    assert result.shape[1] == 320
    assert result.shape[0] > 1
    assert result.dtype == np.float32
