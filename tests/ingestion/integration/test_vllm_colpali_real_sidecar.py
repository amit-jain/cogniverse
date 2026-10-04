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
from pathlib import Path

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
def vllm_colpali_url(remote_inference):
    return remote_inference.resolve("vllm_colpali").base_url


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


CLIP = (
    Path(__file__).resolve().parents[3]
    / "tests/system/resources/videos/v_-D1gdv_gQyw.mp4"
)

# Decodes the ten frames _generate_chunk_embeddings samples from the 18 s,
# 1280x720 clip, encodes them through the remote client, and reports
# resident memory before the call and the high-water mark it reached.
_PEAK_SCRIPT = """
import json, sys
import cv2
from PIL import Image
from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader

def status(key):
    with open("/proc/self/status") as f:
        line = next(l for l in f if l.startswith(key))
    return int(line.split()[1]) // 1024

cap = cv2.VideoCapture(sys.argv[2])
step = int(cap.get(cv2.CAP_PROP_FPS) / 0.5)
frames = []
for index in list(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), step))[:10]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
cap.release()
client, _ = RemoteColPaliLoader(
    sys.argv[3], {"remote_inference_url": sys.argv[1]}
).load_model()
with open("/proc/self/clear_refs", "w") as f:
    f.write("5")
before = status("VmRSS")
result = client.process_images(frames, model_name=sys.argv[3])
print(json.dumps({
    "before_mib": before,
    "peak_mib": status("VmHWM"),
    "dtype": str(result["embeddings"].dtype),
    "shape": list(result["embeddings"].shape),
}))
"""


def test_a_chunk_s_frames_come_back_as_one_float32_array(remote_colpali_client):
    import cv2

    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    for index in (0, 60, 120):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok, index
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    result = remote_colpali_client.process_images(frames, model_name=COLPALI_MODEL)

    assert result["embeddings"].dtype == np.float32
    assert result["embeddings"].shape == (3, 887, 320)


def test_encoding_a_chunk_s_frames_holds_only_their_vectors(vllm_colpali_url):
    """Ten 1280x720 frames return 10 x 887 x 320 values, about 11 MiB as
    float32. Parsed as JSON they are 2.8 million Python floats, about 90 MiB,
    and an object array of them doubles that: the call then raised the
    process's resident peak by 249 MiB. Converting each response to float32
    as it arrives keeps the rise to the in-flight responses (71 MiB)."""
    import json
    import subprocess
    import sys

    done = subprocess.run(
        [
            sys.executable,
            "-c",
            _PEAK_SCRIPT,
            vllm_colpali_url,
            str(CLIP),
            COLPALI_MODEL,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    report = json.loads(done.stdout.strip().splitlines()[-1])

    assert (report["dtype"], report["shape"]) == ("float32", [10, 887, 320])
    assert report["peak_mib"] - report["before_mib"] < 128, report
