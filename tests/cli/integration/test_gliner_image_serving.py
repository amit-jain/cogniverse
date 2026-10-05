"""The served GLiNER image answers with its pinned checkpoint's own entities.

The cluster's gliner pod runs the image built from ``deploy/gliner/Dockerfile``.
These tests check what the gateway depends on: the image carries the ONNX
export of the pinned checkpoint and serves it (no model download), on as many
threads as the pod's CPU quota grants; the service identifies the checkpoint;
the exported graph reproduces the PyTorch checkpoint's entities; and
concurrent requests each get their own answer. The pod is read with
``kubectl get``/``logs``/``exec`` only.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tests.utils.vllm_sidecar import E2E_CONTEXT

pytestmark = [pytest.mark.integration, pytest.mark.local_only]

MODEL = "urchade/gliner_large-v2.1"
REVISION = "abd49a1f1ebc12af1be84d06f6848221cf96dcad"
GATEWAY_LABELS = [
    "video_content",
    "text_information",
    "audio_content",
    "image_content",
    "document_content",
    "summary_request",
    "detailed_report_request",
]
# predict_entities of the PyTorch checkpoint (gliner 0.2.26, torch 2.5.1 CPU)
# for these inputs: (label, text, start, end, score).
TORCH_REFERENCE = [
    (
        "Can you describe the diving maneuvers performed in the video?",
        GATEWAY_LABELS,
        0.3,
        [("video_content", "video", 55, 60, 0.5044011473655701)],
    ),
    (
        "man lifting",
        GATEWAY_LABELS,
        0.3,
        [("video_content", "man lifting", 0, 11, 0.41419652104377747)],
    ),
    (
        "summarize the podcast episode about climate change",
        GATEWAY_LABELS,
        0.3,
        [
            ("summary_request", "summarize", 0, 9, 0.8040828704833984),
            ("audio_content", "podcast episode", 14, 29, 0.7753766179084778),
        ],
    ),
    (
        "find the PDF report on quarterly earnings",
        GATEWAY_LABELS,
        0.3,
        [
            ("document_content", "PDF report", 9, 19, 0.5643568634986877),
            ("text_information", "quarterly earnings", 23, 41, 0.702149510383606),
        ],
    ),
    (
        "show me images of red cars",
        GATEWAY_LABELS,
        0.3,
        [("image_content", "images", 8, 14, 0.3810480535030365)],
    ),
    (
        "What caused the biker to crash in the dirt field?",
        GATEWAY_LABELS,
        0.3,
        [
            (
                "text_information",
                "What caused the biker to crash in the dirt field?",
                0,
                49,
                0.4076498746871948,
            )
        ],
    ),
    (
        "Marie Curie founded institutes in Paris and Warsaw.",
        ["person", "city"],
        0.4,
        [
            ("person", "Marie Curie", 0, 11, 0.9956076741218567),
            ("city", "Paris", 34, 39, 0.9724048376083374),
            ("city", "Warsaw", 44, 50, 0.9849109649658203),
        ],
    ),
]
# float32 graph against float32 eager: the same arithmetic in another order.
SCORE_TOLERANCE = 1e-5


def _predict(url: str, text: str, labels: list[str], threshold: float) -> list:
    response = httpx.post(
        f"{url}/predict_entities",
        json={"text": text, "labels": labels, "threshold": threshold},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    return response.json()["entities"]


NAMESPACE = "cogniverse"
POD_SELECTOR = "app.kubernetes.io/component=inference-gliner"


def _kubectl(*args: str) -> str:
    result = subprocess.run(
        ["kubectl", "--context", E2E_CONTEXT, "-n", NAMESPACE, *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def served(remote_inference):
    return remote_inference.resolve("gliner").base_url


@pytest.fixture(scope="module")
def gliner_pod(remote_inference) -> dict:
    """The e2e pod behind the resolved endpoint, as ``kubectl get`` reports it."""
    endpoint = remote_inference.resolve("gliner")
    assert endpoint.provider == "e2e", endpoint
    pods = json.loads(_kubectl("get", "pods", "-l", POD_SELECTOR, "-o", "json"))
    running = [pod for pod in pods["items"] if pod["status"].get("phase") == "Running"]
    assert len(running) == 1, [pod["metadata"]["name"] for pod in pods["items"]]
    return running[0]


def test_image_serves_its_baked_export_of_the_pinned_checkpoint(gliner_pod):
    name = gliner_pod["metadata"]["name"]
    export_dir = _kubectl("exec", name, "--", "printenv", "ONNX_MODEL_DIR").strip()
    source = _kubectl("exec", name, "--", "cat", f"{export_dir}/source.json")

    assert json.loads(source) == {"model": MODEL, "revision": REVISION}


def test_image_serves_the_export_on_quota_sized_threads(gliner_pod):
    name = gliner_pod["metadata"]["name"]
    cpus = gliner_pod["spec"]["containers"][0]["resources"]["limits"]["cpu"]

    loaded = [
        line for line in _kubectl("logs", name).splitlines() if "GLiNER loaded" in line
    ]

    assert [line.split(" INFO gliner_server: ")[1] for line in loaded] == [
        f"GLiNER loaded: {MODEL} backend=onnxruntime device=cpu threads={cpus}"
    ]


def test_service_identifies_the_pinned_checkpoint(served):
    assert httpx.get(f"{served}/health", timeout=10).json() == {
        "status": "ready",
        "model": MODEL,
        "model_revision": REVISION,
        "loaded_models": [MODEL],
    }


def test_export_reproduces_the_checkpoint_entities(served):
    for text, labels, threshold, expected in TORCH_REFERENCE:
        entities = _predict(served, text, labels, threshold)

        assert [(e["label"], e["text"], e["start"], e["end"]) for e in entities] == [
            entity[:4] for entity in expected
        ], text
        for entity, reference in zip(entities, expected):
            assert abs(entity["score"] - reference[4]) <= SCORE_TOLERANCE, (
                text,
                entity,
                reference,
            )


def test_concurrent_requests_each_get_their_own_entities(served):
    cases = TORCH_REFERENCE * 2
    sequential = [_predict(served, *case[:3]) for case in cases]

    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        concurrent = list(executor.map(lambda case: _predict(served, *case[:3]), cases))

    assert concurrent == sequential
