"""The GLiNER image serves its baked ONNX export of the pinned checkpoint.

Each test runs the image built from ``deploy/gliner/Dockerfile`` under the
CPU and memory limits the chart gives the pod, and checks what the gateway
depends on: the exported graph answers with the checkpoint's own entities,
on as many threads as the quota grants, with no network and no model
download, and refuses to start on a missing artifact.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import yaml

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

REPO = Path(__file__).resolve().parents[3]
IMAGE = "cogniverse/gliner:integration-test"
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


def _pod_limits() -> tuple[str, str]:
    """The gliner pod's CPU and memory limits from the chart, as docker flags."""
    values = yaml.safe_load((REPO / "charts/cogniverse/values.yaml").read_text())
    limits = values["inference"]["gliner"]["resources"]["limits"]
    return str(limits["cpu"]), str(limits["memory"]).removesuffix("Gi") + "g"


@pytest.fixture(scope="module")
def gliner_image() -> str:
    build = subprocess.run(
        ["docker", "build", "-f", "deploy/gliner/Dockerfile", "-t", IMAGE, "."],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert build.returncode == 0, build.stderr[-4000:]
    return IMAGE


def _free_port() -> int:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


@contextmanager
def _container(image: str, *, env: dict[str, str] | None = None, network: bool = True):
    cpus, memory = _pod_limits()
    name = f"gliner-it-{uuid.uuid4().hex[:8]}"
    port = _free_port()
    command = ["docker", "run", "-d", "--name", name, f"--cpus={cpus}"]
    command += [f"--memory={memory}"]
    command += ["-p", f"127.0.0.1:{port}:8080"] if network else ["--network", "none"]
    for key, value in (env or {}).items():
        command += ["-e", f"{key}={value}"]
    started = subprocess.run(
        [*command, image], capture_output=True, text=True, timeout=120
    )
    assert started.returncode == 0, started.stderr
    try:
        yield f"http://127.0.0.1:{port}", name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)


def _health_inside(name: str) -> tuple[int, dict]:
    """GET /health from inside the container; works without a network."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        probe = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "curl",
                "-s",
                "-w",
                "\n%{http_code}",
                "http://localhost:8080/health",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        body, _, code = probe.stdout.rpartition("\n")
        if code.isdigit() and int(code) != 0:
            return int(code), json.loads(body)
        time.sleep(1)
    raise AssertionError(f"{name} never answered /health: {_logs(name)}")


def _logs(name: str) -> str:
    return subprocess.run(
        ["docker", "logs", name], capture_output=True, text=True, timeout=60
    ).stderr


def _predict(url: str, text: str, labels: list[str], threshold: float) -> list:
    response = httpx.post(
        f"{url}/predict_entities",
        json={"text": text, "labels": labels, "threshold": threshold},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    return response.json()["entities"]


@pytest.fixture(scope="module")
def served(gliner_image):
    with _container(gliner_image) as (url, name):
        status, body = _health_inside(name)
        assert status == 200, body
        yield url, name


def test_image_serves_the_export_on_quota_sized_threads(served):
    url, name = served
    cpus, _ = _pod_limits()

    loaded = [line for line in _logs(name).splitlines() if "GLiNER loaded" in line]

    assert [line.split(" INFO gliner_server: ")[1] for line in loaded] == [
        f"GLiNER loaded: {MODEL} backend=onnxruntime device=cpu threads={cpus}"
    ]
    assert httpx.get(f"{url}/health", timeout=10).json() == {
        "status": "ready",
        "model": MODEL,
        "model_revision": REVISION,
        "loaded_models": [MODEL],
    }


def test_export_reproduces_the_checkpoint_entities(served):
    url, _ = served

    for text, labels, threshold, expected in TORCH_REFERENCE:
        entities = _predict(url, text, labels, threshold)

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
    url, _ = served
    cases = TORCH_REFERENCE * 2
    sequential = [_predict(url, *case[:3]) for case in cases]

    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        concurrent = list(executor.map(lambda case: _predict(url, *case[:3]), cases))

    assert concurrent == sequential


def test_image_serves_without_network_access(gliner_image):
    with _container(gliner_image, network=False) as (_, name):
        status, body = _health_inside(name)
        predicted = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "curl",
                "-s",
                "-X",
                "POST",
                "http://localhost:8080/predict_entities",
                "-H",
                "content-type: application/json",
                "-d",
                json.dumps(
                    {"text": "man lifting", "labels": GATEWAY_LABELS, "threshold": 0.3}
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert status == 200, body
    entities = json.loads(predicted.stdout)["entities"]
    assert [(e["label"], e["text"], e["start"], e["end"]) for e in entities] == [
        ("video_content", "man lifting", 0, 11)
    ]
    assert abs(entities[0]["score"] - 0.41419652104377747) <= SCORE_TOLERANCE


def test_missing_export_fails_readiness_with_its_path(gliner_image):
    with _container(gliner_image, env={"ONNX_MODEL_DIR": "/opt/absent"}) as (
        _,
        name,
    ):
        status, body = _health_inside(name)

    assert status == 503
    assert body == {
        "detail": (
            f"gliner: model {MODEL} load failed (FileNotFoundError): "
            "[Errno 2] No such file or directory: '/opt/absent/source.json'"
        )
    }
