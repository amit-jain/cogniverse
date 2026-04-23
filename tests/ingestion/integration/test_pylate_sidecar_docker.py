"""End-to-end smoke test for the pylate inference Docker image.

Builds ``deploy/pylate/`` into a local image, runs it with a real LateOn
model, and hits ``/pooling`` with real text. Asserts the response shape is
what ``RemoteColBERTLoader`` expects. This is the end-to-end proof the
sidecar actually produces correct embeddings; unit tests with mocks do not.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
SIDECAR_DIR = REPO_ROOT / "deploy" / "pylate"
IMAGE_TAG = "cogniverse/pylate:inttest"
CONTAINER_NAME = "cogniverse-pylate-inttest"

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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(
    cmd: list[str], *, timeout: int = 60, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=check, timeout=timeout
    )


def _wait_for_health(base_url: str, deadline_seconds: int = 300) -> None:
    end = time.monotonic() + deadline_seconds
    while time.monotonic() < end:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    # Dump container logs so the failure is debuggable.
    logs = subprocess.run(
        ["docker", "logs", CONTAINER_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    raise AssertionError(
        f"sidecar at {base_url} did not become healthy within {deadline_seconds}s\n"
        f"--- container logs ---\n{logs.stdout}\n{logs.stderr}"
    )


@pytest.fixture(scope="module")
def running_sidecar():
    # Always rebuild to pick up server.py / Dockerfile / requirements changes.
    _run(
        ["docker", "build", "-t", IMAGE_TAG, str(SIDECAR_DIR)],
        timeout=1200,
    )

    _run(["docker", "rm", "-f", CONTAINER_NAME], check=False, timeout=30)
    port = _free_port()
    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{port}:8080",
            "-e",
            "MODEL_NAME=lightonai/LateOn",
            "-e",
            "DEVICE=cpu",
            IMAGE_TAG,
        ],
        timeout=30,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url)
        yield base_url
    finally:
        _run(["docker", "rm", "-f", CONTAINER_NAME], check=False, timeout=30)


def test_health_reports_lateon_model(running_sidecar):
    body = requests.get(f"{running_sidecar}/health", timeout=5).json()
    assert body["status"] == "ok"
    assert body["model"] == "lightonai/LateOn"


def test_pooling_document_returns_128_dim_per_token(running_sidecar):
    resp = requests.post(
        f"{running_sidecar}/pooling",
        json={
            "input": ["Vespa is a vector database for low-latency retrieval."],
            "is_query": False,
        },
        timeout=60,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "lightonai/LateOn"
    tokens = body["data"][0]["data"]
    assert len(tokens) > 0
    assert all(len(row) == 128 for row in tokens), (
        f"LateOn must produce 128-dim per-token outputs (lateon_mv schema); "
        f"got {len(tokens[0])}. If this is 1536, the sidecar is serving an "
        f"intermediate hidden state instead of applying the projection head."
    )


def test_pooling_query_respects_max_length(running_sidecar):
    resp = requests.post(
        f"{running_sidecar}/pooling",
        json={"input": ["what is a vector database"], "is_query": True},
        timeout=60,
    )
    body = resp.json()
    tokens = body["data"][0]["data"]
    assert all(len(row) == 128 for row in tokens)
    # LateOn query max is 32 tokens (short queries are padded up to that).
    assert len(tokens) <= 32


def test_pooling_is_query_flag_changes_output(running_sidecar):
    text = "vector retrieval for code search"
    doc_resp = requests.post(
        f"{running_sidecar}/pooling",
        json={"input": [text], "is_query": False},
        timeout=60,
    ).json()
    query_resp = requests.post(
        f"{running_sidecar}/pooling",
        json={"input": [text], "is_query": True},
        timeout=60,
    ).json()
    # Query and doc encoding differ — either token count or actual values.
    doc_tokens = doc_resp["data"][0]["data"]
    query_tokens = query_resp["data"][0]["data"]
    assert len(doc_tokens) != len(query_tokens) or doc_tokens != query_tokens


def test_pooling_rejects_empty_input(running_sidecar):
    resp = requests.post(
        f"{running_sidecar}/pooling",
        json={"input": []},
        timeout=10,
    )
    assert resp.status_code == 400
