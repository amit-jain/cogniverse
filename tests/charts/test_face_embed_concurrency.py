"""Clients keep one face-embed request in flight per sidecar CPU.

The sidecar runs one inference per CPU; requests beyond its CPU count only
queue, and the queueing counts against each request's timeout. The runtime
and the ingestion worker, the two processes that run the face pipeline, get
``FACE_EMBED_MAX_CONCURRENCY`` from the sidecar's CPU limit.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART_PATH = Path(__file__).resolve().parents[2] / "charts" / "cogniverse"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _concurrency(*set_args: str) -> dict:
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "-f",
        str(CHART_PATH / "values.k3s.yaml"),
        "-f",
        str(CHART_PATH / "values.rocm.yaml"),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for value in set_args:
        command += ["--set", value]
    rendered = subprocess.run(command, capture_output=True, text=True, check=True)
    found = {}
    for document in yaml.safe_load_all(rendered.stdout):
        if not document or document.get("kind") != "Deployment":
            continue
        for container in document["spec"]["template"]["spec"]["containers"]:
            for entry in container.get("env") or []:
                if entry.get("name") == "FACE_EMBED_MAX_CONCURRENCY":
                    found[document["metadata"]["name"]] = entry["value"]
    return found


def test_the_deployed_sidecar_s_two_cpus_allow_two_requests():
    assert _concurrency() == {
        "cogniverse-runtime": "2",
        "cogniverse-ingestor": "2",
    }


@pytest.mark.parametrize(
    ("cpu", "expected"), [("4", "4"), ("1500m", "1"), ("500m", "1")]
)
def test_the_concurrency_follows_the_sidecar_cpu_limit(cpu, expected):
    assert _concurrency(f"inference.face_embed.resources.limits.cpu={cpu}") == {
        "cogniverse-runtime": expected,
        "cogniverse-ingestor": expected,
    }


def test_the_sidecar_runs_one_onnx_thread_per_in_flight_request():
    """Clients keep one request in flight per sidecar CPU and the sidecar
    gives each one ONNX Runtime thread, so inference fills the CPU limit
    and no more."""
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "-f",
        str(CHART_PATH / "values.k3s.yaml"),
        "-f",
        str(CHART_PATH / "values.rocm.yaml"),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        "--set",
        "inference.face_embed.enabled=true",
    ]
    rendered = subprocess.run(command, capture_output=True, text=True, check=True)
    (sidecar,) = [
        container
        for document in yaml.safe_load_all(rendered.stdout)
        if document
        and document.get("kind") == "Deployment"
        and document["metadata"]["name"] == "cogniverse-face-embed"
        for container in document["spec"]["template"]["spec"]["containers"]
    ]
    env = {entry["name"]: entry.get("value") for entry in sidecar["env"]}

    assert env["FACE_EMBED_INTRA_OP_THREADS"] == "1"
    assert sidecar["resources"]["limits"]["cpu"] == "2"
    assert _concurrency()["cogniverse-ingestor"] == "2"
