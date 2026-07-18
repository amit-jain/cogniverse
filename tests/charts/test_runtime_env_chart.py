"""Unit tests for the runtime Deployment's instrumentation/tuning env vars.

``OPENINFERENCE_DSPY`` and ``ITER_RETRIEVAL_WALL_CLOCK_MS`` were previously
set only by patching the live Deployment; every redeploy silently dropped
them, killing DSPy LM span export and shrinking the iterative-retrieval
wall clock back to the 30s library default. These tests render the chart
with ``helm template`` and pin both env vars on the runtime container so
the wiring can only be removed deliberately.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"


pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render_chart(*set_args: str) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for s in set_args:
        args += ["--set", s]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _runtime_container_env(manifests: list) -> dict:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-runtime"
    ]
    assert len(deployments) == 1, (
        f"Expected exactly one cogniverse-runtime Deployment, got {len(deployments)}"
    )
    containers = deployments[0]["spec"]["template"]["spec"]["containers"]
    runtime = [c for c in containers if c["name"] == "runtime"]
    assert len(runtime) == 1, "runtime container missing from the Deployment"
    return {e["name"]: e.get("value") for e in runtime[0].get("env", [])}


def _ingestor_container_env(manifests: list) -> dict:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-ingestor"
    ]
    assert len(deployments) == 1, (
        f"Expected exactly one cogniverse-ingestor Deployment, got {len(deployments)}"
    )
    containers = deployments[0]["spec"]["template"]["spec"]["containers"]
    ingestor = [c for c in containers if c["name"] == "ingestor"]
    assert len(ingestor) == 1, "ingestor container missing from the Deployment"
    return {e["name"]: e.get("value") for e in ingestor[0].get("env", [])}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestIngestorMinioEnv:
    def test_ingestor_has_minio_default_bucket_matching_runtime(self):
        """The ingestor uploads extracted keyframes to object storage during
        ingestion, so it needs MINIO_DEFAULT_BUCKET — the same bucket the videos
        land in, so answer-time agents resolve keyframes from the hit's
        source_url bucket."""
        manifests = _render_chart()
        ingestor = _ingestor_container_env(manifests)
        runtime = _runtime_container_env(manifests)
        assert ingestor.get("MINIO_DEFAULT_BUCKET")
        assert ingestor["MINIO_DEFAULT_BUCKET"] == runtime.get("MINIO_DEFAULT_BUCKET")


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRuntimeInstrumentationEnv:
    def test_openinference_dspy_enabled(self):
        env = _runtime_container_env(_render_chart())
        assert env.get("OPENINFERENCE_DSPY") == "1", (
            "OPENINFERENCE_DSPY must be '1' on the runtime container — without "
            "it DSPy LM spans never reach the cogniverse-dspy-instrumentation "
            "Phoenix project"
        )

    def test_iter_retrieval_wall_clock_set_from_values(self):
        env = _runtime_container_env(_render_chart())
        assert env.get("ITER_RETRIEVAL_WALL_CLOCK_MS") == "120000", (
            "ITER_RETRIEVAL_WALL_CLOCK_MS must come from "
            "runtime.iterRetrieval.wallClockMs (default 120000) — the 30s "
            "library default hits wall_clock before max_iter on the "
            "in-cluster LM"
        )

    def test_iter_retrieval_wall_clock_override(self):
        env = _runtime_container_env(
            _render_chart("runtime.iterRetrieval.wallClockMs=45000")
        )
        assert env.get("ITER_RETRIEVAL_WALL_CLOCK_MS") == "45000"


def _render_with_values(*values_files: str) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for f in values_files:
        args += ["-f", str(CHART_PATH / f)]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _dev_mount_count(manifests: list) -> int:
    count = 0
    for m in manifests:
        if m.get("kind") != "Deployment":
            continue
        for vol in (
            m.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
            or []
        ):
            if vol.get("name") == "src-libs":
                count += 1
    return count


class TestDeviceOverlaysKeepDevMode:
    """The device overlays layer on top of values.k3s.yaml in `cogniverse up`
    (base first, device file second — later files win per key). A devMode
    key in a device overlay silently disables the k3s dev-source mounts on
    that device's hosts, killing the edit→restart loop."""

    def test_k3s_plus_cuda_keeps_dev_mounts(self):
        manifests = _render_with_values("values.k3s.yaml", "values.cuda.yaml")
        assert _dev_mount_count(manifests) > 0

    def test_base_plus_cuda_stays_non_dev(self):
        manifests = _render_with_values("values.cuda.yaml")
        assert _dev_mount_count(manifests) == 0

    def test_k3s_plus_rocm_keeps_dev_mounts(self):
        manifests = _render_with_values("values.k3s.yaml", "values.rocm.yaml")
        assert _dev_mount_count(manifests) > 0
