"""Unit tests for the cogniverse Helm chart's QualityMonitor sidecar wiring.

Audit fix #1 — the QualityMonitor sidecar previously crash-looped because
``data/testset/`` was never mounted into the container. The CLI was passing
a relative path to a file that didn't exist inside the pod. These tests
render the chart with ``helm template`` and assert that:

1. A ConfigMap with the bundled golden dataset is created.
2. The sidecar mounts that ConfigMap at the expected absolute path.
3. The CLI ``--golden-dataset-path`` argument matches the mountPath.

These three checks together would have caught the original bug, and they
fail loudly if anyone breaks the wiring in the future.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
EXPECTED_MOUNT_PATH = "/app/data/quality-monitor/golden_dataset.json"
EXPECTED_CONFIGMAP_NAME = "cogniverse-quality-monitor-data"
EXPECTED_VOLUME_NAME = "quality-monitor-data"
GOLDEN_FILE_IN_CHART = (
    CHART_PATH / "files" / "quality-monitor" / "golden_dataset.json"
)


pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render_chart() -> list:
    """Run ``helm template`` against the chart and return all parsed manifests."""
    result = subprocess.run(
        ["helm", "template", "cogniverse", str(CHART_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestQualityMonitorDatasetMount:
    def test_golden_dataset_file_exists_in_chart(self):
        """The bundled file must exist for the ConfigMap to render."""
        assert GOLDEN_FILE_IN_CHART.exists(), (
            f"Bundled golden dataset missing at {GOLDEN_FILE_IN_CHART}. "
            "Run: cp data/testset/evaluation/sample_videos_retrieval_queries.json "
            f"{GOLDEN_FILE_IN_CHART}"
        )
        # Make sure the file is real JSON, not a stale stub.
        json.loads(GOLDEN_FILE_IN_CHART.read_text())

    def test_chart_renders_configmap_with_dataset(self):
        """The chart must render a ConfigMap named ``cogniverse-quality-monitor-data``
        whose ``golden_dataset.json`` key holds parseable JSON."""
        manifests = _render_chart()

        configmaps = [
            m
            for m in manifests
            if m.get("kind") == "ConfigMap"
            and m.get("metadata", {}).get("name") == EXPECTED_CONFIGMAP_NAME
        ]
        assert len(configmaps) == 1, (
            f"Expected exactly one ConfigMap named {EXPECTED_CONFIGMAP_NAME}, "
            f"got {len(configmaps)}"
        )

        cm = configmaps[0]
        assert "golden_dataset.json" in cm["data"], (
            "ConfigMap is missing the 'golden_dataset.json' data key — the chart "
            "is failing to load files/quality-monitor/golden_dataset.json"
        )
        # Content must be valid JSON.
        parsed = json.loads(cm["data"]["golden_dataset.json"])
        assert isinstance(parsed, list) and len(parsed) > 0, (
            "Golden dataset embedded in ConfigMap is empty or not a list"
        )

    def test_sidecar_mounts_dataset_at_expected_path(self):
        """The QualityMonitor sidecar inside the runtime Deployment must
        declare a volumeMount that points the ConfigMap volume at the
        expected absolute path."""
        manifests = _render_chart()

        deployments = [
            m
            for m in manifests
            if m.get("kind") == "Deployment"
            and "runtime" in m.get("metadata", {}).get("name", "")
        ]
        assert deployments, "No runtime Deployment found in rendered chart"

        # Find the qualityMonitor sidecar in any runtime deployment.
        sidecar = None
        for deployment in deployments:
            containers = (
                deployment.get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("containers", [])
            )
            for container in containers:
                if container.get("name") == "quality-monitor":
                    sidecar = container
                    break
            if sidecar:
                break

        assert sidecar is not None, (
            "quality-monitor sidecar not found in any runtime Deployment. "
            "Check that runtime.qualityMonitor.enabled is true by default."
        )

        mounts = sidecar.get("volumeMounts", [])
        matching = [m for m in mounts if m.get("mountPath") == EXPECTED_MOUNT_PATH]
        assert len(matching) == 1, (
            f"Expected exactly one volumeMount at {EXPECTED_MOUNT_PATH}, "
            f"got {len(matching)}. All mounts: {mounts}"
        )
        assert matching[0]["name"] == EXPECTED_VOLUME_NAME, (
            f"Mount at {EXPECTED_MOUNT_PATH} should reference volume "
            f"{EXPECTED_VOLUME_NAME}, got {matching[0]['name']}"
        )
        assert matching[0].get("subPath") == "golden_dataset.json", (
            "Mount must use subPath: golden_dataset.json so the ConfigMap "
            "key projects to a single file rather than a directory"
        )

    def test_sidecar_volume_references_configmap(self):
        """The volume named ``quality-monitor-data`` must be a configMap
        volume referencing the dataset ConfigMap."""
        manifests = _render_chart()

        for deployment in manifests:
            if deployment.get("kind") != "Deployment":
                continue
            volumes = (
                deployment.get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("volumes", [])
            )
            for volume in volumes:
                if volume.get("name") == EXPECTED_VOLUME_NAME:
                    cm_ref = volume.get("configMap", {})
                    assert cm_ref.get("name") == EXPECTED_CONFIGMAP_NAME, (
                        f"Volume {EXPECTED_VOLUME_NAME} should reference "
                        f"ConfigMap {EXPECTED_CONFIGMAP_NAME}, got {cm_ref}"
                    )
                    return
        pytest.fail(
            f"Volume {EXPECTED_VOLUME_NAME} not declared in any runtime Deployment"
        )

    def test_cli_arg_matches_mount_path(self):
        """The ``--golden-dataset-path`` argument passed to the CLI MUST equal
        the mountPath. If they drift apart, the sidecar opens a file that
        doesn't exist and crash-loops — exactly the original bug."""
        manifests = _render_chart()

        for deployment in manifests:
            if deployment.get("kind") != "Deployment":
                continue
            containers = (
                deployment.get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("containers", [])
            )
            for container in containers:
                if container.get("name") != "quality-monitor":
                    continue
                args = container.get("args", [])
                assert "--golden-dataset-path" in args, (
                    "quality-monitor sidecar must pass --golden-dataset-path"
                )
                idx = args.index("--golden-dataset-path")
                # The next item is the path value.
                assert args[idx + 1] == EXPECTED_MOUNT_PATH, (
                    f"--golden-dataset-path={args[idx + 1]!r} does not match "
                    f"the mountPath {EXPECTED_MOUNT_PATH!r}. The sidecar will "
                    "FileNotFoundError on startup."
                )
                return
        pytest.fail("quality-monitor sidecar not found in rendered chart")
