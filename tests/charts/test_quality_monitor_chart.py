"""Unit tests for the cogniverse Helm chart's QualityMonitor sidecar wiring.

the QualityMonitor sidecar previously crash-looped because
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
GOLDEN_FILE_IN_CHART = CHART_PATH / "files" / "quality-monitor" / "golden_dataset.json"


pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


SEED_SET = ("--set", f"runtime.qualityMonitor.goldenDatasetPath={EXPECTED_MOUNT_PATH}")


def _render_chart(*extra_args: str) -> list:
    """Run ``helm template`` against the chart and return all parsed manifests.

    The shipped default configures no golden-set seed; pass ``*SEED_SET`` to
    render the seeded variant the dev and e2e overlays use."""
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            # The chart fail-fasts on empty qualityMonitor.tenantId; supply a
            # placeholder so the wiring under test is the only variable.
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
            *extra_args,
        ],
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
        manifests = _render_chart(*SEED_SET)

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
        manifests = _render_chart(*SEED_SET)

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
        manifests = _render_chart(*SEED_SET)

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
        manifests = _render_chart(*SEED_SET)

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


def _sidecar_and_volumes(manifests: list) -> tuple[dict | None, list]:
    for deployment in manifests:
        if deployment.get("kind") != "Deployment":
            continue
        if "runtime" not in deployment.get("metadata", {}).get("name", ""):
            continue
        spec = deployment.get("spec", {}).get("template", {}).get("spec", {})
        for container in spec.get("containers", []):
            if container.get("name") == "quality-monitor":
                return container, spec.get("volumes", [])
    return None, []


class TestGoldenSetSeedIsOptIn:
    """The shipped default seeds nothing: no seed file is mounted, no ConfigMap
    is rendered and the CLI receives no ``--golden-dataset-path``, so the monitor
    reports ``golden_set_missing`` until the tenant uploads a golden set. The dev
    and e2e overlays opt in by setting the seed path."""

    def test_default_render_has_no_seed_wiring(self):
        manifests = _render_chart()
        sidecar, volumes = _sidecar_and_volumes(manifests)
        assert sidecar["name"] == "quality-monitor"
        assert "--golden-dataset-path" not in sidecar.get("args", [])
        assert [
            m
            for m in sidecar.get("volumeMounts", [])
            if m.get("mountPath") == EXPECTED_MOUNT_PATH
        ] == []
        assert [v for v in volumes if v.get("name") == EXPECTED_VOLUME_NAME] == []
        assert [
            m
            for m in manifests
            if m.get("kind") == "ConfigMap"
            and m.get("metadata", {}).get("name") == EXPECTED_CONFIGMAP_NAME
        ] == []

    def test_seeded_render_passes_the_mounted_path_once(self):
        manifests = _render_chart(*SEED_SET)
        sidecar, volumes = _sidecar_and_volumes(manifests)
        assert sidecar["name"] == "quality-monitor"
        args = sidecar.get("args", [])
        assert args.count("--golden-dataset-path") == 1
        assert args[args.index("--golden-dataset-path") + 1] == EXPECTED_MOUNT_PATH
        assert [
            v.get("name") for v in volumes if v.get("name") == EXPECTED_VOLUME_NAME
        ] == [EXPECTED_VOLUME_NAME]

    def test_k3s_overlay_seeds_the_bundled_corpus(self):
        overlay = yaml.safe_load((CHART_PATH / "values.k3s.yaml").read_text())
        assert (
            overlay["runtime"]["qualityMonitor"]["goldenDatasetPath"]
            == EXPECTED_MOUNT_PATH
        )
        shipped = yaml.safe_load((CHART_PATH / "values.yaml").read_text())
        assert shipped["runtime"]["qualityMonitor"]["goldenDatasetPath"] == ""

    def test_scheduled_distillation_seed_wiring_follows_the_same_switch(self):
        def distillation(manifests):
            for m in manifests:
                if m.get("kind") == "CronWorkflow" and m["metadata"]["name"].endswith(
                    "-scheduled-distillation"
                ):
                    return m
            raise AssertionError("scheduled-distillation CronWorkflow not rendered")

        def seed_wiring(cron):
            spec = cron["spec"]["workflowSpec"]
            args = 0
            mounts = 0
            for template in spec.get("templates", []):
                container = template.get("container") or {}
                args += container.get("args", []).count("--golden-dataset-path")
                mounts += sum(
                    1
                    for m in container.get("volumeMounts", [])
                    if m.get("mountPath") == EXPECTED_MOUNT_PATH
                )
            declared = list(spec.get("volumes", [])) + [
                v
                for template in spec.get("templates", [])
                for v in template.get("volumes", [])
            ]
            volumes = sum(1 for v in declared if v.get("name") == EXPECTED_VOLUME_NAME)
            return args, mounts, volumes

        assert seed_wiring(distillation(_render_chart())) == (0, 0, 0)
        assert seed_wiring(distillation(_render_chart(*SEED_SET))) == (1, 1, 1)
