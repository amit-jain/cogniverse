"""Unit tests for the cogniverse Helm chart's QualityMonitor wiring.

The monitor runs as its own Deployment (``cogniverse-quality-monitor``) and
reaches the runtime over the Service DNS name, so a monitor crash or restart
never drops the runtime Service endpoints. These tests render the chart with
``helm template`` and assert:

1. A ConfigMap with the bundled golden dataset is created.
2. The monitor mounts that ConfigMap at the expected absolute path.
3. The CLI ``--golden-dataset-path`` argument matches the mountPath.
4. The monitor is a standalone singleton Deployment, absent from the
   runtime pod, targeting ``http://cogniverse-runtime:8000``.
5. A config.json or golden-dataset content change rolls the monitor pod.
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
MONITOR_DEPLOYMENT_NAME = "cogniverse-quality-monitor"
CHECKSUM_ANNOTATION = "cogniverse.io/quality-monitor-config-checksum"


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

    def test_monitor_mounts_dataset_at_expected_path(self):
        """The monitor container must declare a volumeMount that points the
        ConfigMap volume at the expected absolute path."""
        manifests = _render_chart(*SEED_SET)

        monitor, _ = _monitor_and_volumes(manifests)

        mounts = monitor.get("volumeMounts", [])
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

    def test_monitor_volume_references_configmap(self):
        """The volume named ``quality-monitor-data`` must be a configMap
        volume referencing the dataset ConfigMap."""
        manifests = _render_chart(*SEED_SET)

        _, volumes = _monitor_and_volumes(manifests)
        matching = [v for v in volumes if v.get("name") == EXPECTED_VOLUME_NAME]
        assert len(matching) == 1, volumes
        assert matching[0]["configMap"]["name"] == EXPECTED_CONFIGMAP_NAME, (
            f"Volume {EXPECTED_VOLUME_NAME} should reference "
            f"ConfigMap {EXPECTED_CONFIGMAP_NAME}, got {matching[0]}"
        )

    def test_cli_arg_matches_mount_path(self):
        """The ``--golden-dataset-path`` argument passed to the CLI MUST equal
        the mountPath. If they drift apart, the monitor opens a file that
        doesn't exist and crash-loops."""
        manifests = _render_chart(*SEED_SET)

        monitor, _ = _monitor_and_volumes(manifests)
        args = monitor.get("args", [])
        assert "--golden-dataset-path" in args, (
            "quality-monitor must pass --golden-dataset-path"
        )
        idx = args.index("--golden-dataset-path")
        assert args[idx + 1] == EXPECTED_MOUNT_PATH, (
            f"--golden-dataset-path={args[idx + 1]!r} does not match "
            f"the mountPath {EXPECTED_MOUNT_PATH!r}. The monitor will "
            "FileNotFoundError on startup."
        )


def _monitor_deployment(manifests: list) -> dict:
    for deployment in manifests:
        if (
            deployment.get("kind") == "Deployment"
            and deployment.get("metadata", {}).get("name") == MONITOR_DEPLOYMENT_NAME
        ):
            return deployment
    raise AssertionError(
        f"{MONITOR_DEPLOYMENT_NAME} Deployment not found in rendered chart"
    )


def _monitor_and_volumes(manifests: list) -> tuple[dict, list]:
    spec = _monitor_deployment(manifests)["spec"]["template"]["spec"]
    containers = [c["name"] for c in spec["containers"]]
    assert containers == ["quality-monitor"], containers
    return spec["containers"][0], spec.get("volumes", [])


class TestGoldenSetSeedIsOptIn:
    """The shipped default seeds nothing: no seed file is mounted, no ConfigMap
    is rendered and the CLI receives no ``--golden-dataset-path``, so the monitor
    reports ``golden_set_missing`` until the tenant uploads a golden set. The dev
    and e2e overlays opt in by setting the seed path."""

    def test_default_render_has_no_seed_wiring(self):
        manifests = _render_chart()
        monitor, volumes = _monitor_and_volumes(manifests)
        assert monitor["name"] == "quality-monitor"
        assert "--golden-dataset-path" not in monitor.get("args", [])
        assert [
            m
            for m in monitor.get("volumeMounts", [])
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
        monitor, volumes = _monitor_and_volumes(manifests)
        assert monitor["name"] == "quality-monitor"
        args = monitor.get("args", [])
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


class TestMonitorIsItsOwnDeployment:
    """A monitor crash-loop or OOMKill inside the runtime pod drops the pod to
    1/2 Ready and removes it from the runtime Service endpoints — a monitoring
    defect becomes a serving outage. As its own Deployment the monitor can die
    without touching serving."""

    def test_monitor_renders_as_a_standalone_singleton_deployment(self):
        deployment = _monitor_deployment(_render_chart())

        assert deployment["metadata"]["labels"]["app.kubernetes.io/component"] == (
            "quality-monitor"
        )
        assert deployment["spec"]["replicas"] == 1
        assert deployment["spec"]["strategy"] == {"type": "Recreate"}
        selector = deployment["spec"]["selector"]["matchLabels"]
        assert selector["app.kubernetes.io/component"] == "quality-monitor"
        template = deployment["spec"]["template"]["metadata"]["labels"]
        assert template["app.kubernetes.io/component"] == "quality-monitor"
        containers = deployment["spec"]["template"]["spec"]["containers"]
        assert [c["name"] for c in containers] == ["quality-monitor"]

    def test_runtime_pod_runs_only_the_runtime_container(self):
        manifests = _render_chart(*SEED_SET)
        for deployment in manifests:
            if (
                deployment.get("kind") == "Deployment"
                and deployment["metadata"]["name"] == "cogniverse-runtime"
            ):
                spec = deployment["spec"]["template"]["spec"]
                assert [c["name"] for c in spec["containers"]] == ["runtime"]
                assert [
                    v["name"]
                    for v in spec.get("volumes", [])
                    if v["name"] == EXPECTED_VOLUME_NAME
                ] == []
                return
        pytest.fail("cogniverse-runtime Deployment not rendered")

    def test_runtime_url_targets_the_runtime_service(self):
        monitor, _ = _monitor_and_volumes(_render_chart())
        args = monitor["args"]
        assert args[args.index("--runtime-url") + 1] == (
            "http://cogniverse-runtime:8000"
        )

    def test_monitor_pod_carries_the_runtime_service_account(self):
        """The monitor spawns Argo optimization workflows; the RBAC that allows
        that rides the shared service account."""
        spec = _monitor_deployment(_render_chart())["spec"]["template"]["spec"]
        assert spec["serviceAccountName"] == "cogniverse"

    def test_monitor_memory_requests_equal_limits(self):
        monitor, _ = _monitor_and_volumes(_render_chart())
        resources = monitor["resources"]
        assert resources["requests"]["memory"] == resources["limits"]["memory"]
        assert resources["limits"]["memory"] == "3Gi"

    def test_disabled_renders_no_monitor_anywhere(self):
        manifests = _render_chart("--set", "runtime.qualityMonitor.enabled=false")
        assert [
            m["metadata"]["name"]
            for m in manifests
            if m.get("kind") == "Deployment"
            and m["metadata"]["name"] == MONITOR_DEPLOYMENT_NAME
        ] == []
        offenders = [
            (m["metadata"]["name"], c["name"])
            for m in manifests
            if m.get("kind") == "Deployment"
            for c in m["spec"]["template"]["spec"]["containers"]
            if c["name"] == "quality-monitor"
        ]
        assert offenders == []


class TestConfigChangeRollsTheMonitorPod:
    """The monitor reads config.json and the golden dataset at startup only and
    its image tag does not change on a config edit, so without a checksum
    annotation a pure config/golden change produces an identical pod spec and
    the monitor keeps running the content it booted with."""

    def _annotations(self, manifests: list) -> dict:
        template = _monitor_deployment(manifests)["spec"]["template"]
        return template["metadata"].get("annotations") or {}

    def test_monitor_pod_carries_a_config_checksum(self):
        annotations = self._annotations(_render_chart())
        assert CHECKSUM_ANNOTATION in annotations
        assert len(annotations[CHECKSUM_ANNOTATION]) == 64

    def test_a_config_change_changes_the_checksum(self):
        before = self._annotations(_render_chart())
        after = self._annotations(
            _render_chart(
                "--set",
                "runtime.primaryLLM.apiBase=https://changed.example.modal.run/v1",
            )
        )
        assert before[CHECKSUM_ANNOTATION] != after[CHECKSUM_ANNOTATION]

    def test_an_unrelated_change_leaves_the_checksum_alone(self):
        before = self._annotations(_render_chart())
        after = self._annotations(_render_chart("--set", "runtime.replicaCount=2"))
        assert before[CHECKSUM_ANNOTATION] == after[CHECKSUM_ANNOTATION]

    def test_a_golden_dataset_change_changes_the_seeded_checksum(self, tmp_path):
        """Golden content is chart-baked (no ``--set`` reaches it), so this
        renders a mutated copy of the chart. Seeded pods consume the file and
        must roll on its change; unseeded pods do not mount it and must not."""
        import shutil as _shutil

        chart_copy = tmp_path / "cogniverse"
        _shutil.copytree(CHART_PATH, chart_copy)
        golden = chart_copy / "files" / "quality-monitor" / "golden_dataset.json"
        rows = json.loads(golden.read_text())
        rows.append(dict(rows[0]))
        golden.write_text(json.dumps(rows))

        def render_copy(*extra):
            result = subprocess.run(
                [
                    "helm",
                    "template",
                    "cogniverse",
                    str(chart_copy),
                    "--set",
                    "runtime.qualityMonitor.tenantId=test-tenant",
                    *extra,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            return [doc for doc in yaml.safe_load_all(result.stdout) if doc]

        seeded_before = self._annotations(_render_chart(*SEED_SET))
        seeded_after = self._annotations(render_copy(*SEED_SET))
        assert seeded_before[CHECKSUM_ANNOTATION] != seeded_after[CHECKSUM_ANNOTATION]

        unseeded_before = self._annotations(_render_chart())
        unseeded_after = self._annotations(render_copy())
        assert (
            unseeded_before[CHECKSUM_ANNOTATION] == unseeded_after[CHECKSUM_ANNOTATION]
        )
