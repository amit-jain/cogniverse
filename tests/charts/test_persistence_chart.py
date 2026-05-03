"""Chart tests for the prod-persistence stack — PVC-backed HF cache, the
MinIO mirror init container, the populate Job, and nightly DR snapshot
CronWorkflows for vespa + phoenix.

The chart supports three model-cache volume modes that must coexist:

* ``hostStorage.enabled=true`` (dev) — single hostPath bind-mount shared
  by every pod. Lets ``cogniverse cluster create`` mount a host directory
  so model downloads survive pod restarts on the local k3d cluster.
* ``hfCache.persistence.enabled=true`` (prod) — one PVC per pod, with an
  init container that pre-warms the cache via ``snapshot_download``.
* default (legacy) — anonymous emptyDir hostPath fallback. Pods download
  from HF Hub on cold start every time.

Persistence mode also has an optional MinIO sub-mode that runs an extra
init container to mirror models from a MinIO bucket BEFORE
``snapshot_download`` so air-gapped clusters never reach out to HF Hub.

These tests pin the wiring so a refactor of any of those templates can't
silently break the others.
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


def _render(*set_args: str) -> list[dict]:
    cmd = ["helm", "template", "cogniverse", str(CHART_PATH)]
    # The chart fail-fasts if qualityMonitor.tenantId is empty; supply a
    # placeholder so persistence wiring is the only variable under test.
    cmd.extend(["--set", "runtime.qualityMonitor.tenantId=test-tenant"])
    for arg in set_args:
        cmd.extend(["--set", arg])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n{result.stderr}"
        )
    return [d for d in yaml.safe_load_all(result.stdout) if d is not None]


def _by_kind(docs: list[dict], kind: str) -> list[dict]:
    return [d for d in docs if d.get("kind") == kind]


def _named(docs: list[dict], kind: str, name: str) -> dict | None:
    for d in _by_kind(docs, kind):
        if d.get("metadata", {}).get("name") == name:
            return d
    return None


def _model_cache_volume(pod_spec: dict) -> dict:
    for v in pod_spec.get("volumes", []):
        if v["name"] == "model-cache":
            return v
    raise AssertionError("model-cache volume not found in pod spec")


def _container(pod_spec: dict, name: str) -> dict | None:
    for c in pod_spec.get("containers", []) + pod_spec.get("initContainers", []):
        if c["name"] == name:
            return c
    return None


def _runtime_pod_spec(docs: list[dict]) -> dict:
    for d in _by_kind(docs, "Deployment"):
        labels = d.get("metadata", {}).get("labels", {})
        if labels.get("app.kubernetes.io/component") == "runtime":
            return d["spec"]["template"]["spec"]
    raise AssertionError("runtime Deployment not found")


def _inference_pod_specs(docs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in _by_kind(docs, "Deployment"):
        labels = d.get("metadata", {}).get("labels", {})
        component = labels.get("app.kubernetes.io/component", "")
        if component.startswith("inference-"):
            out[component.removeprefix("inference-")] = d["spec"]["template"]["spec"]
    return out


# ---------- Mode 1: legacy default (no persistence, no hostStorage) -----------


def test_default_mode_uses_legacy_hostpath_and_no_pvcs():
    docs = _render()
    runtime = _runtime_pod_spec(docs)
    vol = _model_cache_volume(runtime)
    assert "hostPath" in vol, "default mode should use hostPath fallback"

    pvcs = [
        d
        for d in _by_kind(docs, "PersistentVolumeClaim")
        if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
        == "hf-cache"
    ]
    assert pvcs == [], "default mode should not create hf-cache PVCs"


def test_default_mode_inference_pods_have_no_model_warm_init():
    docs = _render()
    for name, spec in _inference_pod_specs(docs).items():
        assert _container(spec, "model-warm") is None, (
            f"inference svc {name} should not have model-warm init in default mode"
        )
        assert _container(spec, "model-warm-minio") is None, (
            f"inference svc {name} should not have model-warm-minio init "
            f"in default mode"
        )


# ---------- Mode 2: dev hostStorage bind-mount ----------


def test_host_storage_mode_uses_hostpath_bind_mount():
    docs = _render(
        "hostStorage.enabled=true",
        "hostStorage.path=/var/lib/cogniverse",
    )
    runtime = _runtime_pod_spec(docs)
    vol = _model_cache_volume(runtime)
    assert "hostPath" in vol
    assert vol["hostPath"]["path"].startswith("/var/lib/cogniverse"), (
        f"hostStorage mode should mount under the configured path, got {vol}"
    )


def test_host_storage_mode_does_not_create_pvcs():
    docs = _render("hostStorage.enabled=true")
    pvcs = [
        d
        for d in _by_kind(docs, "PersistentVolumeClaim")
        if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
        == "hf-cache"
    ]
    assert pvcs == []


# ---------- Mode 3: prod PVC persistence ----------


def test_persistence_mode_creates_pvc_per_pod():
    docs = _render(
        "hfCache.persistence.enabled=true",
        "hfCache.persistence.size=100Gi",
    )
    pvcs = {
        d["metadata"]["name"]: d
        for d in _by_kind(docs, "PersistentVolumeClaim")
        if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
        == "hf-cache"
    }
    # Runtime + ingestor + each enabled inference svc each get their own PVC.
    assert "cogniverse-runtime-model-cache" in pvcs
    assert "cogniverse-ingestor-model-cache" in pvcs
    assert "cogniverse-colbert-pylate-model-cache" in pvcs
    assert "cogniverse-denseon-model-cache" in pvcs

    runtime_pvc = pvcs["cogniverse-runtime-model-cache"]
    assert runtime_pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert runtime_pvc["spec"]["resources"]["requests"]["storage"] == "100Gi"


def test_persistence_mode_runtime_pod_uses_pvc_volume():
    docs = _render("hfCache.persistence.enabled=true")
    runtime = _runtime_pod_spec(docs)
    vol = _model_cache_volume(runtime)
    assert "persistentVolumeClaim" in vol, (
        f"runtime pod should reference PVC in persistence mode, got {vol}"
    )
    assert vol["persistentVolumeClaim"]["claimName"] == "cogniverse-runtime-model-cache"


def test_persistence_mode_inference_pods_use_their_own_pvc():
    docs = _render("hfCache.persistence.enabled=true")
    specs = _inference_pod_specs(docs)
    assert specs, "expected at least one inference Deployment"
    for name, spec in specs.items():
        vol = _model_cache_volume(spec)
        assert "persistentVolumeClaim" in vol, (
            f"inference {name} should reference its own PVC, got {vol}"
        )
        # PVC name follows ``cogniverse-<svc-kebab>-model-cache``.
        kebab = name.replace("_", "-")
        assert vol["persistentVolumeClaim"]["claimName"] == (
            f"cogniverse-{kebab}-model-cache"
        ), (
            f"inference {name} PVC ref should target its own claim, "
            f"got {vol['persistentVolumeClaim']['claimName']}"
        )


def test_persistence_mode_inference_pods_have_model_warm_init():
    docs = _render("hfCache.persistence.enabled=true")
    for name, spec in _inference_pod_specs(docs).items():
        warm = _container(spec, "model-warm")
        assert warm is not None, (
            f"inference {name} must have model-warm init container in persistence mode"
        )
        assert "snapshot_download" in (warm.get("args", [""])[-1]), (
            f"model-warm init for {name} should call snapshot_download, got {warm}"
        )
        # init container must mount the same PVC the runtime container uses.
        mounts = {m["name"]: m["mountPath"] for m in warm.get("volumeMounts", [])}
        assert mounts.get("model-cache") == "/root/.cache/huggingface"


def test_persistence_mode_without_minio_has_no_minio_init_or_job():
    docs = _render("hfCache.persistence.enabled=true")
    for name, spec in _inference_pod_specs(docs).items():
        assert _container(spec, "model-warm-minio") is None, (
            f"inference {name} should NOT have model-warm-minio when minio disabled"
        )
    populate = _named(docs, "Job", "cogniverse-hf-cache-minio-populate")
    assert populate is None


# ---------- Mode 3 + MinIO mirror sub-mode ----------


def test_minio_mode_adds_mirror_init_before_model_warm():
    docs = _render(
        "hfCache.persistence.enabled=true",
        "hfCache.persistence.minio.enabled=true",
        "hfCache.persistence.minio.existingSecret=cogniverse-minio",
        "hfCache.persistence.minio.models[0]=hf-internal-testing/tiny-random-bert",
    )
    for name, spec in _inference_pod_specs(docs).items():
        init_names = [c["name"] for c in spec.get("initContainers", [])]
        assert "model-warm-minio" in init_names, (
            f"inference {name} should have model-warm-minio init when minio enabled, "
            f"got {init_names}"
        )
        assert "model-warm" in init_names
        # Mirror must run before snapshot_download so HF-Hub reach-out is skipped
        # when the bucket already has the model.
        assert init_names.index("model-warm-minio") < init_names.index("model-warm"), (
            f"model-warm-minio must precede model-warm for {name}, got {init_names}"
        )


def test_minio_mode_creates_populate_job_with_helm_hooks():
    docs = _render(
        "hfCache.persistence.enabled=true",
        "hfCache.persistence.minio.enabled=true",
        "hfCache.persistence.minio.existingSecret=cogniverse-minio",
        "hfCache.persistence.minio.models[0]=hf-internal-testing/tiny-random-bert",
    )
    populate = _named(docs, "Job", "cogniverse-hf-cache-minio-populate")
    assert populate is not None, "MinIO populate Job must render"
    annotations = populate["metadata"].get("annotations", {})
    assert "post-install" in annotations.get("helm.sh/hook", "")
    assert "post-upgrade" in annotations.get("helm.sh/hook", "")


def test_minio_mode_without_models_skips_populate_job():
    """Operators may enable minio mode but defer model selection. The populate
    Job is gated on a non-empty models list — wiring it without inputs would
    schedule a no-op pod every helm upgrade."""
    docs = _render(
        "hfCache.persistence.enabled=true",
        "hfCache.persistence.minio.enabled=true",
        "hfCache.persistence.minio.existingSecret=cogniverse-minio",
    )
    populate = _named(docs, "Job", "cogniverse-hf-cache-minio-populate")
    assert populate is None


# ---------- Backup CronWorkflows ----------


def test_backup_disabled_by_default_renders_no_workflows():
    docs = _render()
    cws = [
        d
        for d in _by_kind(docs, "CronWorkflow")
        if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
        == "backup"
    ]
    assert cws == [], "backup CronWorkflows must not render when disabled"


def test_backup_enabled_renders_vespa_and_phoenix_cronworkflows():
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
    )
    names = sorted(
        d["metadata"]["name"]
        for d in _by_kind(docs, "CronWorkflow")
        if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
        == "backup"
    )
    assert names == ["cogniverse-backup-phoenix", "cogniverse-backup-vespa"]


def test_backup_workflow_dump_step_targets_correct_data_path():
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
    )
    vespa = _named(docs, "CronWorkflow", "cogniverse-backup-vespa")
    phoenix = _named(docs, "CronWorkflow", "cogniverse-backup-phoenix")
    assert vespa is not None and phoenix is not None

    def _dump_args(cw: dict) -> str:
        for tmpl in cw["spec"]["workflowSpec"]["templates"]:
            if tmpl["name"] == "dump":
                return tmpl["container"]["args"][-1]
        raise AssertionError("dump template not found")

    assert "/opt/vespa/var" in _dump_args(vespa)
    assert "/mnt/data" in _dump_args(phoenix)


def test_backup_grants_pods_exec_via_role():
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
    )
    role = _named(docs, "Role", "cogniverse-backup-exec")
    assert role is not None
    rules = role["rules"]
    exec_rule = next((r for r in rules if "pods/exec" in r["resources"]), None)
    assert exec_rule is not None, f"Role must grant pods/exec, got {rules}"
    assert "create" in exec_rule["verbs"]
