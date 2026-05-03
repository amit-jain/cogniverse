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


def test_model_warm_init_uses_runtime_image_for_consistent_deps():
    """The init container needs python + huggingface_hub + boto3 + tar.
    Inference images are inconsistent (pylate has tar but no boto3;
    vllm-rocm has both but is GPU-flavored). The runtime image is the
    only one we can pin: it ships all four and is already pulled on
    every cluster (the runtime pod uses it)."""
    docs = _render("hfCache.persistence.enabled=true")
    runtime_pod = _runtime_pod_spec(docs)
    runtime_image = runtime_pod["containers"][0]["image"]
    for name, spec in _inference_pod_specs(docs).items():
        warm = _container(spec, "model-warm")
        assert warm is not None
        assert warm["image"] == runtime_image, (
            f"{name}: model-warm must reuse the runtime image "
            f"({runtime_image}) to guarantee boto3+tar+hf_hub. "
            f"Got {warm['image']}"
        )


def test_model_warm_init_avoids_pipefail():
    """Regression: ``set -o pipefail`` is bash-only; vllm-rocm /bin/sh
    is dash which rejects it (``Illegal option -o pipefail``). The init
    script has no pipes, so plain ``set -eu`` is sufficient."""
    docs = _render("hfCache.persistence.enabled=true")
    for name, spec in _inference_pod_specs(docs).items():
        warm = _container(spec, "model-warm")
        assert warm is not None
        script = warm["args"][-1]
        assert "pipefail" not in script, (
            f"inference {name}: model-warm script must not use pipefail "
            f"(dash rejects it). Got:\n{script}"
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


def test_persistence_mode_without_minio_omits_minio_env_and_no_job():
    """Without minio mode, the single model-warm init must not carry MINIO_*
    env (so the script skips the mirror branch) and the populate Job must
    not render."""
    docs = _render("hfCache.persistence.enabled=true")
    for name, spec in _inference_pod_specs(docs).items():
        warm = _container(spec, "model-warm")
        assert warm is not None
        env_names = {e["name"] for e in warm.get("env", [])}
        assert "MINIO_ENDPOINT" not in env_names, (
            f"{name}: model-warm must not carry MINIO_* env when minio disabled, "
            f"got {env_names}"
        )
    populate = _named(docs, "Job", "cogniverse-hf-cache-minio-populate")
    assert populate is None


# ---------- Mode 3 + MinIO mirror sub-mode ----------


def test_minio_mode_injects_minio_env_into_model_warm():
    """A single init container handles both paths: try MinIO tarball
    mirror, fall back to snapshot_download. When MinIO mode is on the
    init must carry MINIO_* env so the script knows to try the mirror."""
    docs = _render(
        "hfCache.persistence.enabled=true",
        "hfCache.persistence.minio.enabled=true",
        "hfCache.persistence.minio.existingSecret=cogniverse-minio",
        "hfCache.persistence.minio.models[0]=hf-internal-testing/tiny-random-bert",
    )
    for name, spec in _inference_pod_specs(docs).items():
        init_names = [c["name"] for c in spec.get("initContainers", [])]
        assert init_names == ["model-warm"], (
            f"inference {name} should have a single ``model-warm`` init "
            f"(combined mirror + snapshot_download), got {init_names}"
        )
        warm = _container(spec, "model-warm")
        env = {e["name"]: e for e in warm.get("env", [])}
        assert "MINIO_ENDPOINT" in env
        assert "MINIO_BUCKET" in env
        assert env["MINIO_ACCESS_KEY"]["valueFrom"]["secretKeyRef"]["key"] == "rootUser"


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


def test_backup_enabled_defaults_to_vespa_and_phoenix():
    """Default backup set covers vespa (kubectl-exec into the source pod)
    AND phoenix (volume-mount of the hostStorage path — phoenix's
    distroless image lacks tar so kubectl-exec doesn't work)."""
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


def test_phoenix_backup_uses_volume_mount_not_kubectl_exec():
    """Regression: phoenix MUST use volume-mount mode (its distroless
    container lacks tar). The dump container should mount the source
    volume directly + run tar locally, NOT kubectl exec into phoenix."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
    )
    cw = _named(docs, "CronWorkflow", "cogniverse-backup-phoenix")
    assert cw is not None
    # Source volume on the workflow spec.
    vols = cw["spec"]["workflowSpec"].get("volumes", [])
    source = next((v for v in vols if v["name"] == "source"), None)
    assert source is not None, (
        f"phoenix workflow must define a ``source`` volume, got volumes={vols}"
    )
    assert "hostPath" in source, (
        f"hostStorage default uses hostPath for phoenix source, got {source}"
    )
    assert source["hostPath"]["path"] == "/host-data/phoenix"

    dump = next(
        t for t in cw["spec"]["workflowSpec"]["templates"] if t["name"] == "dump"
    )
    # Must NOT use kubectl-exec for distroless phoenix.
    assert "kubectl" not in dump["container"].get("image", ""), (
        f"phoenix dump container must not be kubectl image (no exec needed); "
        f"got {dump['container']['image']}"
    )
    # Must mount the source volume read-only.
    mounts = {m["name"]: m for m in dump["container"]["volumeMounts"]}
    assert mounts["source"]["readOnly"] is True
    assert mounts["source"]["mountPath"] == "/source"


def test_phoenix_backup_supports_pvc_mode_for_cloud():
    """Cloud operators set ``pvcName`` instead of ``hostPath``. The
    workflow then mounts the PVC directly (operator must arrange RWX or
    accept downtime / use VolumeSnapshot)."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
        "hostStorage.backup.services[0].name=phoenix",
        "hostStorage.backup.services[0].mode=volume-mount",
        "hostStorage.backup.services[0].pvcName=data-cogniverse-phoenix-0",
    )
    cw = _named(docs, "CronWorkflow", "cogniverse-backup-phoenix")
    assert cw is not None
    vols = cw["spec"]["workflowSpec"]["volumes"]
    source = next(v for v in vols if v["name"] == "source")
    assert "persistentVolumeClaim" in source
    assert source["persistentVolumeClaim"]["claimName"] == "data-cogniverse-phoenix-0"
    assert source["persistentVolumeClaim"]["readOnly"] is True


def test_role_only_renders_when_a_service_uses_kubectl_exec():
    """If every service uses volume-mount, no pods/exec privilege is
    needed → don't render the Role/RoleBinding (least privilege)."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
        # Override services to phoenix-only (volume-mount).
        "hostStorage.backup.services[0].name=phoenix",
        "hostStorage.backup.services[0].mode=volume-mount",
        "hostStorage.backup.services[0].hostPath=/host-data/phoenix",
    )
    role = _named(docs, "Role", "cogniverse-backup-exec")
    assert role is None, "no service uses kubectl-exec → pods/exec Role must not render"


def test_backup_services_list_is_configurable():
    """Operator can extend the backup set to any pod that ships ``tar``."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
        "hostStorage.backup.services[0].name=vespa",
        "hostStorage.backup.services[0].dataPath=/opt/vespa/var",
        "hostStorage.backup.services[0].podLabel=app.kubernetes.io/component=vespa",
        "hostStorage.backup.services[1].name=redis",
        "hostStorage.backup.services[1].dataPath=/data",
        "hostStorage.backup.services[1].podLabel=app.kubernetes.io/component=redis",
    )
    names = sorted(
        d["metadata"]["name"]
        for d in _by_kind(docs, "CronWorkflow")
        if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
        == "backup"
    )
    assert names == ["cogniverse-backup-redis", "cogniverse-backup-vespa"]


def test_backup_workflow_dump_step_targets_configured_data_path_and_label():
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
    )
    vespa = _named(docs, "CronWorkflow", "cogniverse-backup-vespa")
    assert vespa is not None

    def _dump_args(cw: dict) -> str:
        for tmpl in cw["spec"]["workflowSpec"]["templates"]:
            if tmpl["name"] == "dump":
                return tmpl["container"]["args"][-1]
        raise AssertionError("dump template not found")

    args = _dump_args(vespa)
    assert "/opt/vespa/var" in args
    assert "app.kubernetes.io/component=vespa" in args


def test_backup_inner_subkeys_default_when_only_enabled_is_set():
    """Regression: ``helm upgrade --reuse-values --set hostStorage.backup.enabled=true``
    overlays only the operator's --set on top of stored values; chart defaults
    from values.yaml don't reapply. The rendered CronWorkflow must therefore
    supply its own defaults for bucket/schedule/retainLast or it ships with
    null env vars that the upload step would reject."""
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        # Replace the entire backup block so chart defaults from values.yaml
        # do NOT apply to inner subkeys; only ``enabled`` + ``existingSecret``
        # are present, matching what ``--reuse-values`` from a release where
        # the backup block didn't exist would overlay.
        "--set-json",
        'hostStorage.backup={"enabled":true,"existingSecret":"cogniverse-minio"}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
    docs = [d for d in yaml.safe_load_all(result.stdout) if d is not None]
    cw = _named(docs, "CronWorkflow", "cogniverse-backup-vespa")
    assert cw is not None
    assert cw["spec"]["schedule"], "schedule must default, not be empty"
    upload = next(
        t for t in cw["spec"]["workflowSpec"]["templates"] if t["name"] == "upload"
    )
    env = {e["name"]: e.get("value") for e in upload["container"]["env"]}
    assert env["MINIO_BUCKET"], (
        f"MINIO_BUCKET must default, got {env['MINIO_BUCKET']!r}"
    )
    assert env["RETAIN_LAST"], f"RETAIN_LAST must default, got {env['RETAIN_LAST']!r}"


def test_pvc_inner_subkeys_default_when_only_enabled_is_set():
    """Regression: ``helm upgrade --reuse-values --set hfCache.persistence.enabled=true``
    overlays only the operator's --set on top of stored values; chart defaults
    from values.yaml don't reapply. The rendered PVCs must therefore supply
    their own defaults for accessMode/size or the apply fails with
    ``spec.accessModes: Unsupported value: ""``."""
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        # Replace the entire persistence block so chart defaults from
        # values.yaml do NOT apply to inner subkeys; only ``enabled`` is
        # present, matching what ``--reuse-values`` would overlay.
        "--set-json",
        'hfCache.persistence={"enabled":true}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
    docs = [d for d in yaml.safe_load_all(result.stdout) if d is not None]
    runtime_pvc = _named(
        docs, "PersistentVolumeClaim", "cogniverse-runtime-model-cache"
    )
    assert runtime_pvc is not None, "PVC must render with only enabled=true"
    assert runtime_pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert runtime_pvc["spec"]["resources"]["requests"]["storage"], (
        "size must default, not be empty"
    )


def test_render_survives_missing_persistence_block():
    """Regression: ``helm upgrade --reuse-values`` from a release predating
    these keys produces values where ``hfCache.persistence`` and
    ``hostStorage.backup`` are absent. Templates must use ``dig`` so the
    new feature stays disabled instead of crashing the upgrade with
    ``nil pointer evaluating interface {}.enabled``.

    This simulates that case via ``-f`` with a values file that omits both
    blocks (the chart's defaults from values.yaml are bypassed when the
    operator supplies their own values file)."""
    minimal = REPO_ROOT / "charts" / "cogniverse" / "values.k3s.yaml"
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "-f",
        str(minimal),
        # Mimic --reuse-values: erase the new keys by setting their parent
        # blocks to objects that omit them. Plain --set won't drop the
        # values.yaml defaults; --set-json with an explicit empty replaces
        # the parent.
        "--set-json",
        'hfCache={"enabled":false,"path":""}',
        "--set-json",
        'hostStorage={"enabled":false,"path":"/host-data"}',
        "--set",
        "argo-workflows.crds.install=false",
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"chart must render without crash when persistence/backup blocks "
        f"are absent (operator upgrading from older release):\n"
        f"{result.stderr}"
    )
    docs = [d for d in yaml.safe_load_all(result.stdout) if d is not None]
    # Newly added resources stay absent — feature is opt-in.
    assert (
        _named(docs, "PersistentVolumeClaim", "cogniverse-runtime-model-cache") is None
    )
    assert _named(docs, "Job", "cogniverse-hf-cache-minio-populate") is None
    backup_cws = [
        d
        for d in _by_kind(docs, "CronWorkflow")
        if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
        == "backup"
    ]
    assert backup_cws == []


def test_model_warm_uses_runtime_image_for_boto3_and_tar():
    """Regression: the init script downloads a tarball via boto3 and
    extracts it via tar; both deps must be in the image. The runtime
    image is the only one we control + ship that has all of boto3 +
    huggingface_hub + tar (the inference images may have only some)."""
    docs = _render(
        "hfCache.persistence.enabled=true",
        "hfCache.persistence.minio.enabled=true",
        "hfCache.persistence.minio.existingSecret=cogniverse-minio",
        "hfCache.persistence.minio.models[0]=hf-internal-testing/tiny-random-bert",
    )
    runtime_pod = _runtime_pod_spec(docs)
    runtime_image = runtime_pod["containers"][0]["image"]
    for name, spec in _inference_pod_specs(docs).items():
        warm = _container(spec, "model-warm")
        assert warm is not None
        assert warm["image"] == runtime_image, (
            f"inference {name}: model-warm must use the runtime image "
            f"({runtime_image}) which has boto3 + tar. Got {warm['image']}"
        )


def test_model_warm_script_extracts_tar_not_per_file_mirror():
    """Regression: the previous ``mc mirror`` design copied per-file
    and lost HF cache symlinks (snapshots/<sha>/<file> →
    blobs/<hash>), causing snapshot_download to consider the cache
    invalid and re-download from HF Hub. The new script downloads a
    single ``<repo>.tar`` and extracts it — tar preserves symlinks."""
    docs = _render(
        "hfCache.persistence.enabled=true",
        "hfCache.persistence.minio.enabled=true",
        "hfCache.persistence.minio.existingSecret=cogniverse-minio",
        "hfCache.persistence.minio.models[0]=hf-internal-testing/tiny-random-bert",
    )
    for name, spec in _inference_pod_specs(docs).items():
        warm = _container(spec, "model-warm")
        script = warm["args"][-1]
        assert '"tar", "-xf"' in script or "tar -xf" in script, (
            f"{name}: script must extract a tarball with ``tar -xf`` "
            f"(symlinks preserved). Got:\n{script}"
        )
        assert "mc mirror" not in script, (
            f"{name}: must not use ``mc mirror`` (loses symlinks). Got:\n{script}"
        )


def test_backup_upload_uses_bash_and_no_awk():
    """Regression: same image-content reason — minio/mc lacks ``awk``. The
    retention pruning must use bash + tail/sort, not awk, or the pruning
    pipeline silently fails (script still exits 0 because ``while read``
    consumes empty stdin) and snapshots accumulate forever."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=cogniverse-minio",
    )
    for cw_name in ("cogniverse-backup-vespa",):
        cw = _named(docs, "CronWorkflow", cw_name)
        assert cw is not None
        upload = next(
            t for t in cw["spec"]["workflowSpec"]["templates"] if t["name"] == "upload"
        )
        assert upload["container"]["command"] == [
            "bash",
            "-c",
        ], f"{cw_name} upload step must run under bash"
        script = upload["container"]["args"][-1]
        assert "awk " not in script, (
            f"{cw_name} retention must not call awk (mc image lacks it). Got:\n{script}"
        )


# ---------- MinIO durability (its own data, not the backup target) ----------


def _minio_data_volume(docs: list[dict]) -> dict:
    """Return the ``data`` volume from the MinIO Deployment."""
    for d in _by_kind(docs, "Deployment"):
        if d.get("metadata", {}).get("name") == "cogniverse-minio":
            for v in d["spec"]["template"]["spec"]["volumes"]:
                if v["name"] == "data":
                    return v
    raise AssertionError("MinIO Deployment data volume not found")


def test_minio_default_uses_pvc_backed_storage():
    """Default MinIO config provisions a PVC. Suitable for cloud / multi-node
    prod where a real CSI provider gives durability."""
    docs = _render()
    vol = _minio_data_volume(docs)
    assert "persistentVolumeClaim" in vol, f"default MinIO must use PVC, got {vol}"
    assert vol["persistentVolumeClaim"]["claimName"] == "cogniverse-minio"
    pvc = _named(docs, "PersistentVolumeClaim", "cogniverse-minio")
    assert pvc is not None, "default MinIO must provision a PVC"


def test_minio_hostpath_mode_skips_pvc_and_uses_bind_mount():
    """For laptop-dev clusters, MinIO IS the backup target. Setting
    ``minio.persistence.hostPath`` switches to a bind-mounted host
    directory so the data survives ``k3d cluster delete``. The PVC must
    NOT render in this mode (no point provisioning storage we don't use)."""
    docs = _render("minio.persistence.hostPath=/host-data/minio")
    vol = _minio_data_volume(docs)
    assert "hostPath" in vol, f"hostPath mode must mount a host dir, got {vol}"
    assert vol["hostPath"]["path"] == "/host-data/minio"
    assert vol["hostPath"]["type"] == "DirectoryOrCreate"
    pvc = _named(docs, "PersistentVolumeClaim", "cogniverse-minio")
    assert pvc is None, "hostPath mode must NOT provision a PVC"


def test_minio_persistence_block_missing_does_not_crash():
    """Regression: ``--reuse-values`` from a release predating the
    ``hostPath`` subkey leaves ``minio.persistence`` as a dict without
    that key. Template must default safely (PVC mode)."""
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        "--set-json",
        'minio.persistence={"size":"50Gi"}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"render failed:\n{result.stderr}"
    docs = [d for d in yaml.safe_load_all(result.stdout) if d is not None]
    pvc = _named(docs, "PersistentVolumeClaim", "cogniverse-minio")
    assert pvc is not None, "stripped persistence block must still default to PVC mode"
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "50Gi"


# ---------- Backup destination is configurable (single config knob) ----------


def _backup_upload_env(docs: list[dict], cw_name: str) -> dict:
    """Return the upload step's env vars as a name→value-or-secret dict."""
    cw = _named(docs, "CronWorkflow", cw_name)
    assert cw is not None, f"CronWorkflow {cw_name} not found"
    upload = next(
        t for t in cw["spec"]["workflowSpec"]["templates"] if t["name"] == "upload"
    )
    return {e["name"]: e for e in upload["container"]["env"]}


def test_backup_default_endpoint_targets_in_cluster_minio():
    """Default behaviour (no s3.endpoint override): backup targets the
    chart's own MinIO Deployment + secret. This is the dev-laptop config."""
    docs = _render(
        "hostStorage.backup.enabled=true",
    )
    env = _backup_upload_env(docs, "cogniverse-backup-vespa")
    assert env["MINIO_ENDPOINT"]["value"] == "http://cogniverse-minio:9000"
    assert (
        env["MINIO_ACCESS_KEY"]["valueFrom"]["secretKeyRef"]["name"]
        == "cogniverse-minio"
    )
    assert env["MINIO_ACCESS_KEY"]["valueFrom"]["secretKeyRef"]["key"] == "rootUser"


def test_backup_external_s3_endpoint_overrides_default():
    """Cloud config — single values block change points the same template
    + same code path at an external S3-compatible endpoint (R2/B2/AWS).
    No template edits, no chart rebuild."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.s3.endpoint=https://acct.r2.cloudflarestorage.com",
        "hostStorage.backup.s3.existingSecret=my-r2-creds",
    )
    env = _backup_upload_env(docs, "cogniverse-backup-vespa")
    assert env["MINIO_ENDPOINT"]["value"] == "https://acct.r2.cloudflarestorage.com"
    assert env["MINIO_ACCESS_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "my-r2-creds"
    assert env["MINIO_SECRET_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "my-r2-creds"


def test_backup_legacy_existing_secret_field_still_honored():
    """Backward-compat: ``hostStorage.backup.existingSecret`` (legacy
    top-level field) still wins over the chart's MinIO secret default,
    even when the new ``s3.existingSecret`` is empty. Operators who
    upgraded prior charts mustn't have their secret config broken."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=legacy-secret",
    )
    env = _backup_upload_env(docs, "cogniverse-backup-vespa")
    assert (
        env["MINIO_ACCESS_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "legacy-secret"
    )


def test_backup_s3_secret_takes_precedence_over_legacy():
    """When both are set, ``s3.existingSecret`` wins (it's the explicit
    new-style override)."""
    docs = _render(
        "hostStorage.backup.enabled=true",
        "hostStorage.backup.existingSecret=legacy-secret",
        "hostStorage.backup.s3.existingSecret=new-explicit-secret",
    )
    env = _backup_upload_env(docs, "cogniverse-backup-vespa")
    assert (
        env["MINIO_ACCESS_KEY"]["valueFrom"]["secretKeyRef"]["name"]
        == "new-explicit-secret"
    )


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
