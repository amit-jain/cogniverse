"""E2E coverage for CronWorkflow execution paths — light tier.

Each test in this file submits a one-off Workflow derived from the
chart's CronWorkflow ``workflowSpec`` against the live cluster, polls
for completion, and asserts both that the workflow reached
``Succeeded`` AND that its real side effect landed on the live
backend. "Succeeded" alone is too weak — these workflows exist for
specific functional reasons, and the test must prove each one
actually achieved its intent.

Light tier = workflows that complete in roughly 30s-90s and don't
require a live LM endpoint. Heavy tier (DSPy training, distillation,
synthetic data generation) lives in
``test_cronworkflow_execution_heavy_e2e.py`` behind the ``e2e_heavy``
marker.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest

NAMESPACE = "cogniverse"
RUNTIME = (
    "http://localhost:28000"  # runtime.service.nodePort — matches tests/e2e/conftest.py
)
SUBMISSION_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# kubectl / Argo helpers (re-used by every cron in this file)
# ---------------------------------------------------------------------------


def _kubectl_available() -> bool:
    return shutil.which("kubectl") is not None


def _cronworkflow_exists(name: str) -> bool:
    if not _kubectl_available():
        return False
    result = subprocess.run(
        ["kubectl", "get", "cronworkflow", name, "-n", NAMESPACE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def _submit_workflow_from_cron(cron_name: str) -> str:
    """Create a one-off Workflow from the CronWorkflow's workflowSpec."""
    out = subprocess.run(
        ["kubectl", "get", "cronworkflow", cron_name, "-n", NAMESPACE, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"kubectl get cronworkflow {cron_name} failed: {out.stderr.strip()}"
        )
    spec = json.loads(out.stdout)["spec"]["workflowSpec"]

    workflow = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"{cron_name}-e2e-",
            "namespace": NAMESPACE,
            "labels": {"cogniverse.test/source": "e2e-cronworkflow-execution"},
        },
        "spec": spec,
    }
    created = subprocess.run(
        ["kubectl", "create", "-n", NAMESPACE, "-f", "-", "-o", "json"],
        input=json.dumps(workflow),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if created.returncode != 0:
        raise RuntimeError(
            f"kubectl create workflow from {cron_name} failed: {created.stderr.strip()}"
        )
    return json.loads(created.stdout)["metadata"]["name"]


def _workflow_status(name: str) -> dict:
    out = subprocess.run(
        ["kubectl", "get", "workflow", name, "-n", NAMESPACE, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        return {}
    return json.loads(out.stdout).get("status", {}) or {}


def _workflow_pod_logs(workflow_name: str) -> str:
    out = subprocess.run(
        [
            "kubectl",
            "logs",
            "-n",
            NAMESPACE,
            "-l",
            f"workflows.argoproj.io/workflow={workflow_name}",
            "--all-containers",
            "--tail=500",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return out.stdout or out.stderr or ""


def _delete_workflow(name: str) -> None:
    subprocess.run(
        ["kubectl", "delete", "workflow", name, "-n", NAMESPACE, "--wait=false"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _wait_for_workflow_terminal(
    name: str, timeout_s: float = SUBMISSION_TIMEOUT_S
) -> dict:
    deadline = time.monotonic() + timeout_s
    last_phase = "Unknown"
    while time.monotonic() < deadline:
        status = _workflow_status(name)
        phase = status.get("phase") or "Pending"
        if phase != last_phase:
            print(f"workflow {name}: phase={phase}")
            last_phase = phase
        if phase in {"Succeeded", "Failed", "Error"}:
            return status
        time.sleep(POLL_INTERVAL_S)
    return _workflow_status(name)


def _submit_and_wait_succeeded(cron_name: str, timeout_s: float = SUBMISSION_TIMEOUT_S):
    """Submit + poll. Fails the test with pod logs on non-Succeeded terminal."""
    wf = _submit_workflow_from_cron(cron_name)
    try:
        status = _wait_for_workflow_terminal(wf, timeout_s=timeout_s)
        phase = status.get("phase") or "Unknown"
        if phase != "Succeeded":
            logs = _workflow_pod_logs(wf)
            pytest.fail(
                f"Workflow {wf} (from {cron_name}) phase={phase!r}, expected "
                f"'Succeeded'.\nstatus.message={status.get('message')!r}\n"
                f"--- pod logs (tail 500) ---\n{logs[-4000:]}\n--- end logs ---"
            )
        return wf
    finally:
        _delete_workflow(wf)


# ---------------------------------------------------------------------------
# Runtime/Vespa side-effect helpers
# ---------------------------------------------------------------------------


def _seed_org_and_tenant(unique_suffix: str) -> str:
    """Create real org + tenant via the runtime's admin API.

    Returns the tenant_full_id. The daily-cleanup workflow enumerates
    every tenant in every org via the live tenant_manager helpers, so
    the seeded tenant becomes a real participant in the sweep.
    """
    org_id = f"cron_e2e_org_{unique_suffix}"
    tenant_id = f"{org_id}:t1"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            f"{RUNTIME}/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": f"cron-e2e-{unique_suffix}",
                "created_by": "e2e",
            },
        )
        # 409 = already exists from a prior aborted run — acceptable.
        assert r.status_code in (200, 409), r.text
        r = client.post(
            f"{RUNTIME}/admin/tenants",
            json={"tenant_id": tenant_id, "created_by": "e2e"},
        )
        assert r.status_code in (200, 409), r.text
    return tenant_id


def _delete_tenant_and_org(tenant_full_id: str) -> None:
    org_id = tenant_full_id.split(":", 1)[0]
    with httpx.Client(timeout=120.0) as client:
        try:
            client.delete(f"{RUNTIME}/admin/tenants/{tenant_full_id}")
        except httpx.HTTPError:
            pass
        try:
            client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
        except httpx.HTTPError:
            pass


def _add_aged_memory(
    tenant_full_id: str, kind: str, age_days: float, content: str
) -> str:
    """POST /admin/tenant/{t}/memories with kind + backdated created_at."""
    meta: dict = {}
    if age_days > 0:
        meta["created_at"] = (
            datetime.now(timezone.utc) - timedelta(days=age_days)
        ).isoformat()
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            f"{RUNTIME}/admin/tenant/{tenant_full_id}/memories",
            json={"text": content, "kind": kind, "metadata": meta},
        )
        assert r.status_code == 200, r.text
    return r.json()["id"]


def _resolve_memory(tenant_full_id: str, mid: str) -> dict | None:
    """List memories for the tenant and return the one matching mid, or None."""
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{RUNTIME}/admin/tenant/{tenant_full_id}/memories")
        if r.status_code != 200:
            return None
        for m in r.json().get("memories") or []:
            if m.get("id") == mid:
                return m
        return None


def _poll_resolve(
    tenant_full_id: str, mid: str, *, expect_present: bool, timeout_s: float = 30.0
) -> dict | None:
    """Poll _resolve_memory until the desired condition is observed.

    Mem0 writes propagate through Vespa with eventual consistency on
    the /search/ list path — a freshly POSTed memory may take a few
    seconds to surface, and a freshly hard-deleted memory may take a
    few seconds to disappear. Polling either direction avoids racing
    that propagation.
    """
    deadline = time.monotonic() + timeout_s
    last = _resolve_memory(tenant_full_id, mid)
    while time.monotonic() < deadline:
        present = last is not None
        if present == expect_present:
            return last
        time.sleep(2.0)
        last = _resolve_memory(tenant_full_id, mid)
    return last


def _runtime_pod_restart_count() -> int:
    """Read the runtime Deployment's pod restart-count for rollout detection."""
    out = subprocess.run(
        [
            "kubectl",
            "get",
            "deployment",
            "-n",
            NAMESPACE,
            "-l",
            "app.kubernetes.io/component=runtime",
            "-o",
            "jsonpath={.items[*].status.observedGeneration}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(out.stdout.strip() or "0")


# ---------------------------------------------------------------------------
# MinIO helpers (backup tests)
# ---------------------------------------------------------------------------


def _mc_ls_count(prefix: str) -> int:
    """List objects under cogniverse-backups/<prefix>/ via the in-cluster MinIO.

    Spins a one-off mc pod that talks to the cluster's MinIO service —
    same access pattern the backup workflow uses. Returns -1 on any
    failure (caller treats as "skip detection of pre-state").
    """
    result = subprocess.run(
        [
            "kubectl",
            "run",
            f"mc-probe-{uuid.uuid4().hex[:8]}",
            "-n",
            NAMESPACE,
            "--rm",
            "-i",
            "--restart=Never",
            "--image=minio/mc:latest",
            "--overrides",
            json.dumps(
                {
                    "spec": {
                        "containers": [
                            {
                                "name": "mc",
                                "image": "minio/mc:latest",
                                "env": [
                                    {
                                        "name": "ACCESS",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "cogniverse-minio",
                                                "key": "rootUser",
                                            }
                                        },
                                    },
                                    {
                                        "name": "SECRET",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "cogniverse-minio",
                                                "key": "rootPassword",
                                            }
                                        },
                                    },
                                ],
                                "command": ["sh", "-c"],
                                "args": [
                                    'mc alias set dest http://cogniverse-minio:9000 "$ACCESS" "$SECRET" >/dev/null 2>&1 && '
                                    f"mc ls dest/cogniverse-backups/{prefix}/ 2>/dev/null | wc -l"
                                ],
                            }
                        ]
                    }
                }
            ),
            "--",
            "true",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        # mc ls output is the last non-empty line; pod attach prefixes some chatter
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if line.isdigit():
                return int(line)
        return -1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Light-tier tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestDailyCleanupWorkflow:
    """Daily-cleanup must hard-delete EPHEMERAL memories past 2×TTL across
    every tenant — that is its functional purpose, not just "Succeeded"."""

    def test_workflow_hard_deletes_stale_memory_for_seeded_tenant(self):
        if not _cronworkflow_exists("cogniverse-daily-cleanup"):
            pytest.skip("cogniverse-daily-cleanup CronWorkflow not deployed")

        suffix = uuid.uuid4().hex[:8]
        tenant_id = _seed_org_and_tenant(suffix)
        try:
            # Plant one hard-deletable (40d > 28d) and one permanent control.
            stale_id = _add_aged_memory(
                tenant_id, "conversation_turn", 40.0, "stale-victim"
            )
            permanent_id = _add_aged_memory(
                tenant_id, "tenant_instruction", 999.0, "rule-stays-forever"
            )

            # Pre-state: both visible. Poll the list endpoint — Mem0 +
            # Vespa /search/ is eventually consistent after POST.
            assert (
                _poll_resolve(tenant_id, stale_id, expect_present=True) is not None
            ), "precondition: stale memory must be queryable before cleanup runs"
            assert (
                _poll_resolve(tenant_id, permanent_id, expect_present=True) is not None
            )

            _submit_and_wait_succeeded("cogniverse-daily-cleanup", timeout_s=600)

            # Functional outcome: the 40d ephemeral memory is GONE.
            # Same eventual-consistency caveat for the delete side.
            assert _poll_resolve(tenant_id, stale_id, expect_present=False) is None, (
                f"daily-cleanup workflow Succeeded but the 40d-old "
                f"conversation_turn ({stale_id}) is still resolvable — "
                f"workflow ran but its functional intent did not land"
            )
            # And the PERMANENT memory survives.
            assert _resolve_memory(tenant_id, permanent_id) is not None, (
                "daily-cleanup must not touch PERMANENT kinds; "
                "tenant_instruction was wiped"
            )
        finally:
            _delete_tenant_and_org(tenant_id)


@pytest.mark.e2e
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestDailyGatewayWorkflow:
    """Daily-gateway must (1) call run_gateway_thresholds_optimization
    against Phoenix spans and (2) trigger a runtime rollout. The
    workflow uses templateRef → optimization-runner, which is the
    chart path that previously broke with "volume 'config' not found"."""

    def test_workflow_runs_to_succeeded_and_triggers_runtime_rollout(self):
        if not _cronworkflow_exists("cogniverse-daily-gateway"):
            pytest.skip("cogniverse-daily-gateway CronWorkflow not deployed")

        # Pre: capture rollout generation. Post: it must have bumped if
        # the restart-deployment step ran — and that step is
        # sequenced AFTER optimize-gateway in the pipeline, so an
        # advance proves both steps Succeeded against the live cluster.
        gen_before = _runtime_pod_restart_count()

        _submit_and_wait_succeeded("cogniverse-daily-gateway", timeout_s=600)

        # Functional outcome: runtime deployment was rolled. The
        # restart-deployment step needs RBAC to patch deployments + a
        # successful optimize-gateway step upstream; the observed
        # generation bump proves both. This is a stronger contract than
        # reading pod logs (Argo gc's completed pods quickly, so log
        # scraping races the workflow controller).
        #
        # ``kubectl rollout restart`` exits immediately after patching
        # the deployment spec; the deployment controller updates
        # observedGeneration asynchronously. Poll for the bump rather
        # than reading once and racing the controller.
        deadline = time.monotonic() + 120.0
        gen_after = gen_before
        while time.monotonic() < deadline:
            gen_after = _runtime_pod_restart_count()
            if gen_after > gen_before:
                break
            time.sleep(2.0)
        assert gen_after > gen_before, (
            f"daily-gateway workflow Succeeded but the runtime deployment "
            f"observedGeneration did not advance ({gen_before} → {gen_after}) "
            f"within 120s; the restart-deployment step must have run for "
            f"thresholds to take effect"
        )


@pytest.mark.e2e
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestBackupVespaWorkflow:
    """The vespa backup workflow tars vespa data via kubectl-exec and
    uploads to MinIO under cogniverse-backups/vespa/. A new object
    matching ``vespa-<TIMESTAMP>.tar`` must appear post-Succeeded."""

    def test_workflow_uploads_new_vespa_snapshot_to_minio(self):
        if not _cronworkflow_exists("cogniverse-backup-vespa"):
            pytest.skip(
                "cogniverse-backup-vespa CronWorkflow not deployed "
                "(hostStorage.backup.enabled defaults to false)"
            )
        count_before = _mc_ls_count("vespa")
        _submit_and_wait_succeeded("cogniverse-backup-vespa", timeout_s=600)
        count_after = _mc_ls_count("vespa")
        assert count_after > count_before, (
            f"backup-vespa workflow Succeeded but MinIO object count under "
            f"cogniverse-backups/vespa/ did not increase "
            f"({count_before} → {count_after})"
        )


@pytest.mark.e2e
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestBackupPhoenixWorkflow:
    """Same contract as backup-vespa for the phoenix snapshot."""

    def test_workflow_uploads_new_phoenix_snapshot_to_minio(self):
        if not _cronworkflow_exists("cogniverse-backup-phoenix"):
            pytest.skip(
                "cogniverse-backup-phoenix CronWorkflow not deployed "
                "(hostStorage.backup.enabled defaults to false)"
            )
        count_before = _mc_ls_count("phoenix")
        _submit_and_wait_succeeded("cogniverse-backup-phoenix", timeout_s=600)
        count_after = _mc_ls_count("phoenix")
        assert count_after > count_before, (
            f"backup-phoenix workflow Succeeded but MinIO object count under "
            f"cogniverse-backups/phoenix/ did not increase "
            f"({count_before} → {count_after})"
        )
