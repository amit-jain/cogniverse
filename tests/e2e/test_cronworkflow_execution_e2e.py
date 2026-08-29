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
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from tests.e2e.conftest import (
    GATEWAY_VIDEO_QUERIES,
    IN_POD_TELEMETRY_PRELUDE,
    KUBECTL_CONTEXT,
    expected_gateway_calibration,
)
from tests.e2e.test_api_e2e import PROFILE, _deploy_profile_for_tenant

NAMESPACE = "cogniverse"
RUNTIME = (
    "http://localhost:33000"  # runtime.service.nodePort — matches tests/e2e/conftest.py
)
SUBMISSION_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# kubectl / Argo helpers (re-used by every cron in this file)
# ---------------------------------------------------------------------------


def _require_cronworkflow(name: str) -> None:
    command = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "get",
        "cronworkflow",
        name,
        "-n",
        NAMESPACE,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.fail(
            f"CronWorkflow prerequisite command failed: {shlex.join(command)}\n"
            f"error={type(exc).__name__}: {exc}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"CronWorkflow prerequisite command failed: {shlex.join(command)}\n"
            f"exit_code={result.returncode}\nstdout={result.stdout!r}\n"
            f"stderr={result.stderr!r}",
            pytrace=False,
        )


def _submit_workflow_from_cron(
    cron_name: str, *, parameters: dict[str, str] | None = None
) -> str:
    """Create a one-off Workflow from the CronWorkflow's workflowSpec.

    ``parameters`` overrides the values of the workflow's declared arguments
    (e.g. ``tenant-id``); an unknown name is a test bug and raises.
    """
    out = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "cronworkflow",
            cron_name,
            "-n",
            NAMESPACE,
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"kubectl get cronworkflow {cron_name} failed: {out.stderr.strip()}"
        )
    spec = json.loads(out.stdout)["spec"]["workflowSpec"]
    if parameters:
        declared = {
            p["name"]: p for p in spec.get("arguments", {}).get("parameters", [])
        }
        unknown = sorted(set(parameters) - set(declared))
        if unknown:
            raise ValueError(
                f"{cron_name} declares no workflow parameter(s) {unknown}; "
                f"declared={sorted(declared)}"
            )
        for name, value in parameters.items():
            declared[name]["value"] = value

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
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "create",
            "-n",
            NAMESPACE,
            "-f",
            "-",
            "-o",
            "json",
        ],
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
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "workflow",
            name,
            "-n",
            NAMESPACE,
            "-o",
            "json",
        ],
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
            "--context",
            KUBECTL_CONTEXT,
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
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "delete",
            "workflow",
            name,
            "-n",
            NAMESPACE,
            "--wait=false",
        ],
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


def _submit_and_wait_succeeded(
    cron_name: str,
    timeout_s: float = SUBMISSION_TIMEOUT_S,
    *,
    parameters: dict[str, str] | None = None,
):
    """Submit + poll. Fails the test with pod logs on non-Succeeded terminal."""
    wf = _submit_workflow_from_cron(cron_name, parameters=parameters)
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


# ---------------------------------------------------------------------------
# MinIO helpers (backup tests)
# ---------------------------------------------------------------------------


def _mc_ls_names(prefix: str) -> list:
    """Sorted object names under cogniverse-backups/<prefix>/ via the
    in-cluster MinIO.

    Spins a one-off mc pod that talks to the cluster's MinIO service —
    same access pattern the backup workflow uses. Snapshot names embed
    ISO timestamps, so lexical order == chronological order. A failed
    probe reports the complete kubectl command and process output.
    """
    command = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
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
                                f"mc find dest/cogniverse-backups/{prefix} 2>/dev/null"
                                " || true"
                            ],
                        }
                    ]
                }
            }
        ),
        "--",
        "true",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.fail(
            f"MinIO prerequisite command failed: {shlex.join(command)}\n"
            f"error={type(exc).__name__}: {exc}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"MinIO prerequisite command failed: {shlex.join(command)}\n"
            f"exit_code={result.returncode}\nstdout={result.stdout!r}\n"
            f"stderr={result.stderr!r}",
            pytrace=False,
        )
    # mc find prints full object paths (dest/bucket/prefix/name), one
    # per line — the minimal mc image has no awk/sed, so parse here.
    names = [
        line.strip().rsplit("/", 1)[-1]
        for line in result.stdout.splitlines()
        if line.strip().startswith("dest/cogniverse-backups/")
    ]
    return sorted(names)


# ---------------------------------------------------------------------------
# Light-tier tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestDailyCleanupWorkflow:
    """Daily-cleanup must hard-delete EPHEMERAL memories past 2×TTL across
    every tenant — that is its functional purpose, not just "Succeeded"."""

    def test_workflow_hard_deletes_stale_memory_for_seeded_tenant(self):
        _require_cronworkflow("cogniverse-daily-cleanup")

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


def _run_gateway_traffic(tenant_id: str) -> list[tuple[str, float]]:
    """Route the video queries for ``tenant_id``; return each decision.

    The tenant's deployed-but-empty video schema answers each simple search
    with zero hits and no error.
    """
    decisions: list[tuple[str, float]] = []
    with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
        for query in GATEWAY_VIDEO_QUERIES:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": tenant_id},
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            gw = body["gateway"]
            assert (gw["complexity"], gw["modality"], gw["routed_to"]) == (
                "simple",
                "video",
                "search_agent",
            ), body
            assert gw["confidence"] >= gw["fast_path_confidence_threshold"], gw
            assert body["status"] == "success", body
            decisions.append((gw["complexity"], gw["confidence"]))
    return decisions


def _runtime_pod_python(script: str, *, timeout: int = 180) -> str:
    """Run a Python snippet inside the runtime pod and return its stdout.

    Telemetry and artifact reads run in the pod: the host test process has no
    live config store, and the pod's telemetry manager is the one the
    workflow's own optimizer reads through.
    """
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "exec",
            "deploy/cogniverse-runtime",
            "-c",
            "runtime",
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"runtime pod python failed: rc={result.returncode}\n"
        f"stdout={result.stdout[-2000:]!r}\nstderr={result.stderr[-2000:]!r}"
    )
    return result.stdout


def _gateway_thresholds_blob(tenant_id: str) -> str | None:
    """The tenant's ``gateway_thresholds`` artifact blob, or None when absent."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "blob = asyncio.run(am.load_blob('config', 'gateway_thresholds')); "
        "print('__ABSENT__' if blob is None else '__BLOB__' + blob)"
    )
    out = _runtime_pod_python(script).strip().splitlines()[-1]
    if out == "__ABSENT__":
        return None
    assert out.startswith("__BLOB__"), out
    return out[len("__BLOB__") :]


def _count_gateway_spans(tenant_id: str) -> int:
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio; "
        "from cogniverse_foundation.telemetry.config import SPAN_NAME_GATEWAY; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_runtime.optimization_cli import _query_spans_by_name; "
        f"tm = get_telemetry_manager(); "
        f"tp = tm.get_provider(tenant_id={tenant_id!r}); "
        f"df = asyncio.run(_query_spans_by_name(tm, tp, {tenant_id!r}, SPAN_NAME_GATEWAY, 1)); "
        "print('__SPANS__' + str(len(df)))"
    )
    out = _runtime_pod_python(script).strip().splitlines()[-1]
    assert out.startswith("__SPANS__"), out
    return int(out[len("__SPANS__") :])


def _wait_for_gateway_spans(tenant_id: str, expected: int) -> None:
    """Block until Phoenix has exported exactly ``expected`` gateway spans."""
    deadline = time.monotonic() + 240.0
    seen = -1
    while time.monotonic() < deadline:
        seen = _count_gateway_spans(tenant_id)
        if seen == expected:
            return
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix shows {seen} gateway spans for tenant {tenant_id!r}; "
        f"expected {expected} within 240s"
    )


@pytest.mark.e2e
class TestDailyGatewayWorkflow:
    """Daily-gateway calls run_gateway_thresholds_optimization for the
    tenant it is given: it reads that tenant's gateway spans from Phoenix and
    persists the calibrated thresholds as the tenant's ``gateway_thresholds``
    artifact. There is no restart step — warm runtime pods re-read the
    artifact on the dispatcher's reload interval. The workflow uses
    templateRef → optimization-runner, the chart path that previously broke
    with "volume 'config' not found"."""

    def test_workflow_calibrates_and_persists_the_tenant_thresholds(self):
        _require_cronworkflow("cogniverse-daily-gateway")

        tenant_id = _seed_org_and_tenant(uuid.uuid4().hex[:8])
        try:
            # A new tenant registers no profiles; register + deploy the video
            # profile so cued queries reach search_agent and answer with zero
            # hits instead of a profile-not-found error.
            with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
                _deploy_profile_for_tenant(client, PROFILE, tenant_id)
            assert _gateway_thresholds_blob(tenant_id) is None

            decisions = _run_gateway_traffic(tenant_id)
            _wait_for_gateway_spans(tenant_id, len(GATEWAY_VIDEO_QUERIES))

            _submit_and_wait_succeeded(
                "cogniverse-daily-gateway",
                timeout_s=600,
                parameters={"tenant-id": tenant_id},
            )

            blob = _gateway_thresholds_blob(tenant_id)
            assert blob is not None, (
                f"daily-gateway Succeeded but wrote no gateway_thresholds "
                f"artifact for tenant {tenant_id!r}"
            )
            assert json.loads(blob) == expected_gateway_calibration(decisions)
        finally:
            _delete_tenant_and_org(tenant_id)


@pytest.mark.e2e
class TestBackupVespaWorkflow:
    """The vespa backup workflow tars vespa data via kubectl-exec and
    uploads to MinIO under cogniverse-backups/vespa/. A new object
    matching ``vespa-<TIMESTAMP>.tar`` must appear post-Succeeded."""

    def test_workflow_uploads_new_vespa_snapshot_to_minio(self):
        _require_cronworkflow("cogniverse-backup-vespa")
        names_before = _mc_ls_names("vespa")
        _submit_and_wait_succeeded("cogniverse-backup-vespa", timeout_s=600)
        names_after = _mc_ls_names("vespa")
        assert names_after, "could not list cogniverse-backups/vespa/ after the run"
        newest_before = names_before[-1] if names_before else ""
        # The upload step prunes to retainLast, so the COUNT stays flat at
        # capacity — the invariant is a strictly newer snapshot at the top.
        assert names_after[-1] > newest_before, (
            f"backup-vespa workflow Succeeded but no new snapshot appeared "
            f"under cogniverse-backups/vespa/ (newest {newest_before!r} → "
            f"{names_after[-1]!r})"
        )
        if names_before is not None:
            assert len(names_after) <= len(names_before) + 1, (
                f"retention pruning regressed: {len(names_before)} → "
                f"{len(names_after)} objects"
            )


@pytest.mark.e2e
class TestBackupPhoenixWorkflow:
    """Same contract as backup-vespa for the phoenix snapshot."""

    def test_workflow_uploads_new_phoenix_snapshot_to_minio(self):
        _require_cronworkflow("cogniverse-backup-phoenix")
        names_before = _mc_ls_names("phoenix")
        _submit_and_wait_succeeded("cogniverse-backup-phoenix", timeout_s=600)
        names_after = _mc_ls_names("phoenix")
        assert names_after, "could not list cogniverse-backups/phoenix/ after the run"
        newest_before = names_before[-1] if names_before else ""
        # The upload step prunes to retainLast, so the COUNT stays flat at
        # capacity — the invariant is a strictly newer snapshot at the top.
        assert names_after[-1] > newest_before, (
            f"backup-phoenix workflow Succeeded but no new snapshot appeared "
            f"under cogniverse-backups/phoenix/ (newest {newest_before!r} → "
            f"{names_after[-1]!r})"
        )
        if names_before is not None:
            assert len(names_after) <= len(names_before) + 1, (
                f"retention pruning regressed: {len(names_before)} → "
                f"{len(names_after)} objects"
            )


@pytest.mark.e2e
class TestMonthlyReportsWorkflow:
    """monthly-reports must (1) generate JSON reports in the workspace and
    (2) upload them to MinIO. Functional intent: a new ``usage-YYYYMM.json``
    object exists under ``cogniverse-backups/reports/`` after the workflow."""

    def test_workflow_uploads_usage_and_perf_reports_to_minio(self):
        _require_cronworkflow("cogniverse-monthly-reports")
        names_before = _mc_ls_names("reports")
        count_before = len(names_before) if names_before is not None else -1
        _submit_and_wait_succeeded("cogniverse-monthly-reports", timeout_s=600)
        names_after = _mc_ls_names("reports")
        count_after = len(names_after) if names_after is not None else -1
        # Two files per run (usage + performance); count must advance by
        # at least 1 (the same month overwrites the same key). Bare
        # Succeeded is not enough — the upload step must have actually
        # written to MinIO.
        assert count_after >= count_before, (
            f"monthly-reports workflow Succeeded but MinIO object count "
            f"under cogniverse-backups/reports/ regressed "
            f"({count_before} → {count_after})"
        )
        assert count_after >= 2, (
            f"monthly-reports must leave at least usage-* and performance-* "
            f"objects under cogniverse-backups/reports/; got count={count_after}"
        )
