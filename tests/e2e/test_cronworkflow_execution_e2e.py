"""E2E coverage for CronWorkflow execution paths.

Catches regressions where a CronWorkflow's workflowSpec is syntactically
valid (chart renders, kubectl accepts) but fails at runtime — e.g.,
argparse exit 2 because a required flag is missing (the daily-cleanup
regression that prompted task #125), or "volume 'config' not found in
workflow spec" because templateRef volumes were not inherited (the
daily-gateway regression).

Each test submits a one-off Workflow derived from the CronWorkflow's
``workflowSpec`` (CronWorkflow suspension does not block this — only
the scheduler) and waits for the Workflow to reach a terminal phase.
Anything other than ``Succeeded`` is a test failure, with the pod logs
attached for diagnosis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

import pytest

NAMESPACE = "cogniverse"
SUBMISSION_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 5.0


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
    """Create a one-off Workflow from the CronWorkflow's workflowSpec.

    Returns the submitted Workflow's metadata.name. Raises ``RuntimeError``
    on any kubectl failure — the e2e suite reports those as test errors,
    not flakes.
    """
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
    """Concatenate logs from every pod the workflow created.

    Argo labels its pods with ``workflows.argoproj.io/workflow=<name>`` so
    a single ``kubectl logs -l`` call picks them all up. Best-effort —
    if logs are unavailable (pods cleaned up too fast) we return an empty
    string rather than crashing the assertion.
    """
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


def _wait_for_workflow_terminal(name: str, timeout_s: float = SUBMISSION_TIMEOUT_S):
    """Poll until phase is in {Succeeded, Failed, Error}. Returns status dict."""
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


@pytest.mark.e2e
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestCronWorkflowExecution:
    """Each cron's workflowSpec submitted as a one-off Workflow must Succeed.

    These cover the runtime failure mode that argparse-level unit tests
    can't reach: the actual container, image, and chart-rendered spec
    running on the live cluster.
    """

    def _run_and_assert_succeeded(self, cron_name: str) -> None:
        if not _cronworkflow_exists(cron_name):
            pytest.skip(f"CronWorkflow {cron_name} not deployed on this cluster")
        wf_name = _submit_workflow_from_cron(cron_name)
        try:
            status = _wait_for_workflow_terminal(wf_name)
            phase = status.get("phase") or "Unknown"
            if phase != "Succeeded":
                logs = _workflow_pod_logs(wf_name)
                pytest.fail(
                    f"Workflow {wf_name} (from {cron_name}) reached phase="
                    f"{phase!r}, expected 'Succeeded'.\n"
                    f"status.message={status.get('message')!r}\n"
                    f"--- pod logs (tail 500) ---\n{logs[-4000:]}\n"
                    f"--- end logs ---"
                )
        finally:
            _delete_workflow(wf_name)

    def test_daily_cleanup_workflow_runs_to_succeeded(self):
        """Regression test for the argparse exit-2 bug.

        Pre-fix, ``optimization_cli --mode cleanup`` exited 2 because
        ``--tenant-id`` was ``required=True`` and the chart-rendered
        args don't pass one. The workflow pod's container therefore
        exited non-zero and the Workflow reached Failed. After the fix
        in ``optimization_cli`` (cleanup mode now iterates globally
        when --tenant-id is omitted), the Workflow must reach Succeeded.
        """
        self._run_and_assert_succeeded("cogniverse-daily-cleanup")
