"""E2E test for dashboard-triggered Argo optimization.

Exercises the full path from runtime → Argo → Workflow pod using the
live k3d stack. No mocks anywhere — the runtime actually submits to
argo-server, Argo actually creates the Workflow, and we verify via the
Argo API that the Workflow landed with the expected spec.

Requires live stack via ``cogniverse up``:
- Runtime at localhost:28000 (with ARGO_API_URL wired)
- kubectl context: k3d-cogniverse
- Argo Workflows installed in the `cogniverse` namespace

Marked ``slow`` because real Argo submission + status poll is network-heavy
and the test waits for phase transitions.
"""

import subprocess
import time

import httpx
import pytest

from tests.e2e.conftest import (
    TENANT_ID,
    skip_if_no_runtime,
)

pytestmark = pytest.mark.slow

KUBECTL_CONTEXT = "k3d-cogniverse"
NAMESPACE = "cogniverse"
RUNTIME = "http://localhost:28000"


def _kubectl_get_workflow(name: str) -> dict | None:
    """Return the Argo Workflow resource as a dict, or None if missing.

    Uses kubectl rather than a separate Argo-API port-forward because the
    k3d harness already exposes the cluster via the local kubecontext.
    """
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "workflow",
            name,
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    import json

    return json.loads(result.stdout)


def _argo_available() -> bool:
    """Detect whether the k3d cluster has Argo installed."""
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "crd",
            "workflows.argoproj.io",
            "--ignore-not-found",
            "-o",
            "name",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


skip_if_no_argo = pytest.mark.skipif(
    not _argo_available(),
    reason="Argo CRDs not installed in k3d cluster — run with --set argo.enabled=true",
)


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_argo
class TestManualOptimizationE2E:
    """End-to-end: POST /admin/tenant/{id}/optimize creates a real Argo Workflow."""

    def test_submit_creates_workflow_with_correct_spec(self):
        """Runtime submits a real Workflow to Argo; kubectl confirms it
        exists with the expected mode/tenant/image/env/TTL."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                f"/admin/tenant/{TENANT_ID}/optimize",
                json={"mode": "gateway-thresholds"},
            )

        assert resp.status_code == 200, (
            f"Runtime rejected optimize submit: {resp.status_code} {resp.text}"
        )
        data = resp.json()
        assert data["mode"] == "gateway-thresholds"
        assert data["namespace"] == NAMESPACE
        workflow_name = data["workflow_name"]
        assert workflow_name.startswith("manual-optimize-gateway-thresholds-")

        # Argo assigns a suffix; the runtime must return the fully-resolved name.
        wf = _kubectl_get_workflow(workflow_name)
        assert wf is not None, (
            f"Workflow {workflow_name} not found in cluster after submit"
        )

        labels = wf["metadata"]["labels"]
        assert labels["cogniverse.ai/trigger"] == "manual"
        assert labels["cogniverse.ai/mode"] == "gateway-thresholds"
        # K8s labels disallow colons; the manifest builder sanitizes tenant IDs
        # to a label-safe form while the raw tenant flows through the CLI arg.
        expected_label = TENANT_ID.replace(":", "-")
        assert labels["cogniverse.ai/tenant"] == expected_label

        ttl = wf["spec"]["ttlStrategy"]
        assert ttl["secondsAfterCompletion"] == 3600
        assert ttl["secondsAfterSuccess"] == 3600
        assert ttl["secondsAfterFailure"] == 3600

        # Workflow delegates the container spec to the chart-installed
        # WorkflowTemplate; the Workflow resource itself only carries the
        # templateRef + arguments. Container image / env / command live
        # under the WorkflowTemplate resource.
        assert wf["spec"]["workflowTemplateRef"]["name"] == (
            "cogniverse-optimization-runner"
        )
        params = {p["name"]: p["value"] for p in wf["spec"]["arguments"]["parameters"]}
        assert params["mode"] == "gateway-thresholds"
        assert params["tenant-id"] == TENANT_ID
        assert params["lookback-hours"] == "48"

    def test_status_endpoint_reflects_argo_phase(self):
        """After submit, GET status returns a real phase from Argo within
        a reasonable window (≤60s until scheduler picks it up)."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            submit = client.post(
                f"/admin/tenant/{TENANT_ID}/optimize",
                json={"mode": "gateway-thresholds"},
            )
        assert submit.status_code == 200, submit.text
        data = submit.json()
        status_url = data["status_url"]

        # Poll until Argo assigns a phase. Pending → Running → (don't wait
        # for Succeeded — that requires real span data and can take minutes).
        deadline = time.monotonic() + 60.0
        last_body: dict = {}
        with httpx.Client(base_url=RUNTIME, timeout=15.0) as client:
            while time.monotonic() < deadline:
                status_resp = client.get(status_url)
                assert status_resp.status_code == 200, (
                    f"Status endpoint failed: {status_resp.status_code} "
                    f"{status_resp.text}"
                )
                last_body = status_resp.json()
                if last_body.get("phase") in {
                    "Pending",
                    "Running",
                    "Succeeded",
                    "Failed",
                    "Error",
                }:
                    break
                time.sleep(2.0)

        assert last_body.get("phase") in {
            "Pending",
            "Running",
            "Succeeded",
            "Failed",
            "Error",
        }, (
            f"Workflow never transitioned to a known Argo phase; "
            f"last status body: {last_body}"
        )

    def test_rejects_unknown_mode(self):
        """Unknown mode returns 400 before hitting Argo."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                f"/admin/tenant/{TENANT_ID}/optimize",
                json={"mode": "bogus-mode"},
            )
        assert resp.status_code == 400
        assert "bogus-mode" in resp.json()["detail"]

    def test_status_404_for_missing_workflow(self):
        """GET status for a workflow that doesn't exist returns 404."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get(
                f"/admin/tenant/{TENANT_ID}/optimize/runs/nonexistent-workflow-xyz"
            )
        assert resp.status_code == 404


def _wait_for_workflow_phase(
    name: str,
    target_phases: set[str],
    deadline_seconds: float = 300.0,
) -> dict:
    """Block until the Workflow reaches one of ``target_phases`` or the
    deadline expires. Returns the full Workflow resource. Raises if the
    Workflow disappears or the deadline passes.
    """
    deadline = time.monotonic() + deadline_seconds
    last = None
    while time.monotonic() < deadline:
        wf = _kubectl_get_workflow(name)
        if wf is None:
            raise AssertionError(f"Workflow {name} disappeared during poll")
        last = wf
        phase = wf.get("status", {}).get("phase", "")
        if phase in target_phases:
            return wf
        time.sleep(5.0)
    raise AssertionError(
        f"Workflow {name} did not reach {target_phases} within "
        f"{deadline_seconds}s. Last phase: "
        f"{last.get('status', {}).get('phase') if last else 'unknown'}"
    )


def _kubectl_main_container_log(workflow_name: str) -> str:
    """Return the ``main`` container's stdout for the Workflow's single pod.
    Argo names the pod the same as the Workflow in the single-template case.
    """
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "logs",
            workflow_name,
            "-c",
            "main",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_argo
@pytest.mark.slow
class TestManualOptimizationDeepE2E:
    """Deep integration: submit a Workflow, wait for it to run to completion
    against real Phoenix/Vespa/Ollama, and assert on the real artifact.

    Runs optimization_cli --mode gateway-thresholds end-to-end inside the
    k3d cluster. The optimizer queries Phoenix for real gateway spans,
    computes a calibrated threshold, and uploads the result as a Phoenix
    dataset. This test confirms the full pipeline delivers a well-formed
    artifact — not just that the Workflow was accepted by Argo.

    Takes several minutes — the optimization CLI cold-starts the runtime
    image, connects to Phoenix/Vespa, and does real computation.
    """

    def test_gateway_thresholds_workflow_runs_to_success_and_produces_artifact(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            submit = client.post(
                f"/admin/tenant/{TENANT_ID}/optimize",
                json={"mode": "gateway-thresholds"},
            )
        assert submit.status_code == 200, submit.text
        workflow_name = submit.json()["workflow_name"]

        # 10 minutes — cold pull + Phoenix connect + real computation.
        wf = _wait_for_workflow_phase(
            workflow_name,
            target_phases={"Succeeded", "Failed", "Error"},
            deadline_seconds=600.0,
        )
        phase = wf["status"]["phase"]
        assert phase == "Succeeded", (
            f"Workflow did not succeed (phase={phase}). "
            f"Message: {wf['status'].get('message')}. "
            f"Logs:\n{_kubectl_main_container_log(workflow_name)[-2000:]}"
        )

        # Assert the optimization actually ran against real data and produced
        # a usable artifact. The CLI prints a single JSON document at the end
        # of the main container log; parse and assert on its content.
        log = _kubectl_main_container_log(workflow_name)
        # Extract the JSON block (last `{...}` in the log).
        import json

        start = log.rfind('{\n  "status":')
        assert start >= 0, (
            f"Could not find optimization result JSON in log. Log tail:\n{log[-2000:]}"
        )
        # Parse forward until matching close brace.
        depth = 0
        end = start
        for i in range(start, len(log)):
            ch = log[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        payload = json.loads(log[start:end])

        assert payload["status"] == "success", payload
        assert isinstance(payload["spans_found"], int) and payload["spans_found"] > 0, (
            f"Optimizer found no gateway spans — Phoenix integration broken "
            f"or no tenant traffic. Payload: {payload}"
        )
        assert isinstance(payload["artifact_id"], str) and payload["artifact_id"], (
            f"No artifact_id returned. Payload: {payload}"
        )

        thresholds = payload["thresholds"]
        fast = thresholds["fast_path_confidence_threshold"]
        gliner = thresholds["gliner_threshold"]
        # Hard numerical bounds — the optimizer must produce values in the
        # valid range. Outside this range means the calibration algorithm
        # or the real-data inputs are broken.
        assert 0.0 < fast <= 1.0, f"fast_path_confidence_threshold={fast} out of range"
        assert 0.0 < gliner <= 1.0, f"gliner_threshold={gliner} out of range"

        analysis = thresholds["analysis"]
        assert analysis["total_spans"] == payload["spans_found"]
        assert (
            analysis["simple_count"] + analysis["complex_count"]
            == (analysis["total_spans"])
        ), (
            f"span split inconsistent: simple={analysis['simple_count']}, "
            f"complex={analysis['complex_count']}, total={analysis['total_spans']}"
        )
        assert 0.0 <= analysis["mean_confidence"] <= 1.0

        # Verify the artifact was persisted to Phoenix — fetch it by ID and
        # confirm it's a real dataset the runtime can consume.
        artifact_id = payload["artifact_id"]
        phoenix_endpoint = "http://localhost:26006"
        with httpx.Client(base_url=phoenix_endpoint, timeout=30.0) as phx:
            # Phoenix v1 datasets endpoint: GET /v1/datasets/{id}
            dataset_resp = phx.get(f"/v1/datasets/{artifact_id}")
            assert dataset_resp.status_code == 200, (
                f"Phoenix dataset {artifact_id} not retrievable: "
                f"{dataset_resp.status_code} {dataset_resp.text[:500]}"
            )
            dataset_body = dataset_resp.json()
            assert "data" in dataset_body, dataset_body
            assert dataset_body["data"]["id"] == artifact_id

    def test_workflow_mode_runs_to_success_against_real_orchestration_spans(self):
        """``--mode workflow`` is the second dashboard-exposed mode; it reads
        orchestration spans from Phoenix via OrchestrationEvaluator and
        persists extracted workflow templates + agent profiles as artifacts.

        This test proves the /optimize pipeline works end-to-end for a mode
        other than gateway-thresholds — a different span source, a different
        CLI code path (``run_workflow_optimization``), and a different
        artifact kind (``demonstrations`` instead of ``config``). A bug that
        only affects one of the two modes would be caught here.

        Two terminal states count as success for this test:
        - ``no_data``: the optimizer queried Phoenix but found no orchestration
          spans. The pipeline ran, Phoenix was reachable, serialization worked.
        - ``success``: the optimizer extracted workflow executions and
          persisted demonstrations. Argo, Phoenix, ArtifactManager all wired.
        Either way, the Workflow pod reached phase=Succeeded — what must NOT
        happen is a crash, a timeout, or Argo marking the Workflow Error."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            submit = client.post(
                f"/admin/tenant/{TENANT_ID}/optimize",
                json={"mode": "workflow"},
            )
        assert submit.status_code == 200, submit.text
        data = submit.json()
        assert data["mode"] == "workflow"
        workflow_name = data["workflow_name"]
        assert workflow_name.startswith("manual-optimize-workflow-"), (
            f"Workflow name must reflect the mode, got: {workflow_name!r}"
        )

        wf = _wait_for_workflow_phase(
            workflow_name,
            target_phases={"Succeeded", "Failed", "Error"},
            deadline_seconds=600.0,
        )
        phase = wf["status"]["phase"]
        log = _kubectl_main_container_log(workflow_name)
        assert phase == "Succeeded", (
            f"workflow-mode Workflow did not succeed (phase={phase}). "
            f"Message: {wf['status'].get('message')}. "
            f"Logs:\n{log[-2000:]}"
        )

        # The CLI logs a single terminal status in the main container output.
        # Look for either ``no_data`` or ``success`` — both are valid.
        assert ('"status": "no_data"' in log) or ('"status": "success"' in log), (
            "workflow-mode CLI must emit a terminal status dict to stdout "
            "(either no_data or success). Log tail:\n"
            f"{log[-2500:]}"
        )

        # If success, the CLI must have persisted at least one artifact —
        # confirm by the log line from ArtifactManager.save_demonstrations.
        if '"status": "success"' in log:
            assert "save_demonstrations" in log or "Saved " in log, (
                "workflow-mode success must persist demonstrations via "
                "ArtifactManager. Log tail:\n"
                f"{log[-2500:]}"
            )
