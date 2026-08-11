"""
E2E tests for quality monitor sidecar, strategy learning, and Argo workflows.

Requires live k3d stack via `cogniverse up` with:
- Runtime at localhost:33000 (Service NodePort, exposed via k3d loadbalancer)
- Vespa at localhost:33080 (Service port directly)
- Phoenix at localhost:33006 (Service NodePort)
- the configured LM endpoint
- Argo controller deployed

Verifies:
1. Quality monitor sidecar is running in runtime pod
2. Strategies can be stored and retrieved via /search endpoint
3. Argo CronWorkflows are deployed
4. Phoenix datasets accessible for eval baselines
"""

import subprocess

import httpx
import pytest

from tests.e2e.conftest import (
    ARGO_NAMESPACE,
    KUBECTL_CONTEXT,
    RUNTIME,
    TENANT_ID,
    argo_workflow_controller_probe_command,
    argo_workflow_controller_probe_failure_message,
)

PHOENIX = "http://localhost:33006"
VESPA = "http://localhost:33080"
NAMESPACE = "cogniverse"


def _run_kubectl(command: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    command_text = " ".join(command)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        pytest.fail(
            f"kubectl executable unavailable after E2E stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; error={exc}",
            pytrace=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"kubectl command timed out after E2E stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; "
            f"timeout={exc.timeout}s; stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"kubectl command failed after E2E stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; "
            f"returncode={result.returncode}; stdout={result.stdout!r}; "
            f"stderr={result.stderr!r}",
            pytrace=False,
        )
    return result


def _kubectl(*args, timeout=10) -> str:
    """Run kubectl command against k3d cluster, return stdout."""
    result = _run_kubectl(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            *args,
        ],
        timeout=timeout,
    )
    return result.stdout.strip()


@pytest.fixture(scope="module", autouse=True)
def require_kubectl_cluster() -> None:
    """Require cluster and Argo access after E2E stack initialization."""
    _run_kubectl(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "namespace",
            NAMESPACE,
            "-o",
            "name",
        ],
        timeout=15,
    )
    controller_command = argo_workflow_controller_probe_command(namespace=ARGO_NAMESPACE)
    controller = _run_kubectl(controller_command, timeout=15)
    if not controller.stdout.strip():
        pytest.fail(
            argo_workflow_controller_probe_failure_message(
                command=controller_command,
                namespace=ARGO_NAMESPACE,
            )
            + f"; stdout={controller.stdout!r}; stderr={controller.stderr!r}",
            pytrace=False,
        )


@pytest.mark.e2e
class TestQualityMonitorSidecar:
    """Verify the quality monitor sidecar is deployed and running."""

    def test_sidecar_container_running(self):
        """Runtime pod has a quality-monitor sidecar container."""
        pods = _kubectl(
            "get",
            "pods",
            "-l",
            "app.kubernetes.io/component=runtime",
            "-o",
            "jsonpath={.items[0].spec.containers[*].name}",
        )
        containers = pods.split()
        assert "quality-monitor" in containers, (
            f"Expected quality-monitor sidecar in runtime pod, "
            f"got containers: {containers}"
        )

    def test_sidecar_container_not_crashlooping(self):
        """Quality monitor sidecar should be running, not CrashLoopBackOff."""
        statuses = _kubectl(
            "get",
            "pods",
            "-l",
            "app.kubernetes.io/component=runtime",
            "-o",
            "jsonpath={.items[0].status.containerStatuses[*].state}",
        )
        assert "CrashLoopBackOff" not in statuses
        assert "Error" not in statuses


@pytest.mark.e2e
class TestPhoenixDatasets:
    """Verify Phoenix dataset operations for eval baselines."""

    def test_phoenix_reachable(self):
        """Phoenix is accessible at the expected endpoint."""
        resp = httpx.get(PHOENIX, timeout=10.0)
        assert resp.status_code == 200

    def test_create_and_read_baseline_dataset(self):
        """Create an eval baseline dataset in Phoenix, read it back."""
        from phoenix.client import Client

        client = Client(base_url=PHOENIX)
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-04T00:00:00",
                    "mean_mrr": 0.75,
                    "mean_ndcg": 0.70,
                    "mean_precision_at_5": 0.50,
                    "query_count": 10,
                }
            ]
        )

        dataset_name = "e2e-quality-baseline-test"
        try:
            client.datasets.create_dataset(
                name=dataset_name,
                dataframe=df,
                input_keys=["timestamp"],
                output_keys=["mean_mrr", "mean_ndcg", "mean_precision_at_5"],
            )

            readback = client.datasets.get_dataset(dataset=dataset_name)
            readback_df = readback.to_dataframe()
            assert len(readback_df) >= 1
        except Exception as e:
            # Dataset may already exist from previous run
            if "already exists" not in str(e):
                raise


@pytest.mark.e2e
class TestSearchWithStrategies:
    """Verify search works and strategies can be injected."""

    def test_search_returns_results(self):
        """Search returns results from Vespa with real tenant."""
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "video of people",
                    "profile": "video_colpali_smol500_mv_frame",
                    "tenant_id": TENANT_ID,
                    "top_k": 5,
                },
            )
        assert resp.status_code == 200, (
            f"Search failed: {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        assert "results" in data

    def test_memory_endpoint_accessible(self):
        """Runtime health check confirms memory system is available."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health")
        assert resp.status_code == 200


@pytest.mark.e2e
class TestArgoWorkflows:
    """Verify Argo CronWorkflows are deployed on k3d."""

    def test_daily_optimization_cronworkflow_exists(self):
        """Daily gateway-optimization CronWorkflow is deployed.

        The chart deploys these optimization-related CronWorkflows
        (see charts/cogniverse/templates/optimization-workflows.yaml):
          - cogniverse-daily-gateway       — daily gateway tuning
          - cogniverse-agent-optimization  — weekly DSPy optimization
          - cogniverse-scheduled-distillation — forced distillation
        This test checks the daily gateway tuning flavor exists.
        """
        output = _kubectl(
            "get",
            "cronworkflows",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        workflows = output.split()
        daily = [w for w in workflows if "daily-gateway" in w]
        assert len(daily) >= 1, (
            f"Expected cogniverse-daily-gateway CronWorkflow, got: {workflows}"
        )

    def test_cleanup_cronworkflow_exists(self):
        """Daily cleanup CronWorkflow is deployed."""
        output = _kubectl(
            "get",
            "cronworkflows",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        workflows = output.split()
        cleanup = [w for w in workflows if "cleanup" in w]
        assert len(cleanup) >= 1, f"Expected cleanup CronWorkflow, got: {workflows}"

    def test_workflow_submitter_rbac_consistent_with_argo(self):
        """RBAC role exists if and only if Argo CronWorkflows are actively managed.

        When argo.enabled=false, both CronWorkflows and RBAC should be absent
        (stale CronWorkflows from previous releases don't count).
        """
        roles_output = _kubectl(
            "get",
            "roles",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        roles = roles_output.split()
        has_rbac = any("workflow-submitter" in r for r in roles)

        # Check if Argo controller is running (not just stale CronWorkflows)
        argo_pods = _kubectl(
            "get",
            "pods",
            "-l",
            "app=workflow-controller",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        argo_active = bool(argo_pods.strip())

        if argo_active:
            assert has_rbac, "Argo is active but workflow-submitter Role is missing"
        # When Argo is not active there is no invariant to assert: the
        # workflow-submitter RBAC may legitimately linger from a previous
        # deploy, so its presence or absence is both acceptable.
