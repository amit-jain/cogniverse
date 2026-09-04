"""The stack-ready wait after helm install must select only pods that can
become ready within its budget and must fail the deploy when they do not.

Inference pods are sequenced GPU model loads gated separately by the
session fixture's 2400s deployment-available wait; selecting them here
turns every deploy that restarts a model into a timeout. Which inference
pods exist is a cluster fact (helm --set overrides disable services the
values files still enable), so the exclusion set is read from the
deployed deployments' component labels, never derived from config."""

import subprocess
from pathlib import Path

import pytest

import tests.e2e.conftest as e2e_conftest
import tests.e2e.deployment.conftest as deploy_conftest

# Derived from the same place the helper reads it, so a context change fails on
# the real contract rather than on a restated literal.
KUBECTL_CTX = e2e_conftest.KUBECTL_CONTEXT

CHART_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "charts"
    / "cogniverse"
    / "templates"
    / "all-resources.yaml"
)


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_inference_component_prefix_matches_the_chart_label_form():
    assert deploy_conftest.INFERENCE_COMPONENT_PREFIX == "inference-"
    template = CHART_TEMPLATE.read_text()
    assert "app.kubernetes.io/component: inference-{{ $name }}" in template, (
        "chart no longer labels inference pods as inference-<name>"
    )


def test_stack_workloads_lists_non_inference_deployments_and_statefulsets(monkeypatch):
    """The wait must name the workloads, not a snapshot of their pods."""
    calls: list[list[str]] = []

    def fake_cmd(args, *, timeout=120, check=True):
        calls.append(list(args))
        return _completed(
            "Deployment/cogniverse-dashboard dashboard\n"
            "Deployment/cogniverse-gliner inference-gliner\n"
            "Deployment/cogniverse-runtime runtime\n"
            "StatefulSet/cogniverse-vespa vespa\n"
            "Deployment/cogniverse-vllm-asr inference-vllm_asr\n"
            "\n"
        )

    monkeypatch.setattr(deploy_conftest, "_cmd", fake_cmd)
    assert deploy_conftest.stack_workloads("cogniverse") == [
        "deployment/cogniverse-dashboard",
        "deployment/cogniverse-runtime",
        "statefulset/cogniverse-vespa",
    ]
    assert len(calls) == 1 and calls[0][:3] == ["kubectl", "--context", KUBECTL_CTX]


def test_rollout_wait_args_target_one_workload_with_the_remaining_budget():
    assert deploy_conftest.rollout_wait_args(
        "cogniverse", "deployment/cogniverse-runtime", timeout_s=142
    ) == [
        "kubectl",
        "--context",
        KUBECTL_CTX,
        "rollout",
        "status",
        "deployment/cogniverse-runtime",
        "-n",
        "cogniverse",
        "--timeout=142s",
    ]


def test_stack_ready_waits_on_rollouts_not_on_a_pod_snapshot(monkeypatch):
    """`kubectl wait` resolves its pod set once and then watches those pods.

    A rolling update deletes them underneath it, so the wait fails with
    "Error from server (NotFound): pods ... not found" on exactly the deploys
    that changed an image -- the case it exists to cover. `rollout status`
    tracks the workload, so a replaced pod is the success path.
    """
    calls: list[list[str]] = []

    def fake_cmd(args, *, timeout=120, check=True):
        calls.append(list(args))
        if "get" in args:
            return _completed(
                "Deployment/cogniverse-runtime runtime\n"
                "Deployment/cogniverse-gliner inference-gliner\n"
                "StatefulSet/cogniverse-vespa vespa\n"
            )
        return _completed("")

    monkeypatch.setattr(deploy_conftest, "_cmd", fake_cmd)
    deploy_conftest.wait_for_stack_ready("cogniverse")

    issued = [a for a in calls if "rollout" in a]
    assert [a[5] for a in issued] == [
        "deployment/cogniverse-runtime",
        "statefulset/cogniverse-vespa",
    ], issued
    # No pod-snapshot wait may remain: that is the racing command.
    assert [a for a in calls if "wait" in a] == [], calls


def test_stack_ready_dumps_and_raises_when_a_rollout_never_completes(monkeypatch):
    dumps: list[str] = []

    def cmd(args, *, timeout=120, check=True):
        if "get" in args:
            return _completed("Deployment/cogniverse-runtime runtime\n")
        raise subprocess.CalledProcessError(
            1, args, stderr="timed out waiting for the condition"
        )

    monkeypatch.setattr(deploy_conftest, "_cmd", cmd)
    monkeypatch.setattr(
        deploy_conftest, "dump_pod_state", lambda namespace: dumps.append(namespace)
    )
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        deploy_conftest.wait_for_stack_ready("cogniverse")
    assert excinfo.value.stderr == "timed out waiting for the condition"
    assert dumps == ["cogniverse"]


def test_stack_ready_shares_one_budget_across_the_workloads(monkeypatch):
    """Per-workload budgets multiply: 300s each across a dozen workloads is an
    hour, which is not a budget."""
    budgets: list[int] = []

    def cmd(args, *, timeout=120, check=True):
        if "get" in args:
            return _completed(
                "Deployment/a runtime\nDeployment/b dashboard\nDeployment/c minio\n"
            )
        budgets.append(int(args[-1].removeprefix("--timeout=").removesuffix("s")))
        return _completed("")

    monkeypatch.setattr(deploy_conftest, "_cmd", cmd)
    deploy_conftest.wait_for_stack_ready("cogniverse")
    assert len(budgets) == 3
    assert sum(budgets) <= deploy_conftest.STACK_READY_BUDGET_S * len(budgets)
    assert max(budgets) <= deploy_conftest.STACK_READY_BUDGET_S


def test_stack_workloads_raises_when_kubectl_fails(monkeypatch):
    """A listing that fails must not degrade to an empty workload set: waiting
    on nothing reports the stack ready without checking anything."""

    def failing_cmd(args, *, timeout=120, check=True):
        raise subprocess.CalledProcessError(1, args, stderr="connection refused")

    monkeypatch.setattr(deploy_conftest, "_cmd", failing_cmd)
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        deploy_conftest.stack_workloads("cogniverse")
    assert excinfo.value.stderr == "connection refused"


def test_stack_workloads_targets_are_kind_prefixed_and_deduplicated(monkeypatch):
    """`rollout status` needs kind/name; a bare name is ambiguous, and the
    listing can repeat a workload when labels are duplicated."""

    def fake_cmd(args, *, timeout=120, check=True):
        return _completed(
            "Deployment/cogniverse-runtime runtime\n"
            "Deployment/cogniverse-runtime runtime\n"
            "StatefulSet/cogniverse-phoenix phoenix\n"
            "garbage-without-a-component\n"
        )

    monkeypatch.setattr(deploy_conftest, "_cmd", fake_cmd)
    assert deploy_conftest.stack_workloads("cogniverse") == [
        "deployment/cogniverse-runtime",
        "statefulset/cogniverse-phoenix",
    ]


def test_hook_jobs_are_structurally_out_of_scope(monkeypatch):
    """Completed helm-hook pods never reach Ready, and the old pod-snapshot
    wait had to exclude them explicitly. Targeting workloads removes the
    class: a Job is neither a Deployment nor a StatefulSet."""
    listed: list[list[str]] = []

    def fake_cmd(args, *, timeout=120, check=True):
        listed.append(list(args))
        return _completed("Deployment/cogniverse-runtime runtime\n")

    monkeypatch.setattr(deploy_conftest, "_cmd", fake_cmd)
    deploy_conftest.stack_workloads("cogniverse")
    assert "deploy,statefulset" in listed[0]
    assert not any("job" in part for part in listed[0])


class TestDevmodeRefreshWaitsOnDerivedOwners:
    """The refresh deletes every pod carrying the devMode bind-mount, so it
    must rollout-wait every Deployment owning such a pod — derived by the same
    volume predicate, never a hardcoded pair that silently omits a workload
    (the quality-monitor Deployment carries the mount too)."""

    PODS = (
        "cogniverse-runtime-abc-1|src-libs\n"
        "cogniverse-dashboard-def-2|src-libs\n"
        "cogniverse-quality-monitor-ghi-3|src-libs\n"
        "cogniverse-minio-jkl-4|\n"
    )
    DEPLOYS = (
        "cogniverse-dashboard|src-libs\n"
        "cogniverse-minio|\n"
        "cogniverse-quality-monitor|src-libs\n"
        "cogniverse-runtime|src-libs\n"
    )

    def _run(self, monkeypatch, deploys_stdout):
        deleted: list[str] = []
        waited: list[str] = []

        def fake_run(cmd, **kwargs):
            if "ns" in cmd:
                return _completed("namespace/cogniverse")
            if "pods" in cmd:
                return _completed(self.PODS)
            if "deploy" in cmd:
                return _completed(deploys_stdout)
            if "delete" in cmd:
                deleted.append(cmd[cmd.index("pod") + 1])
                return _completed("")
            if "rollout" in cmd:
                waited.append(cmd[cmd.index("status") + 1])
                return _completed("")
            raise AssertionError(f"unexpected kubectl call: {cmd}")

        class _OK:
            status_code = 200

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("httpx.get", lambda *a, **k: _OK())
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.delenv("COGNIVERSE_SKIP_POD_REFRESH", raising=False)
        result = deploy_conftest.refresh_workload_pods_if_devmode(timeout_s=3)
        return result, deleted, waited

    def test_waits_on_exactly_the_deployments_carrying_the_devmode_volume(
        self, monkeypatch
    ):
        result, deleted, waited = self._run(monkeypatch, self.DEPLOYS)
        assert result is True
        assert deleted == [
            "cogniverse-runtime-abc-1",
            "cogniverse-dashboard-def-2",
            "cogniverse-quality-monitor-ghi-3",
        ]
        assert waited == [
            "deployment/cogniverse-dashboard",
            "deployment/cogniverse-quality-monitor",
            "deployment/cogniverse-runtime",
        ]

    def test_refuses_when_no_owner_carries_the_volume(self, monkeypatch):
        """devMode pods with no derivable owners: refuse before deleting
        anything rather than delete pods and wait on nothing."""
        result, deleted, waited = self._run(
            monkeypatch, "cogniverse-dashboard|\ncogniverse-runtime|\n"
        )
        assert result is False
        assert deleted == []
        assert waited == []
