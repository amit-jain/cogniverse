"""Chart unit tests for the optimization-workflows.yaml template.

Renders the chart with ``helm template`` and asserts the workflow-submitter
Role grants every permission the chart's CronWorkflow steps require.
Catches RBAC regressions at chart-render time instead of waiting for a
Friday 3 AM cron to surface them as a workflow ``Failed``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
EXPECTED_INFERENCE_SERVICE_URLS = {
    "colbert_pylate": "http://cogniverse-colbert-pylate:8000",
    "denseon": "http://cogniverse-denseon:8000",
    "gliner": "http://cogniverse-gliner:8080",
    "vllm_asr": "http://cogniverse-vllm-asr:8000",
}

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*set_args: str, values: str | tuple[str, ...] | None = None) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for values_file in (values,) if isinstance(values, str) else (values or ()):
        args += ["-f", str(CHART_PATH / values_file)]
    for s in set_args:
        args += ["--set", s]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _find_role(docs: list, name: str) -> dict:
    for d in docs:
        if d.get("kind") == "Role" and d.get("metadata", {}).get("name") == name:
            return d
    raise AssertionError(f"Role {name!r} not found in rendered chart")


def _find_workflow_template(docs: list, name_suffix: str) -> dict:
    for d in docs:
        if d.get("kind") == "WorkflowTemplate" and d.get("metadata", {}).get(
            "name", ""
        ).endswith(name_suffix):
            return d
    raise AssertionError(f"No WorkflowTemplate ending in {name_suffix!r} rendered")


class TestWorkflowSubmitterRoleGrantsEveryStepNeeds:
    """The cogniverse-workflow-submitter Role must cover every RBAC verb
    the chart's CronWorkflow steps actually use. Each entry below
    corresponds to a real step in optimization-workflows.yaml."""

    def test_role_grants_workflow_submission(self):
        """Quality monitor sidecar submits Workflow CRs on demand."""
        docs = _render()
        role = _find_role(docs, "cogniverse-workflow-submitter")
        for rule in role.get("rules", []):
            if "argoproj.io" in rule.get("apiGroups", []) and "workflows" in rule.get(
                "resources", []
            ):
                assert {"create", "get", "list"}.issubset(set(rule["verbs"])), (
                    f"workflows verbs incomplete: {rule['verbs']}"
                )
                return
        raise AssertionError("Role does not grant any verbs on argoproj.io/workflows")

    def test_role_grants_workflowtaskresults_write(self):
        """Argo Emissary writes workflowtaskresults after main container exits.
        Without this, every Workflow is marked Failed even when the work
        succeeded — the silent-corruption bug the chart comment calls out."""
        docs = _render()
        role = _find_role(docs, "cogniverse-workflow-submitter")
        for rule in role.get("rules", []):
            if "argoproj.io" in rule.get(
                "apiGroups", []
            ) and "workflowtaskresults" in rule.get("resources", []):
                assert {"create", "patch"}.issubset(set(rule["verbs"])), (
                    f"workflowtaskresults verbs incomplete: {rule['verbs']}"
                )
                return
        raise AssertionError(
            "Role does not grant any verbs on argoproj.io/workflowtaskresults"
        )

    def test_role_grants_deployment_restart(self):
        """The weekly agent-optimization's restart-deployment step runs
        ``kubectl rollout restart deployment/cogniverse-runtime`` after
        full recompiles. Without get+patch on apps/deployments the
        kubectl call exits 1 → step Failed → workflow Failed."""
        docs = _render()
        role = _find_role(docs, "cogniverse-workflow-submitter")
        for rule in role.get("rules", []):
            if "apps" in rule.get("apiGroups", []) and "deployments" in rule.get(
                "resources", []
            ):
                assert {"get", "patch"}.issubset(set(rule["verbs"])), (
                    f"deployments verbs incomplete for rollout restart: {rule['verbs']}"
                )
                return
        raise AssertionError(
            "Role does not grant apps/deployments verbs — the weekly "
            "agent-optimization's restart-deployment step will fail with "
            "Forbidden"
        )


def _find_cron_workflow(docs: list, name_suffix: str) -> dict:
    for d in docs:
        if d.get("kind") == "CronWorkflow" and d.get("metadata", {}).get(
            "name", ""
        ).endswith(name_suffix):
            return d
    raise AssertionError(f"No CronWorkflow ending in {name_suffix!r} rendered")


def _container_env(workload: dict) -> dict[str, str]:
    if workload.get("kind") == "WorkflowTemplate":
        container = workload["spec"]["templates"][0]["container"]
    elif workload.get("kind") == "CronWorkflow":
        templates = workload["spec"]["workflowSpec"]["templates"]
        for template in templates:
            container = template.get("container")
            if container is not None:
                break
        else:
            raise AssertionError(
                f"no pod container found in CronWorkflow {workload['metadata']['name']!r}"
            )
    else:
        raise AssertionError(f"unsupported workload kind: {workload.get('kind')!r}")
    return {entry["name"]: entry.get("value") for entry in container.get("env", [])}


def _workflow_template_files() -> list[Path]:
    template_dir = CHART_PATH / "templates"
    workflow_files = []
    for path in sorted(template_dir.glob("*.yaml")):
        text = path.read_text()
        if re.search(r"(?m)^\s*kind:\s*(WorkflowTemplate|CronWorkflow)\s*$", text):
            workflow_files.append(path)
    return workflow_files


def _workflow_containers(workload: dict) -> list[dict]:
    if workload.get("kind") == "WorkflowTemplate":
        return [
            template["container"]
            for template in workload["spec"]["templates"]
            if "container" in template
        ]
    if workload.get("kind") == "CronWorkflow":
        return [
            template["container"]
            for template in workload["spec"]["workflowSpec"]["templates"]
            if "container" in template
        ]
    raise AssertionError(f"unsupported workload kind: {workload.get('kind')!r}")


class TestDailyGatewayHasNoRestartStep:
    def test_daily_gateway_relies_on_the_reload_interval(self):
        """The runtime picks up recalibrated gateway thresholds on warm pods
        via the dispatcher's reload interval, so the daily cron must not
        rolling-restart the deployment every morning; the weekly full
        recompile keeps its restart step."""
        docs = _render()
        daily = _find_cron_workflow(docs, "-daily-gateway")
        spec = daily["spec"]["workflowSpec"]

        templates = {t["name"]: t for t in spec["templates"]}
        assert "restart-deployment" not in templates
        pipeline = templates["daily-gateway-pipeline"]
        step_names = [s["name"] for group in pipeline["steps"] for s in group]
        assert step_names == ["optimize-gateway"]

        weekly = _find_cron_workflow(docs, "-agent-optimization")
        weekly_templates = {
            t["name"] for t in weekly["spec"]["workflowSpec"]["templates"]
        }
        assert "restart-deployment" in weekly_templates


class TestSyntheticGenerationUsesOnlyApprovedOptimizers:
    """The synthetic-generation CronWorkflow must only request optimizer
    types that have an approved training-data consumer. The optimization
    CLI hard-fails on any other type (b15 e2e sweep: ``--agents
    workflow,profile`` failed the whole workflow with "synthetic optimizer
    types have no approved training-data consumer: ['workflow']").
    """

    def test_agents_arg_subset_of_approved_optimizers(self):
        from cogniverse_synthetic.registry import (
            APPROVED_TRAINING_AGENT_BY_OPTIMIZER,
        )

        docs = _render()
        cron = _find_cron_workflow(docs, "synthetic-generation")
        templates = cron["spec"]["workflowSpec"]["templates"]
        agents_values = []
        for tpl in templates:
            container = tpl.get("container") or {}
            args = container.get("args") or []
            for i, arg in enumerate(args):
                if arg == "--agents" and i + 1 < len(args):
                    agents_values.append(args[i + 1])
        assert agents_values, (
            "no --agents arg found in the synthetic-generation CronWorkflow"
        )
        for value in agents_values:
            requested = {a.strip() for a in value.split(",") if a.strip()}
            unapproved = requested - set(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
            assert not unapproved, (
                "synthetic-generation requests optimizer types with no "
                f"approved training-data consumer: {sorted(unapproved)}"
            )


@pytest.mark.parametrize(
    ("kind", "name_suffix"),
    [
        ("WorkflowTemplate", "-optimization-runner"),
        ("CronWorkflow", "-daily-cleanup"),
        ("CronWorkflow", "-synthetic-generation"),
        ("CronWorkflow", "-monthly-reports"),
    ],
)
def test_optimizer_workloads_carry_inference_service_urls(kind: str, name_suffix: str):
    """Every optimizer pod needs the denseon URL map entry.

    The shared WorkflowTemplate feeds agent-optimization and daily-gateway
    through templateRef, while the direct CronWorkflows here launch their own
    optimizer pods.
    """
    docs = _render()
    workload = (
        _find_workflow_template(docs, name_suffix)
        if kind == "WorkflowTemplate"
        else _find_cron_workflow(docs, name_suffix)
    )
    env = _container_env(workload)

    assert json.loads(env["INFERENCE_SERVICE_URLS"]) == EXPECTED_INFERENCE_SERVICE_URLS


def test_job_workflow_template_carries_inference_service_urls():
    """job_executor routes post-actions through denseon using the same map."""
    docs = _render()
    workload = _find_workflow_template(docs, "-job-runner")
    env = _container_env(workload)

    assert json.loads(env["INFERENCE_SERVICE_URLS"]) == EXPECTED_INFERENCE_SERVICE_URLS


def test_every_workflow_pod_carries_the_modal_bearer():
    """Every rendered workflow pod needs the same bearer secret wiring.

    The source tree drives the set of chart files under test, so any new
    WorkflowTemplate/CronWorkflow YAML added under charts/cogniverse/templates/
    is automatically included here.
    """
    workflow_files = _workflow_template_files()
    assert workflow_files, (
        "no workflow templates found under charts/cogniverse/templates"
    )

    docs = _render(
        "hostStorage.backup.enabled=true",
        values=("values.rocm.yaml", "values.modal-llm.yaml"),
    )
    workloads = [
        doc for doc in docs if doc.get("kind") in {"WorkflowTemplate", "CronWorkflow"}
    ]
    assert workloads, "no workflow or cronworkflow manifests rendered"

    expected_secret_ref = {
        "name": "cogniverse-inference-api-key",
        "key": "COGNIVERSE_INFERENCE_API_KEY",
        "optional": False,
    }
    missing: list[str] = []
    for workload in workloads:
        for container in _workflow_containers(workload):
            env = {entry["name"]: entry for entry in container.get("env", [])}
            entry = env.get("COGNIVERSE_INFERENCE_API_KEY")
            if entry is None:
                missing.append(
                    f"{workload['metadata']['name']} / {container.get('name', '<unnamed>')} "
                    "is missing COGNIVERSE_INFERENCE_API_KEY"
                )
                continue
            secret_ref = entry.get("valueFrom", {}).get("secretKeyRef")
            if secret_ref != expected_secret_ref:
                missing.append(
                    f"{workload['metadata']['name']} / {container.get('name', '<unnamed>')} "
                    f"has the wrong bearer wiring: {secret_ref!r}"
                )

    assert not missing, "missing bearer secretKeyRef:\n- " + "\n- ".join(missing)
