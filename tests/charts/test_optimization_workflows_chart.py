"""Chart unit tests for the optimization-workflows.yaml template.

Renders the chart with ``helm template`` and asserts the workflow-submitter
Role grants every permission the chart's CronWorkflow steps require.
Catches RBAC regressions at chart-render time instead of waiting for a
Friday 3 AM cron to surface them as a workflow ``Failed``.
"""

from __future__ import annotations

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


def _render() -> list:
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
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


def _find_role(docs: list, name: str) -> dict:
    for d in docs:
        if d.get("kind") == "Role" and d.get("metadata", {}).get("name") == name:
            return d
    raise AssertionError(f"Role {name!r} not found in rendered chart")


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
        """daily-gateway's restart-deployment step runs ``kubectl rollout
        restart deployment/cogniverse-runtime`` so freshly-trained DSPy
        artifacts get picked up. Without get+patch on apps/deployments
        the kubectl call exits 1 → step Failed → workflow Failed → next
        day's run never benefits from the recomputed thresholds."""
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
            "Role does not grant apps/deployments verbs — daily-gateway's "
            "restart-deployment step will fail with Forbidden"
        )
