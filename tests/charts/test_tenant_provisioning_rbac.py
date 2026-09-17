"""Chart tests for the account the tenant-provisioning workflow runs under.

``workflows/tenant-provisioning.yaml`` creates a Namespace, a ResourceQuota
and a PersistentVolumeClaim through Argo resource templates, which act with
the workflow pod's ServiceAccount. The chart ships that account and a
ClusterRole granting exactly the verbs those steps use, only when
``tenantProvisioning.enabled`` is set.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
WORKFLOW = REPO_ROOT / "workflows" / "tenant-provisioning.yaml"
RELEASE_NAMESPACE = "cogniverse"
ACCOUNT = "cogniverse-tenant-provisioning"

KIND_RESOURCES = {
    "Namespace": "namespaces",
    "ResourceQuota": "resourcequotas",
    "PersistentVolumeClaim": "persistentvolumeclaims",
}

EXPECTED_RULES = [
    {"apiGroups": [""], "resources": ["namespaces"], "verbs": ["create", "get"]},
    {
        "apiGroups": [""],
        "resources": ["persistentvolumeclaims"],
        "verbs": ["create", "get"],
    },
    {"apiGroups": [""], "resources": ["resourcequotas"], "verbs": ["create"]},
]


def _render(*extra_args: str) -> list[dict]:
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "--namespace",
            RELEASE_NAMESPACE,
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
            *extra_args,
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


def _provisioning_objects(docs: list[dict]) -> dict[str, dict]:
    """Every rendered object named for the provisioning account, by kind."""
    found = [doc for doc in docs if doc["metadata"]["name"] == ACCOUNT]
    by_kind = {doc["kind"]: doc for doc in found}
    assert len(by_kind) == len(found), [doc["kind"] for doc in found]
    return by_kind


def _workflow_template() -> dict:
    (document,) = [
        d
        for d in yaml.safe_load_all(WORKFLOW.read_text())
        if d and d["kind"] == "WorkflowTemplate"
    ]
    return document


def _workflow_resource_verbs() -> list[dict]:
    """The rules the workflow's resource templates need, derived from them."""
    verbs: dict[str, set[str]] = {}
    for template in _workflow_template()["spec"]["templates"]:
        if "resource" not in template:
            continue
        (kind,) = re.findall(r"^kind: (\w+)$", template["resource"]["manifest"], re.M)
        verbs.setdefault(KIND_RESOURCES[kind], set()).add(
            template["resource"]["action"]
        )
    return [
        {"apiGroups": [""], "resources": [resource], "verbs": sorted(verbs[resource])}
        for resource in sorted(verbs)
    ]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestTenantProvisioningAccount:
    def test_the_shipped_default_is_disabled(self):
        values = yaml.safe_load((CHART_PATH / "values.yaml").read_text())
        assert values["tenantProvisioning"] == {"enabled": False}

    def test_a_default_render_ships_no_provisioning_account(self):
        assert _provisioning_objects(_render()) == {}

    def test_an_explicitly_disabled_render_ships_no_provisioning_account(self):
        docs = _render(
            "-f",
            str(CHART_PATH / "values.k3s.yaml"),
            "--set",
            "tenantProvisioning.enabled=false",
        )
        assert _provisioning_objects(docs) == {}

    def test_the_enabled_render_grants_exactly_the_workflow_steps_verbs(self):
        objects = _provisioning_objects(
            _render("--set", "tenantProvisioning.enabled=true")
        )
        assert sorted(objects) == [
            "ClusterRole",
            "ClusterRoleBinding",
            "ServiceAccount",
        ]

        account = objects["ServiceAccount"]
        assert (account["apiVersion"], account["metadata"]["namespace"]) == (
            "v1",
            RELEASE_NAMESPACE,
        )

        role = objects["ClusterRole"]
        assert role["apiVersion"] == "rbac.authorization.k8s.io/v1"
        assert "namespace" not in role["metadata"]
        assert role["rules"] == EXPECTED_RULES
        assert role["rules"] == _workflow_resource_verbs()

        binding = objects["ClusterRoleBinding"]
        assert binding["apiVersion"] == "rbac.authorization.k8s.io/v1"
        assert binding["roleRef"] == {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "ClusterRole",
            "name": ACCOUNT,
        }
        assert binding["subjects"] == [
            {
                "kind": "ServiceAccount",
                "name": ACCOUNT,
                "namespace": RELEASE_NAMESPACE,
            }
        ]

    def test_the_e2e_overlay_enables_the_account(self):
        objects = _provisioning_objects(
            _render("-f", str(CHART_PATH / "values.k3s.yaml"))
        )
        assert sorted(objects) == [
            "ClusterRole",
            "ClusterRoleBinding",
            "ServiceAccount",
        ]
        assert objects["ClusterRole"]["rules"] == EXPECTED_RULES

    def test_the_workflow_runs_under_the_chart_account(self):
        objects = _provisioning_objects(
            _render("--set", "tenantProvisioning.enabled=true")
        )
        template = _workflow_template()
        assert template["spec"]["serviceAccountName"] == ACCOUNT
        assert (
            template["metadata"]["namespace"]
            == objects["ServiceAccount"]["metadata"]["namespace"]
        )
