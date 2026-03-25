"""Argo Workflows controller install and workflow template deployment."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import yaml

ARGO_VERSION = "v3.5.0"
ARGO_INSTALL_URL = (
    f"https://github.com/argoproj/argo-workflows/releases/download/"
    f"{ARGO_VERSION}/install.yaml"
)
ALLOWED_KINDS = {"WorkflowTemplate", "CronWorkflow"}


def install_argo_controller(namespace: str = "argo") -> None:
    """Install Argo Workflows controller.

    Creates the namespace if it does not already exist, applies the
    upstream install manifest, and waits for the ``argo-server``
    deployment to become available.
    """
    subprocess.run(
        ["kubectl", "create", "namespace", namespace],
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["kubectl", "apply", "-n", namespace, "-f", ARGO_INSTALL_URL],
        check=True,
    )
    subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=condition=available",
            "deployment/argo-server",
            "-n",
            namespace,
            "--timeout=300s",
        ],
        check=True,
    )


def filter_workflow_templates(yaml_file: Path) -> list[dict]:
    """Parse a YAML file and return only WorkflowTemplate/CronWorkflow documents."""
    text = yaml_file.read_text(encoding="utf-8")
    docs: list[dict] = []
    for doc in yaml.safe_load_all(text):
        if isinstance(doc, dict) and doc.get("kind") in ALLOWED_KINDS:
            docs.append(doc)
    return docs


def deploy_workflow_templates(
    workflows_dir: Path, namespace: str = "cogniverse"
) -> None:
    """Deploy workflow templates from a directory.

    For each ``.yaml`` file in *workflows_dir*, only
    ``WorkflowTemplate`` and ``CronWorkflow`` documents are applied;
    plain ``Workflow`` resources are filtered out.
    """
    for yaml_file in sorted(workflows_dir.glob("*.yaml")):
        filtered = filter_workflow_templates(yaml_file)
        if not filtered:
            continue
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            yaml.dump_all(filtered, tmp, default_flow_style=False)
        try:
            subprocess.run(
                ["kubectl", "apply", "-f", str(tmp_path), "-n", namespace],
                check=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
