"""Unit tests for cogniverse_cli.argo workflow filtering."""

from __future__ import annotations

from pathlib import Path

import yaml
from cogniverse_cli.argo import filter_workflow_templates


class TestFilterWorkflowTemplates:
    """Tests for :func:`filter_workflow_templates`."""

    def test_filter_excludes_workflow_kind(self, tmp_path: Path) -> None:
        """Plain Workflow documents are excluded; WorkflowTemplate is kept."""
        docs = [
            {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "WorkflowTemplate",
                "metadata": {"name": "ingest-template"},
            },
            {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Workflow",
                "metadata": {"name": "manual-run"},
            },
        ]
        yaml_file = tmp_path / "mixed.yaml"
        yaml_file.write_text(
            yaml.dump_all(docs, default_flow_style=False),
            encoding="utf-8",
        )

        result = filter_workflow_templates(yaml_file)

        assert len(result) == 1
        assert result[0]["kind"] == "WorkflowTemplate"
        assert result[0]["metadata"]["name"] == "ingest-template"

    def test_filter_keeps_cronworkflow(self, tmp_path: Path) -> None:
        """CronWorkflow documents are retained alongside WorkflowTemplate."""
        docs = [
            {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "CronWorkflow",
                "metadata": {"name": "nightly-ingest"},
            },
            {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Workflow",
                "metadata": {"name": "one-off"},
            },
        ]
        yaml_file = tmp_path / "cron.yaml"
        yaml_file.write_text(
            yaml.dump_all(docs, default_flow_style=False),
            encoding="utf-8",
        )

        result = filter_workflow_templates(yaml_file)

        assert len(result) == 1
        assert result[0]["kind"] == "CronWorkflow"
        assert result[0]["metadata"]["name"] == "nightly-ingest"
