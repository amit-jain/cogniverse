"""Unit tests for cogniverse_cli.argo workflow filtering."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from cogniverse_cli.argo import (
    deploy_workflow_templates,
    filter_workflow_templates,
    install_argo_controller,
)


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


class TestArgoTimeouts:
    """Every cluster call carries a subprocess timeout — an unreachable API
    server otherwise hangs the argo install/deploy forever."""

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_install_controller_calls_are_timeout_bounded(
        self, mock_run: object
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        install_argo_controller("argo")

        assert mock_run.call_count == 4  # type: ignore[attr-defined]
        for call in mock_run.call_args_list:  # type: ignore[attr-defined]
            timeout = call.kwargs.get("timeout")
            assert isinstance(timeout, int) and timeout > 0
        # The wait step's subprocess timeout must exceed kubectl's own
        # --timeout=300s so the outer bound never fires first.
        wait_call = next(
            c
            for c in mock_run.call_args_list  # type: ignore[attr-defined]
            if "wait" in c.args[0]
        )
        assert wait_call.kwargs["timeout"] > 300

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_deploy_templates_apply_is_timeout_bounded(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )
        wf = tmp_path / "wf.yaml"
        wf.write_text(
            yaml.dump(
                {
                    "apiVersion": "argoproj.io/v1alpha1",
                    "kind": "WorkflowTemplate",
                    "metadata": {"name": "t"},
                }
            )
        )

        deploy_workflow_templates(tmp_path, namespace="cogniverse")

        assert mock_run.call_count == 1  # type: ignore[attr-defined]
        assert mock_run.call_args.kwargs["timeout"] == 120  # type: ignore[attr-defined]

    def test_install_controller_hang_aborts_with_message(self) -> None:
        with patch(
            "cogniverse_cli.argo.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="kubectl create", timeout=30),
        ):
            with pytest.raises(SystemExit) as se:
                install_argo_controller("argo")
        assert "timed out" in str(se.value)
