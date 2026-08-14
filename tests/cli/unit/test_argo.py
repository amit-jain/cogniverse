"""Unit tests for cogniverse_cli.argo workflow filtering."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from cogniverse_cli.argo import (
    ARGO_INSTALL_URL,
    deploy_workflow_templates,
    filter_workflow_templates,
    install_argo_controller,
)

from tests.e2e.conftest import (
    ARGO_NAMESPACE,
    ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR,
    KUBECTL_CONTEXT,
    argo_workflow_controller_probe_command,
    argo_workflow_controller_probe_failure_message,
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


class TestArgoProbeContract:
    """The readiness probe must target the authoritative Argo namespace and label."""

    def test_controller_probe_targets_argo_namespace_and_real_label(self) -> None:
        command = argo_workflow_controller_probe_command()

        assert command == [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            ARGO_NAMESPACE,
            "get",
            "pods",
            "-l",
            ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR,
            "--field-selector=status.phase=Running",
            "-o",
            "name",
        ]

        message = argo_workflow_controller_probe_failure_message(command=command)
        assert f"namespace={ARGO_NAMESPACE!r}" in message
        assert f"selector={ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR!r}" in message
        assert "command='kubectl --context" in message


class TestInstallArgoController:
    """`install_argo_controller` issues create-ns → apply-manifest →
    patch-auth-mode → wait, in that exact order."""

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_issues_exact_kubectl_sequence(self, mock_run: object) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        install_argo_controller("argo")

        cmds = [call.args[0] for call in mock_run.call_args_list]  # type: ignore[attr-defined]
        assert cmds == [
            ["kubectl", "create", "namespace", "argo"],
            ["kubectl", "apply", "-n", "argo", "-f", ARGO_INSTALL_URL],
            [
                "kubectl",
                "patch",
                "deployment",
                "argo-server",
                "-n",
                "argo",
                "--type=json",
                "-p",
                (
                    '[{"op":"replace",'
                    '"path":"/spec/template/spec/containers/0/args",'
                    '"value":["server","--auth-mode=server"]}]'
                ),
            ],
            [
                "kubectl",
                "wait",
                "--for=condition=available",
                "deployment/argo-server",
                "-n",
                "argo",
                "--timeout=300s",
            ],
        ]

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_create_namespace_is_non_fatal_and_quiet(self, mock_run: object) -> None:
        """Namespace creation ignores an already-existing namespace
        (check=False) and suppresses its output (capture_output=True), so a
        re-run of the install does not abort."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        install_argo_controller("argo")

        create_call = mock_run.call_args_list[0]  # type: ignore[attr-defined]
        assert create_call.args[0] == ["kubectl", "create", "namespace", "argo"]
        assert create_call.kwargs["check"] is False
        assert create_call.kwargs["capture_output"] is True

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_patch_replaces_args_with_auth_mode_server(self, mock_run: object) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        install_argo_controller("argo")

        patch_call = next(
            c
            for c in mock_run.call_args_list  # type: ignore[attr-defined]
            if "patch" in c.args[0]
        )
        cmd = patch_call.args[0]
        payload = cmd[cmd.index("-p") + 1]
        assert "--type=json" in cmd  # strategic json-patch, not a merge patch
        assert '"value":["server","--auth-mode=server"]' in payload
        assert '"path":"/spec/template/spec/containers/0/args"' in payload


class TestDeployWorkflowTemplates:
    """`deploy_workflow_templates` globs *.yaml sorted, filters each to
    WorkflowTemplate/CronWorkflow, writes a temp file, and applies it."""

    def _capture_applies(self, mock_run: object) -> list[tuple[list[str], str]]:
        applied: list[tuple[list[str], str]] = []

        def fake_run(cmd, **kwargs):
            # Read the temp manifest the apply points at before it is unlinked.
            path = cmd[cmd.index("-f") + 1]
            applied.append((cmd, Path(path).read_text(encoding="utf-8")))
            return subprocess.CompletedProcess(cmd, 0)

        mock_run.side_effect = fake_run  # type: ignore[attr-defined]
        return applied

    @staticmethod
    def _doc(kind: str, name: str) -> dict:
        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": kind,
            "metadata": {"name": name},
        }

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_applies_only_template_docs_per_file(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        applied = self._capture_applies(mock_run)
        (tmp_path / "a.yaml").write_text(
            yaml.dump_all(
                [
                    self._doc("WorkflowTemplate", "tmpl-a"),
                    self._doc("Workflow", "run-a"),
                ]
            )
        )
        (tmp_path / "b.yaml").write_text(yaml.dump(self._doc("CronWorkflow", "cron-b")))
        (tmp_path / "c.yaml").write_text(yaml.dump(self._doc("Workflow", "run-c")))

        deploy_workflow_templates(tmp_path, namespace="cogniverse")

        # a.yaml + b.yaml apply (sorted glob -> a first); c.yaml is skipped.
        assert len(applied) == 2
        cmd_a, content_a = applied[0]
        cmd_b, content_b = applied[1]
        assert cmd_a[:3] == ["kubectl", "apply", "-f"]
        assert cmd_a[3].endswith(".yaml")
        assert cmd_a[-2:] == ["-n", "cogniverse"]
        assert cmd_b[-2:] == ["-n", "cogniverse"]
        # The plain Workflow in a.yaml is filtered OUT before apply.
        assert {d["metadata"]["name"] for d in yaml.safe_load_all(content_a)} == {
            "tmpl-a"
        }
        assert {d["metadata"]["name"] for d in yaml.safe_load_all(content_b)} == {
            "cron-b"
        }

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_file_without_templates_applies_nothing(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        applied = self._capture_applies(mock_run)
        (tmp_path / "only-workflow.yaml").write_text(
            yaml.dump(self._doc("Workflow", "x"))
        )

        deploy_workflow_templates(tmp_path, namespace="cogniverse")

        assert applied == []

    @patch("cogniverse_cli.argo.subprocess.run")
    def test_temp_manifest_is_unlinked_after_apply(
        self, mock_run: object, tmp_path: Path
    ) -> None:
        seen: list[str] = []

        def fake_run(cmd, **kwargs):
            path = cmd[cmd.index("-f") + 1]
            seen.append(path)
            assert Path(path).exists()  # live during the apply
            return subprocess.CompletedProcess(cmd, 0)

        mock_run.side_effect = fake_run  # type: ignore[attr-defined]
        (tmp_path / "a.yaml").write_text(yaml.dump(self._doc("WorkflowTemplate", "t")))

        deploy_workflow_templates(tmp_path, namespace="cogniverse")

        assert len(seen) == 1
        assert not Path(seen[0]).exists()  # cleaned up in the finally block
