"""Unit tests for cogniverse_cli.main CLI entrypoint."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from cogniverse_cli.main import (
    SERVICE_ENDPOINTS,
    SERVICE_HEALTH_URLS,
    _probe_host_llm,
    cli,
)


def _install_fake_cluster_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cluster_state: dict,
    *,
    start_returncode: int = 0,
    start_stderr: str = "",
    inspect_returncode: int = 0,
    post_start_cluster_state: dict | None = None,
) -> Path:
    """Run cluster lifecycle tests through harmless executable boundaries."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "k3d-calls"
    k3d = bin_dir / "k3d"
    k3d.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
marker = Path(os.environ["FAKE_K3D_MARKER"])
if args[:2] == ["cluster", "list"] and "-o" in args:
    started = marker.exists() and "start\\n" in marker.read_text()
    with marker.open("a") as calls:
        calls.write("inspect\\n")
    state_key = "FAKE_K3D_POST_START_STATE" if started else "FAKE_K3D_STATE"
    print(os.environ[state_key])
    raise SystemExit(int(os.environ["FAKE_K3D_INSPECT_RETURNCODE"]))
if args[:2] == ["cluster", "start"]:
    with marker.open("a") as calls:
        calls.write("start\\n")
    message = os.environ.get("FAKE_K3D_START_STDERR", "")
    if message:
        print(message, file=sys.stderr)
    raise SystemExit(int(os.environ["FAKE_K3D_START_RETURNCODE"]))
raise SystemExit(0)
"""
    )
    k3d.chmod(0o755)
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        """#!/usr/bin/env python3
import sys

if "configmap" in sys.argv:
    print("forward . 1.1.1.1 8.8.8.8")
raise SystemExit(0)
"""
    )
    kubectl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_K3D_MARKER", str(marker))
    monkeypatch.setenv("FAKE_K3D_STATE", json.dumps([cluster_state]))
    if post_start_cluster_state is None:
        post_start_cluster_state = json.loads(json.dumps(cluster_state))
        network_name = (post_start_cluster_state.get("network") or {}).get("name")
        for node in post_start_cluster_state["nodes"]:
            if node["role"] == "loadbalancer":
                node["State"] = {"Running": True}
                node["Networks"] = [network_name]
    monkeypatch.setenv(
        "FAKE_K3D_POST_START_STATE", json.dumps([post_start_cluster_state])
    )
    monkeypatch.setenv("FAKE_K3D_START_RETURNCODE", str(start_returncode))
    monkeypatch.setenv("FAKE_K3D_START_STDERR", start_stderr)
    monkeypatch.setenv("FAKE_K3D_INSPECT_RETURNCODE", str(inspect_returncode))
    return marker


def _cluster_state(host_port: int) -> dict:
    return {
        "name": "cogniverse",
        "network": {"name": "k3d-cogniverse"},
        "nodes": [
            {
                "name": "k3d-cogniverse-serverlb",
                "role": "loadbalancer",
                "portMappings": {
                    "11434/tcp": [{"HostIp": "", "HostPort": str(host_port)}]
                },
                "State": {"Running": False},
                "Networks": [],
            },
            {
                "name": "k3d-cogniverse-server-0",
                "role": "server",
                "portMappings": {},
                "State": {"Running": True},
            },
        ],
    }


class TestCli:
    """Tests for the top-level CLI group."""

    def test_cli_help(self) -> None:
        """--help exits 0 and shows all four commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ("up", "down", "status", "logs", "stop", "start"):
            assert cmd in result.output

    def test_up_help(self) -> None:
        """up --help shows --llm, --llm-url, --image-source options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--help"])
        assert result.exit_code == 0
        assert "--llm" in result.output
        assert "--llm-url" in result.output
        assert "--image-source" in result.output

    def test_logs_help(self) -> None:
        """logs --help shows the service argument choices."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--help"])
        assert result.exit_code == 0
        for svc in ("runtime", "dashboard", "vespa", "phoenix", "llm", "argo"):
            assert svc in result.output


class TestProbeHostLlm:
    """Tests for :func:`_probe_host_llm`."""

    @patch("cogniverse_cli.main.httpx.get")
    def test_probe_host_llm_healthy(self, mock_get: MagicMock) -> None:
        """Returns True when Host LLM responds with HTTP 200."""
        mock_get.return_value = MagicMock(status_code=200)
        assert _probe_host_llm() is True

    @patch("cogniverse_cli.main.httpx.get", side_effect=OSError("refused"))
    def test_probe_host_llm_connection_error(self, mock_get: MagicMock) -> None:
        """Returns False when Host LLM is unreachable."""
        assert _probe_host_llm() is False

    @patch("cogniverse_cli.main.httpx.get")
    def test_probe_host_llm_non_200(self, mock_get: MagicMock) -> None:
        """Returns False when Host LLM returns a non-200 status."""
        mock_get.return_value = MagicMock(status_code=500)
        assert _probe_host_llm() is False


class TestUpCommand:
    """Tests for the ``up`` command."""

    @pytest.fixture(autouse=True)
    def _no_live_secret_sync(self):
        # up() imports the sync helpers from cogniverse_cli.secrets and
        # kubectl-applies the HF token and inference API key; without this,
        # the full-flow tests mutate a live k3d cluster's Secrets on a dev box.
        with (
            patch("cogniverse_cli.secrets.sync_hf_token_to_cluster"),
            patch("cogniverse_cli.secrets.sync_inference_api_key_to_cluster"),
        ):
            yield

    @patch("cogniverse_cli.main.check_prerequisites", return_value=["docker"])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=False)
    def test_up_aborts_on_missing_prerequisites(
        self, mock_k8s: MagicMock, mock_prereq: MagicMock
    ) -> None:
        """Exits with error code when prerequisites are missing."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code != 0
        assert (
            "Failed to install" in result.output
            or "Missing prerequisites" in result.output
        )

    @patch("cogniverse_cli.main._print_status_table")
    @patch("cogniverse_cli.main.deploy_workflow_templates")
    @patch("cogniverse_cli.main.install_argo_controller")
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.wait_for_url", return_value=True)
    @patch("cogniverse_cli.main.helm_install")
    @patch("cogniverse_cli.main.pull_and_import_third_party")
    @patch("cogniverse_cli.main.get_values_file", return_value=Path("/v.yaml"))
    @patch("cogniverse_cli.main.get_chart_path", return_value=Path("/chart"))
    @patch("cogniverse_cli.main.get_workflows_path", return_value=Path("/wf"))
    @patch("cogniverse_cli.main._probe_host_llm", return_value=False)
    @patch("cogniverse_cli.main.has_workspace_source", return_value=False)
    @patch("cogniverse_cli.main.resolve_project_root", return_value=Path("/root"))
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    @patch("cogniverse_cli.main.check_prerequisites", return_value=[])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=False)
    @patch("cogniverse_cli.main.start_port_forwards")
    def test_up_k3d_builtin_llm(
        self,
        mock_start_pf: MagicMock,
        mock_k8s: MagicMock,
        mock_prereq: MagicMock,
        mock_cluster: MagicMock,
        mock_root: MagicMock,
        mock_ws: MagicMock,
        mock_probe: MagicMock,
        mock_wf_path: MagicMock,
        mock_chart: MagicMock,
        mock_values: MagicMock,
        mock_pull: MagicMock,
        mock_helm: MagicMock,
        mock_wait: MagicMock,
        mock_subprocess: MagicMock,
        mock_argo: MagicMock,
        mock_deploy_wf: MagicMock,
        mock_status: MagicMock,
    ) -> None:
        """Full up flow in k3d mode with builtin LLM (auto, no host LLM)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        mock_prereq.assert_called_once_with(require_k3d=True)
        mock_values.assert_called_once_with(prod=False)
        mock_helm.assert_called_once()
        call_kwargs = mock_helm.call_args
        set_vals = call_kwargs[1].get("set_values") or {}
        assert set_vals["argo-workflows.crds.install"] == "false"
        assert set_vals["runtime.backend"] in {"cpu", "cuda", "rocm"}
        assert set_vals["dashboard.backend"] == set_vals["runtime.backend"]
        assert "llm.builtin.enabled" not in set_vals
        assert "llm.external.enabled" not in set_vals

    @patch("cogniverse_cli.main._print_status_table")
    @patch("cogniverse_cli.main.deploy_workflow_templates")
    @patch("cogniverse_cli.main.install_argo_controller")
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.wait_for_url", return_value=True)
    @patch("cogniverse_cli.main.helm_install")
    @patch("cogniverse_cli.main.pull_and_import_third_party")
    @patch("cogniverse_cli.main.get_values_file", return_value=Path("/v.yaml"))
    @patch("cogniverse_cli.main.get_chart_path", return_value=Path("/chart"))
    @patch("cogniverse_cli.main.get_workflows_path", return_value=Path("/wf"))
    @patch("cogniverse_cli.main._probe_host_llm", return_value=False)
    @patch("cogniverse_cli.main.has_workspace_source", return_value=False)
    @patch("cogniverse_cli.main.resolve_project_root", return_value=Path("/root"))
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    @patch("cogniverse_cli.main.check_prerequisites", return_value=[])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=False)
    @patch("cogniverse_cli.main.start_port_forwards")
    def test_up_syncs_hf_token_and_inference_api_key(
        self,
        mock_start_pf: MagicMock,
        mock_k8s: MagicMock,
        mock_prereq: MagicMock,
        mock_cluster: MagicMock,
        mock_root: MagicMock,
        mock_ws: MagicMock,
        mock_probe: MagicMock,
        mock_wf_path: MagicMock,
        mock_chart: MagicMock,
        mock_values: MagicMock,
        mock_pull: MagicMock,
        mock_helm: MagicMock,
        mock_wait: MagicMock,
        mock_subprocess: MagicMock,
        mock_argo: MagicMock,
        mock_deploy_wf: MagicMock,
        mock_status: MagicMock,
    ) -> None:
        """up() bootstraps both chart-referenced Secrets before helm install."""
        with (
            patch("cogniverse_cli.secrets.sync_hf_token_to_cluster") as mock_hf,
            patch(
                "cogniverse_cli.secrets.sync_inference_api_key_to_cluster"
            ) as mock_inference,
        ):
            result = CliRunner().invoke(cli, ["up"])
        assert result.exit_code == 0
        mock_hf.assert_called_once_with(required=False)
        mock_inference.assert_called_once_with(required=False)

    @patch("cogniverse_cli.main._print_status_table")
    @patch("cogniverse_cli.main.deploy_workflow_templates")
    @patch("cogniverse_cli.main.install_argo_controller")
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.wait_for_url", return_value=True)
    @patch("cogniverse_cli.main.helm_install")
    @patch("cogniverse_cli.main.pull_and_import_third_party")
    @patch("cogniverse_cli.main.get_values_file", return_value=Path("/v.yaml"))
    @patch("cogniverse_cli.main.get_chart_path", return_value=Path("/chart"))
    @patch("cogniverse_cli.main.get_workflows_path", return_value=Path("/wf"))
    @patch("cogniverse_cli.main._probe_host_llm", return_value=True)
    @patch("cogniverse_cli.main.has_workspace_source", return_value=False)
    @patch("cogniverse_cli.main.resolve_project_root", return_value=Path("/root"))
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    @patch("cogniverse_cli.main.check_prerequisites", return_value=[])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=False)
    @patch("cogniverse_cli.main.start_port_forwards")
    def test_up_k3d_auto_detects_host_llm(
        self,
        mock_start_pf: MagicMock,
        mock_k8s: MagicMock,
        mock_prereq: MagicMock,
        mock_cluster: MagicMock,
        mock_root: MagicMock,
        mock_ws: MagicMock,
        mock_probe: MagicMock,
        mock_wf_path: MagicMock,
        mock_chart: MagicMock,
        mock_values: MagicMock,
        mock_pull: MagicMock,
        mock_helm: MagicMock,
        mock_wait: MagicMock,
        mock_subprocess: MagicMock,
        mock_argo: MagicMock,
        mock_deploy_wf: MagicMock,
        mock_status: MagicMock,
    ) -> None:
        """When auto-detect finds host LLM on k3d, LLM overrides point at
        the k3d-side host alias."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        call_kwargs = mock_helm.call_args
        set_vals = (
            call_kwargs[1].get("set_values") or call_kwargs[0][2]
            if len(call_kwargs[0]) > 2
            else call_kwargs[1].get("set_values")
        )
        assert set_vals is not None
        assert set_vals["llm.builtin.enabled"] == "false"
        assert set_vals["llm.external.enabled"] == "true"
        assert "host.k3d.internal" in set_vals["llm.external.url"]

    @patch("cogniverse_cli.main._print_status_table")
    @patch("cogniverse_cli.main._print_status_table")
    @patch("cogniverse_cli.main.deploy_workflow_templates")
    @patch("cogniverse_cli.main.install_argo_controller")
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.wait_for_url", return_value=True)
    @patch("cogniverse_cli.main.helm_install")
    @patch("cogniverse_cli.main.get_values_file", return_value=Path("/v.yaml"))
    @patch("cogniverse_cli.main.get_chart_path", return_value=Path("/chart"))
    @patch("cogniverse_cli.main.get_workflows_path", return_value=Path("/wf"))
    @patch("cogniverse_cli.main.has_workspace_source", return_value=False)
    @patch("cogniverse_cli.main.resolve_project_root", return_value=Path("/root"))
    @patch("cogniverse_cli.main.cluster_exists", return_value=False)
    @patch("cogniverse_cli.main.check_prerequisites", return_value=[])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=True)
    @patch("cogniverse_cli.main.start_port_forwards")
    def test_up_existing_k8s_uses_prod_values(
        self,
        mock_start_pf: MagicMock,
        mock_k8s: MagicMock,
        mock_prereq: MagicMock,
        mock_cluster: MagicMock,
        mock_root: MagicMock,
        mock_ws: MagicMock,
        mock_wf_path: MagicMock,
        mock_chart: MagicMock,
        mock_values: MagicMock,
        mock_helm: MagicMock,
        mock_wait: MagicMock,
        mock_subprocess: MagicMock,
        mock_argo: MagicMock,
        mock_deploy_wf: MagicMock,
        mock_print_status: MagicMock,
        _extra: MagicMock,
    ) -> None:
        """Existing K8s uses prod values and does not require k3d."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        mock_prereq.assert_called_once_with(require_k3d=False)
        mock_values.assert_called_once_with(prod=True)

    @patch("cogniverse_cli.main.has_workspace_source", return_value=False)
    @patch("cogniverse_cli.main.resolve_project_root", return_value=Path("/root"))
    @patch("cogniverse_cli.main.cluster_exists", return_value=False)
    @patch("cogniverse_cli.main.check_prerequisites", return_value=[])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=True)
    def test_up_external_llm_requires_url_on_existing_k8s(
        self,
        mock_k8s: MagicMock,
        mock_prereq: MagicMock,
        mock_cluster: MagicMock,
        mock_root: MagicMock,
        mock_ws: MagicMock,
    ) -> None:
        """--llm=external without --llm-url on existing K8s exits with error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--llm", "external"])
        assert result.exit_code != 0
        assert "--llm-url is required" in result.output

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main._print_status_table")
    @patch("cogniverse_cli.main.deploy_workflow_templates")
    @patch("cogniverse_cli.main.install_argo_controller")
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.wait_for_url", return_value=True)
    @patch("cogniverse_cli.main.helm_install")
    @patch("cogniverse_cli.main.pull_and_import_third_party")
    @patch("cogniverse_cli.main.get_values_file", return_value=Path("/v.yaml"))
    @patch("cogniverse_cli.main.get_chart_path", return_value=Path("/chart"))
    @patch("cogniverse_cli.main.get_workflows_path", return_value=Path("/wf"))
    @patch("cogniverse_cli.main._probe_host_llm", return_value=False)
    @patch("cogniverse_cli.main.has_workspace_source", return_value=False)
    @patch("cogniverse_cli.main.resolve_project_root", return_value=Path("/root"))
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    @patch("cogniverse_cli.main.check_prerequisites", return_value=[])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=False)
    def test_up_starts_port_forwards(
        self,
        mock_k8s: MagicMock,
        mock_prereq: MagicMock,
        mock_cluster: MagicMock,
        mock_root: MagicMock,
        mock_ws: MagicMock,
        mock_probe: MagicMock,
        mock_wf_path: MagicMock,
        mock_chart: MagicMock,
        mock_values: MagicMock,
        mock_pull: MagicMock,
        mock_helm: MagicMock,
        mock_wait: MagicMock,
        mock_subprocess: MagicMock,
        mock_argo: MagicMock,
        mock_deploy_wf: MagicMock,
        mock_status: MagicMock,
        mock_start_pf: MagicMock,
    ) -> None:
        """A full up establishes the Argo port-forward exactly once."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        mock_start_pf.assert_called_once()

    def test_up_reports_cluster_creation_failure(self) -> None:
        from cogniverse_cli.cluster import ClusterStartError

        with (
            patch("cogniverse_cli.main.check_prerequisites", return_value=[]),
            patch("cogniverse_cli.main.has_existing_k8s", return_value=False),
            patch("cogniverse_cli.main.cluster_exists", return_value=False),
            patch("cogniverse_cli.main.has_workspace_source", return_value=False),
            patch(
                "cogniverse_cli.main.resolve_project_root", return_value=Path("/root")
            ),
            patch("cogniverse_cli.main._probe_host_llm", return_value=False),
            patch(
                "cogniverse_cli.main.create_cluster",
                side_effect=ClusterStartError(
                    "Could not pin CoreDNS upstreams for k3d cluster 'cogniverse'"
                ),
            ),
        ):
            result = CliRunner().invoke(cli, ["up"])

        assert result.exit_code == 1
        assert "Could not pin CoreDNS upstreams" in result.output
        assert "Traceback" not in result.output


class TestUpImagePrune:
    """After building + importing images, `up` prunes the superseded
    generation so repeated deploys don't fill the disk into Vespa's feed
    block — best-effort, and never fatal to the deploy."""

    def _patches(self, *, existing_k8s: bool):
        return {
            "has_existing_k8s": existing_k8s,
            "cluster_exists": not existing_k8s,
            "check_prerequisites": [],
            "resolve_project_root": Path("/root"),
            "has_workspace_source": True,
            "dev_version": "0.1.dev99-gabc",
            "build_images": ["cogniverse/runtime-rocm:0.1.dev99-gabc"],
            "dev_image_set_values": {},
            "_probe_host_llm": False,
        }

    def _run(self, *, existing_k8s: bool, prune_side_effect=None):
        from contextlib import ExitStack

        vals = self._patches(existing_k8s=existing_k8s)
        returns = {
            "has_existing_k8s": vals["has_existing_k8s"],
            "cluster_exists": vals["cluster_exists"],
            "check_prerequisites": vals["check_prerequisites"],
            "resolve_project_root": vals["resolve_project_root"],
            "has_workspace_source": vals["has_workspace_source"],
            "dev_version": vals["dev_version"],
            "build_images": vals["build_images"],
            "dev_image_set_values": vals["dev_image_set_values"],
            "_probe_host_llm": vals["_probe_host_llm"],
            "get_values_file": Path("/v.yaml"),
            "get_chart_path": Path("/chart"),
            "get_workflows_path": Path("/wf"),
            "wait_for_url": True,
        }
        no_return = (
            "import_images",
            "verify_local_images_cover_deploy",
            "helm_install",
            "pull_and_import_third_party",
            "subprocess.run",
            "install_argo_controller",
            "deploy_workflow_templates",
            "_print_status_table",
        )
        with ExitStack() as stack:
            for name, ret in returns.items():
                stack.enter_context(
                    patch(f"cogniverse_cli.main.{name}", return_value=ret)
                )
            for name in no_return:
                stack.enter_context(patch(f"cogniverse_cli.main.{name}"))
            # Sourced from cogniverse_cli.secrets (local import in up()); patch
            # them there so the test never kubectl-applies to a live cluster.
            stack.enter_context(
                patch("cogniverse_cli.secrets.sync_hf_token_to_cluster")
            )
            stack.enter_context(
                patch("cogniverse_cli.secrets.sync_inference_api_key_to_cluster")
            )
            mock_prune = stack.enter_context(
                patch(
                    "cogniverse_cli.main.prune_superseded_images",
                    side_effect=prune_side_effect,
                )
            )
            result = CliRunner().invoke(cli, ["up"])
        return result, mock_prune

    def test_prunes_the_k3d_node_on_the_current_version(self):
        result, mock_prune = self._run(existing_k8s=False)
        assert result.exit_code == 0, result.output
        mock_prune.assert_called_once_with(
            "0.1.dev99-gabc", node_container="k3d-cogniverse-server-0"
        )

    def test_non_k3d_deploy_prunes_host_only(self):
        result, mock_prune = self._run(existing_k8s=True)
        assert result.exit_code == 0, result.output
        mock_prune.assert_called_once_with("0.1.dev99-gabc", node_container=None)

    def test_prune_failure_does_not_fail_the_deploy(self):
        result, mock_prune = self._run(
            existing_k8s=False, prune_side_effect=RuntimeError("docker gone")
        )
        assert result.exit_code == 0, result.output
        assert "Image prune skipped" in result.output
        mock_prune.assert_called_once()


class TestDownCommand:
    """Tests for the ``down`` command."""

    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    @patch("cogniverse_cli.main.delete_cluster")
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.helm_uninstall")
    @patch("cogniverse_cli.main.stop_port_forwards")
    def test_down_full_teardown(
        self,
        mock_stop_pf: MagicMock,
        mock_uninstall: MagicMock,
        mock_run: MagicMock,
        mock_delete: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        """Without --keep-data, removes release, namespace, and k3d cluster."""
        mock_run.return_value.returncode = 0
        runner = CliRunner()
        result = runner.invoke(cli, ["down"])
        assert result.exit_code == 0
        mock_uninstall.assert_called_once()
        mock_delete.assert_called_once()
        # kubectl delete namespace called twice (cogniverse + argo)
        assert mock_run.call_count == 2
        namespaces_deleted = [call[0][0][3] for call in mock_run.call_args_list]
        assert "cogniverse" in namespaces_deleted
        assert "argo" in namespaces_deleted

    @patch("cogniverse_cli.main.stop_port_forwards")
    @patch("cogniverse_cli.main.cluster_exists", return_value=False)
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.helm_uninstall")
    def test_down_surfaces_namespace_delete_failure(
        self,
        mock_uninstall: MagicMock,
        mock_run: MagicMock,
        mock_exists: MagicMock,
        mock_stop_pf: MagicMock,
    ) -> None:
        """A failed `kubectl delete namespace` surfaces stderr and exits
        nonzero — it previously printed "stack removed" and exited 0."""
        failing = MagicMock()
        failing.returncode = 1
        failing.stderr = "Error: connection refused"
        mock_run.return_value = failing

        result = CliRunner().invoke(cli, ["down"])

        assert result.exit_code != 0
        assert "connection refused" in result.output
        # Both namespaces are attempted before the nonzero exit.
        assert mock_run.call_count == 2
        assert "Cogniverse stack removed." not in result.output

    @patch("cogniverse_cli.main.stop_port_forwards")
    @patch("cogniverse_cli.main.helm_uninstall")
    def test_down_keep_data(
        self, mock_uninstall: MagicMock, mock_stop_pf: MagicMock
    ) -> None:
        """With --keep-data, only removes the Helm release."""
        runner = CliRunner()
        result = runner.invoke(cli, ["down", "--keep-data"])
        assert result.exit_code == 0
        mock_uninstall.assert_called_once()

    @patch("cogniverse_cli.main.stop_port_forwards")
    @patch("cogniverse_cli.main.helm_uninstall")
    def test_down_reaps_port_forwards(
        self, mock_uninstall: MagicMock, mock_stop_pf: MagicMock
    ) -> None:
        """Teardown reaps any running port-forward daemons exactly once."""
        runner = CliRunner()
        result = runner.invoke(cli, ["down", "--keep-data"])
        assert result.exit_code == 0
        mock_stop_pf.assert_called_once()


class TestStatusCommand:
    """Tests for the ``status`` command."""

    @patch("cogniverse_cli.main.check_service_health")
    def test_status_prints_table(self, mock_health: MagicMock) -> None:
        """Status command prints a table with all services."""
        mock_health.return_value = {name: False for name in SERVICE_HEALTH_URLS}
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        for name in SERVICE_ENDPOINTS:
            assert name in result.output


class TestLogsCommand:
    """Tests for the ``logs`` command."""

    @patch("cogniverse_cli.main.subprocess.run")
    def test_logs_runtime(self, mock_run: MagicMock) -> None:
        """Logs for runtime uses deployment resource."""
        mock_run.return_value.returncode = 0
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "runtime"])
        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "deployment/cogniverse-runtime" in cmd
        assert "-f" not in cmd

    @patch("cogniverse_cli.main.subprocess.run")
    def test_logs_vespa_follow(self, mock_run: MagicMock) -> None:
        """Logs for vespa with -f uses statefulset and follow flag."""
        mock_run.return_value.returncode = 0
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "vespa", "-f"])
        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "statefulset/cogniverse-vespa" in cmd
        assert "-f" in cmd

    @patch("cogniverse_cli.main._llm_statefulset_exists", return_value=False)
    def test_logs_llm_external_mode(self, mock_exists: MagicMock) -> None:
        """When LLM statefulset does not exist, prints message and returns."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "llm"])
        assert result.exit_code == 0
        assert "No builtin LLM pod found" in result.output

    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main._llm_statefulset_exists", return_value=True)
    def test_logs_llm_builtin_mode(
        self, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        """When LLM statefulset exists, shows logs from it."""
        mock_run.return_value.returncode = 0
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "llm"])
        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "statefulset/cogniverse-llm" in cmd

    @patch("cogniverse_cli.main.subprocess.run")
    def test_logs_argo_uses_argo_namespace(self, mock_run: MagicMock) -> None:
        """Argo logs use the 'argo' namespace, not 'cogniverse'."""
        mock_run.return_value.returncode = 0
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "argo"])
        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "deployment/argo-server" in cmd
        # Check namespace is "argo"
        ns_idx = cmd.index("-n")
        assert cmd[ns_idx + 1] == "argo"

    def test_logs_invalid_service(self) -> None:
        """Invalid service name is rejected by Click."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "invalid"])
        assert result.exit_code != 0


class TestServiceConstants:
    """Tests for service URL constants."""

    def test_health_urls_cover_all_endpoints(self) -> None:
        """Every service in SERVICE_ENDPOINTS has a health URL."""
        assert set(SERVICE_HEALTH_URLS.keys()) == set(SERVICE_ENDPOINTS.keys())

    def test_kubectl_resource_covers_log_services(self) -> None:
        """Every valid logs service has a kubectl resource mapping."""
        from cogniverse_cli.main import _SERVICE_KUBECTL_RESOURCE

        expected_services = {"runtime", "dashboard", "vespa", "phoenix", "llm", "argo"}
        assert set(_SERVICE_KUBECTL_RESOURCE.keys()) == expected_services


class TestStopStartCommands:
    @patch("cogniverse_cli.main.stop_cluster")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_stop_targets_named_cluster(
        self, mock_exists: MagicMock, mock_stop: MagicMock
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stop", "--name", "cogniverse-e2e"])

        assert result.exit_code == 0
        mock_stop.assert_called_once_with("cogniverse-e2e")

    @patch("cogniverse_cli.main.stop_cluster")
    @patch("cogniverse_cli.main.cluster_exists", return_value=False)
    def test_stop_unknown_cluster_fails(
        self, mock_exists: MagicMock, mock_stop: MagicMock
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stop", "--name", "nope"])

        assert result.exit_code != 0
        mock_stop.assert_not_called()

    @patch("cogniverse_cli.main.stop_port_forwards")
    @patch("cogniverse_cli.main.stop_cluster")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_stop_dev_cluster_reaps_port_forwards(
        self,
        mock_exists: MagicMock,
        mock_stop: MagicMock,
        mock_forwards: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stop"])

        assert result.exit_code == 0
        mock_stop.assert_called_once_with("cogniverse")
        mock_forwards.assert_called_once()

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main.start_cluster")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_start_dev_cluster_restores_port_forwards(
        self,
        mock_exists: MagicMock,
        mock_start: MagicMock,
        mock_forwards: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["start"])

        assert result.exit_code == 0
        mock_start.assert_called_once_with("cogniverse")
        mock_forwards.assert_called_once()

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main.start_cluster")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_start_e2e_cluster_skips_port_forwards(
        self,
        mock_exists: MagicMock,
        mock_start: MagicMock,
        mock_forwards: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["start", "--name", "cogniverse-e2e"])

        assert result.exit_code == 0
        mock_start.assert_called_once_with("cogniverse-e2e")
        mock_forwards.assert_not_called()

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_start_refuses_occupied_loadbalancer_port_before_k3d(
        self,
        mock_exists: MagicMock,
        mock_forwards: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            host_port = listener.getsockname()[1]
            marker = _install_fake_cluster_tools(
                tmp_path, monkeypatch, _cluster_state(host_port)
            )

            result = CliRunner().invoke(cli, ["start"])

        assert result.exit_code == 1
        assert marker.read_text() == "inspect\n"
        assert f"Host port {host_port} required by LLM (Ollama) is in use." in (
            result.output
        )
        assert (
            "k3d cannot remove or remap published ports on an existing cluster."
            in result.output
        )
        assert "Free or reconfigure the host listener, then retry:" in result.output
        assert "cogniverse start --name cogniverse" in result.output
        assert (
            f"recreate the cluster with host port {host_port} excluded or remapped."
            in result.output
        )
        assert "Cluster cogniverse started." not in result.output
        assert "Traceback" not in result.output
        mock_forwards.assert_not_called()

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_start_reports_mapping_inspection_failure_without_starting(
        self,
        mock_exists: MagicMock,
        mock_forwards: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        marker = _install_fake_cluster_tools(
            tmp_path,
            monkeypatch,
            _cluster_state(11434),
            inspect_returncode=17,
        )

        result = CliRunner().invoke(cli, ["start"])

        assert result.exit_code == 1
        assert marker.read_text() == "inspect\n"
        assert (
            "Could not inspect port mappings for k3d cluster 'cogniverse'"
            in result.output
        )
        assert "Cluster cogniverse started." not in result.output
        assert "Traceback" not in result.output
        mock_forwards.assert_not_called()

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_start_reports_k3d_failure_without_calledprocesserror_traceback(
        self,
        mock_exists: MagicMock,
        mock_forwards: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            host_port = probe.getsockname()[1]
        marker = _install_fake_cluster_tools(
            tmp_path,
            monkeypatch,
            _cluster_state(host_port),
            start_returncode=23,
            start_stderr="server started; serverlb failed",
        )

        result = CliRunner().invoke(cli, ["start"])

        assert result.exit_code == 1
        assert marker.read_text() == "inspect\nstart\n"
        assert (
            "Could not start k3d cluster 'cogniverse': server started; serverlb failed"
        ) in result.output
        assert "Cluster cogniverse started." not in result.output
        assert "Traceback" not in result.output
        assert not isinstance(result.exception, subprocess.CalledProcessError)
        mock_forwards.assert_not_called()

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_start_delegates_after_clean_preflight(
        self,
        mock_exists: MagicMock,
        mock_forwards: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            host_port = probe.getsockname()[1]
        marker = _install_fake_cluster_tools(
            tmp_path, monkeypatch, _cluster_state(host_port)
        )

        result = CliRunner().invoke(cli, ["start"])

        assert result.exit_code == 0, result.output
        assert marker.read_text() == "inspect\nstart\ninspect\n"
        assert result.output == (
            "Starting cluster cogniverse...\nCluster cogniverse started.\n"
        )
        mock_forwards.assert_called_once()

    @patch("cogniverse_cli.main.start_port_forwards")
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    def test_start_reports_loadbalancer_detached_from_cluster_network(
        self,
        mock_exists: MagicMock,
        mock_forwards: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            host_port = probe.getsockname()[1]
        stopped_state = _cluster_state(host_port)
        detached_state = json.loads(json.dumps(stopped_state))
        detached_state["nodes"][0]["State"] = {"Running": True}
        marker = _install_fake_cluster_tools(
            tmp_path,
            monkeypatch,
            stopped_state,
            post_start_cluster_state=detached_state,
        )

        result = CliRunner().invoke(cli, ["start"])

        assert result.exit_code == 1
        assert marker.read_text() == "inspect\nstart\ninspect\n"
        assert (
            "k3d cluster 'cogniverse' started, but load balancer "
            "'k3d-cogniverse-serverlb' is not attached to network "
            "'k3d-cogniverse'."
        ) in result.output
        assert "The cluster API and published services are unavailable." in (
            result.output
        )
        assert "Repair the existing cluster without recreating it:" in result.output
        assert (
            "docker network connect k3d-cogniverse k3d-cogniverse-serverlb"
            in result.output
        )
        assert "docker restart k3d-cogniverse-serverlb" in result.output
        assert "Cluster cogniverse started." not in result.output
        assert "Traceback" not in result.output
        mock_forwards.assert_not_called()

    def test_concurrent_preflights_do_not_share_cluster_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cogniverse_cli.cluster import ClusterStartError, start_cluster

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            host_port = listener.getsockname()[1]
            marker = _install_fake_cluster_tools(
                tmp_path, monkeypatch, _cluster_state(host_port)
            )

            def attempt_start() -> str:
                with pytest.raises(ClusterStartError) as exc:
                    start_cluster("cogniverse")
                return str(exc.value)

            with ThreadPoolExecutor(max_workers=2) as pool:
                messages = list(pool.map(lambda _: attempt_start(), range(2)))

        expected = f"Host port {host_port} required by LLM (Ollama) is in use."
        assert messages == [messages[0], messages[0]]
        assert expected in messages[0]
        assert marker.read_text().splitlines() == ["inspect", "inspect"]


class TestIndexCommandGate:
    """`cogniverse index` routes code AND docs through index_files — the docs
    pipeline is fully wired (extension→profile map, markdown graph
    extraction), and the old blanket gate refused it as "not yet
    implemented". Only video, which has no collector branch, stays gated."""

    def _invoke(self, tmp_path: Path, args: list[str], record: dict):
        import cogniverse_cli.index as index_mod

        def _fake_index_files(root, content_type, tenant_id, profile=None):
            record.update(root=root, content_type=content_type, tenant_id=tenant_id)

        with patch.object(index_mod, "index_files", _fake_index_files):
            runner = CliRunner()
            return runner.invoke(
                cli, ["index", str(tmp_path), *args, "--tenant", "acme:acme"]
            )

    def test_docs_type_reaches_index_files(self, tmp_path: Path) -> None:
        record: dict = {}
        result = self._invoke(tmp_path, ["--type", "docs"], record)
        assert result.exit_code == 0, result.output
        assert "not yet implemented" not in result.output
        assert record["content_type"] == "docs"
        assert record["tenant_id"] == "acme:acme"

    def test_code_type_reaches_index_files(self, tmp_path: Path) -> None:
        record: dict = {}
        result = self._invoke(tmp_path, ["--type", "code"], record)
        assert result.exit_code == 0, result.output
        assert record["content_type"] == "code"

    def test_video_type_stays_gated(self, tmp_path: Path) -> None:
        record: dict = {}
        result = self._invoke(tmp_path, ["--type", "video"], record)
        assert result.exit_code == 0
        assert "not yet implemented" in result.output
        assert record == {}


class TestLogsExitCode:
    def test_logs_propagates_kubectl_failure(self) -> None:
        """`cogniverse logs` exits with kubectl's code — a NotFound
        previously exited 0, invisible to wrapping scripts."""
        failing = MagicMock()
        failing.returncode = 1
        with patch("cogniverse_cli.main.subprocess.run", return_value=failing):
            result = CliRunner().invoke(cli, ["logs", "runtime"])
        assert result.exit_code == 1

    def test_logs_success_exits_0(self) -> None:
        ok = MagicMock()
        ok.returncode = 0
        with patch("cogniverse_cli.main.subprocess.run", return_value=ok):
            result = CliRunner().invoke(cli, ["logs", "runtime"])
        assert result.exit_code == 0

    def test_logs_missing_kubectl_exits_127(self) -> None:
        with patch(
            "cogniverse_cli.main.subprocess.run",
            side_effect=FileNotFoundError("kubectl"),
        ):
            result = CliRunner().invoke(cli, ["logs", "runtime"])
        assert result.exit_code == 127


class TestResolveCliTenant:
    def test_explicit_flag_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cogniverse_cli.main import _resolve_cli_tenant

        monkeypatch.setenv("COGNIVERSE_TENANT_ID", "env:tenant")
        assert _resolve_cli_tenant("flag:tenant") == "flag:tenant"

    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cogniverse_cli.main import _resolve_cli_tenant

        monkeypatch.setenv("COGNIVERSE_TENANT_ID", "env:tenant")
        assert _resolve_cli_tenant(None) == "env:tenant"

    def test_missing_tenant_is_a_clear_click_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import click
        from cogniverse_cli.main import _resolve_cli_tenant

        monkeypatch.delenv("COGNIVERSE_TENANT_ID", raising=False)
        with pytest.raises(click.ClickException) as exc:
            _resolve_cli_tenant(None)
        assert "--tenant" in str(exc.value)


class TestStatusClusterOutage:
    def test_docker_outage_is_named_not_flattened(self) -> None:
        """A docker/k3d outage prints why cluster listing failed instead of
        rendering as "no clusters"."""
        with (
            patch(
                "cogniverse_cli.main.list_cluster_states",
                side_effect=RuntimeError("docker daemon unreachable"),
            ),
            patch("cogniverse_cli.main._print_status_table"),
        ):
            result = CliRunner().invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Could not list k3d clusters" in result.output
        assert "docker daemon unreachable" in result.output


class TestUpImageSource:
    _PATCHES = [
        ("cogniverse_cli.main._print_status_table", {}),
        ("cogniverse_cli.main.deploy_workflow_templates", {}),
        ("cogniverse_cli.main.install_argo_controller", {}),
        ("cogniverse_cli.main.subprocess.run", {}),
        ("cogniverse_cli.main.wait_for_url", {"return_value": True}),
        ("cogniverse_cli.main.helm_install", {}),
        ("cogniverse_cli.main.pull_and_import_third_party", {}),
        ("cogniverse_cli.main.get_values_file", {"return_value": Path("/v.yaml")}),
        ("cogniverse_cli.main.get_chart_path", {"return_value": Path("/chart")}),
        ("cogniverse_cli.main.get_workflows_path", {"return_value": Path("/wf")}),
        ("cogniverse_cli.main._probe_host_llm", {"return_value": False}),
        (
            "cogniverse_cli.main.resolve_project_root",
            {"return_value": Path("/checkout")},
        ),
        ("cogniverse_cli.main.cluster_exists", {"return_value": True}),
        ("cogniverse_cli.main.check_prerequisites", {"return_value": []}),
        ("cogniverse_cli.main.has_existing_k8s", {"return_value": False}),
        ("cogniverse_cli.main.prune_superseded_images", {}),
        ("cogniverse_cli.main.import_images", {}),
        ("cogniverse_cli.main.dev_version", {"return_value": "0.1.dev1"}),
        ("cogniverse_cli.main.dev_image_set_values", {"return_value": {}}),
        ("cogniverse_cli.main.verify_local_images_cover_deploy", {}),
    ]

    def _invoke(self, args, workspace_ok=True):
        import contextlib

        with contextlib.ExitStack() as stack:
            mocks = {
                target.rsplit(".", 1)[1]: stack.enter_context(patch(target, **kwargs))
                for target, kwargs in self._PATCHES
            }
            mocks["has_workspace_source"] = stack.enter_context(
                patch(
                    "cogniverse_cli.main.has_workspace_source",
                    return_value=workspace_ok,
                )
            )
            mocks["build_images"] = stack.enter_context(
                patch("cogniverse_cli.main.build_images", return_value=["t:1"])
            )
            result = CliRunner().invoke(cli, args)
        return result, mocks

    def test_image_source_overrides_build_root(self, tmp_path: Path) -> None:
        """`up --image-source <dir>` builds from <dir> — the option was
        parsed and silently ignored, always building from the checkout."""
        result, mocks = self._invoke(["up", "--image-source", str(tmp_path)])
        assert result.exit_code == 0, result.output
        built_from = mocks["build_images"].call_args.args[0]
        assert built_from == tmp_path.resolve()
        assert mocks["dev_version"].call_args.args[0] == tmp_path.resolve()

    def test_default_builds_from_checkout(self) -> None:
        result, mocks = self._invoke(["up"])
        assert result.exit_code == 0, result.output
        assert mocks["build_images"].call_args.args[0] == Path("/checkout")

    def test_image_source_without_workspace_errors(self, tmp_path: Path) -> None:
        result, mocks = self._invoke(
            ["up", "--image-source", str(tmp_path)], workspace_ok=False
        )
        assert result.exit_code == 1
        assert "no buildable workspace" in " ".join(result.output.split())
        mocks["build_images"].assert_not_called()


class TestCodeCommand:
    """`cogniverse code` resolves the tenant, then hands off to the REPL loop
    with the parsed options."""

    def test_invokes_repl_with_parsed_args(self) -> None:
        with patch("cogniverse_cli.code.run_repl") as mock_repl:
            result = CliRunner().invoke(
                cli,
                [
                    "code",
                    "--tenant",
                    "acme:acme",
                    "-l",
                    "rust",
                    "-n",
                    "7",
                    "-c",
                    "/src/proj",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_repl.assert_called_once_with(
            tenant_id="acme:acme",
            language="rust",
            max_iterations=7,
            codebase_path="/src/proj",
        )

    def test_defaults_flow_through_to_repl(self) -> None:
        with patch("cogniverse_cli.code.run_repl") as mock_repl:
            result = CliRunner().invoke(cli, ["code", "--tenant", "acme:acme"])
        assert result.exit_code == 0, result.output
        mock_repl.assert_called_once_with(
            tenant_id="acme:acme",
            language="python",
            max_iterations=5,
            codebase_path="",
        )

    def test_missing_tenant_errors_before_repl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("COGNIVERSE_TENANT_ID", raising=False)
        with patch("cogniverse_cli.code.run_repl") as mock_repl:
            result = CliRunner().invoke(cli, ["code"])
        assert result.exit_code != 0
        assert "--tenant" in result.output
        mock_repl.assert_not_called()


class TestGraphWrappers:
    """The `graph` click wrappers forward parsed options to the cmd_*
    functions and propagate their exit codes to the shell."""

    def test_search_forwards_top_k_and_query(self) -> None:
        with patch("cogniverse_cli.graph.cmd_search", return_value=0) as m:
            result = CliRunner().invoke(
                cli,
                ["graph", "search", "my query", "--tenant", "acme:acme", "-k", "25"],
            )
        assert result.exit_code == 0
        m.assert_called_once_with("acme:acme", "my query", top_k=25)

    def test_search_propagates_nonzero_exit(self) -> None:
        with patch("cogniverse_cli.graph.cmd_search", return_value=3):
            result = CliRunner().invoke(
                cli, ["graph", "search", "q", "--tenant", "acme:acme"]
            )
        assert result.exit_code == 3

    def test_neighbors_forwards_node_and_depth(self) -> None:
        with patch("cogniverse_cli.graph.cmd_neighbors", return_value=0) as m:
            result = CliRunner().invoke(
                cli,
                ["graph", "neighbors", "Alice", "--tenant", "acme:acme", "-d", "3"],
            )
        assert result.exit_code == 0
        m.assert_called_once_with("acme:acme", "Alice", depth=3)

    def test_path_forwards_source_target_and_max_depth(self) -> None:
        with patch("cogniverse_cli.graph.cmd_path", return_value=0) as m:
            result = CliRunner().invoke(
                cli,
                ["graph", "path", "Alice", "Carol", "--tenant", "acme:acme", "-d", "6"],
            )
        assert result.exit_code == 0
        m.assert_called_once_with("acme:acme", "Alice", "Carol", max_depth=6)

    def test_path_propagates_nonzero_exit(self) -> None:
        with patch("cogniverse_cli.graph.cmd_path", return_value=4):
            result = CliRunner().invoke(
                cli, ["graph", "path", "A", "B", "--tenant", "acme:acme"]
            )
        assert result.exit_code == 4


class _ModalLifecycleStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.close_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self):
        self.close_calls += 1

    @staticmethod
    def _status(service: str, active_containers: int):
        status = MagicMock()
        status.service = service
        status.web_url = f"https://{service}.modal.run"
        status.active_containers = active_containers
        return status

    def deploy(self, services):
        self.calls.append(("deploy", services))
        return tuple(self._status(service, 0) for service in services)

    def warm(self, services):
        self.calls.append(("warm", services))
        endpoints = []
        for service in services:
            endpoint = MagicMock()
            endpoint.service = service
            endpoint.base_url = f"https://{service}.modal.run"
            endpoint.model_id = f"model/{service}"
            endpoints.append(endpoint)
        return tuple(endpoints)

    def release(self, services):
        self.calls.append(("release", services))
        return tuple(self._status(service, 0) for service in services)

    def status(self, services):
        self.calls.append(("status", services))
        return tuple(self._status(service, 1) for service in services)

    def qualify(self, service, candidates):
        self.calls.append(("qualify", (service, candidates)))
        result = MagicMock()
        result.service = service
        result.selected_gpu = "L4"
        result.considered_gpus = ("L4", "A10")
        return result

    def undeploy(self, service, confirmation):
        self.calls.append(("undeploy", (service, confirmation)))


class TestModalInferenceCommands:
    @pytest.mark.parametrize(
        ("operation", "expected_output", "expected_active"),
        [
            (
                "deploy",
                "vllm_colpali: https://vllm_colpali.modal.run (active_containers=0)",
                0,
            ),
            (
                "release",
                "vllm_colpali: https://vllm_colpali.modal.run (active_containers=0)",
                0,
            ),
            (
                "status",
                "vllm_colpali: https://vllm_colpali.modal.run (active_containers=1)",
                1,
            ),
        ],
    )
    def test_status_commands_forward_all_services_and_print_exact_state(
        self,
        operation: str,
        expected_output: str,
        expected_active: int,
    ) -> None:
        lifecycle = _ModalLifecycleStub()
        with patch(
            "cogniverse_cli.main._build_modal_inference_lifecycle",
            return_value=lifecycle,
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "inference",
                    "modal",
                    operation,
                    "vllm_colpali",
                    "denseon",
                ],
            )

        assert result.exit_code == 0, result.output
        assert lifecycle.calls == [(operation, ("vllm_colpali", "denseon"))]
        assert lifecycle.close_calls == 1
        assert result.output.splitlines() == [
            expected_output,
            f"denseon: https://denseon.modal.run (active_containers={expected_active})",
        ]

    def test_warm_prints_the_verified_model_endpoint_and_live_runner_count(
        self,
    ) -> None:
        lifecycle = _ModalLifecycleStub()
        with patch(
            "cogniverse_cli.main._build_modal_inference_lifecycle",
            return_value=lifecycle,
        ):
            result = CliRunner().invoke(
                cli,
                ["inference", "modal", "warm", "vllm_colpali"],
            )

        assert result.exit_code == 0, result.output
        assert lifecycle.calls == [
            ("warm", ("vllm_colpali",)),
            ("status", ("vllm_colpali",)),
        ]
        assert lifecycle.close_calls == 1
        assert result.output == (
            "vllm_colpali: https://vllm_colpali.modal.run "
            "(model=model/vllm_colpali, active_containers=1)\n"
        )

    def test_qualify_forwards_candidates_and_prints_ordered_decision(self) -> None:
        lifecycle = _ModalLifecycleStub()
        with patch(
            "cogniverse_cli.main._build_modal_inference_lifecycle",
            return_value=lifecycle,
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "inference",
                    "modal",
                    "qualify",
                    "vllm_colpali",
                    "--gpu",
                    "A10",
                    "--gpu",
                    "L4",
                ],
            )

        assert result.exit_code == 0, result.output
        assert lifecycle.calls == [("qualify", ("vllm_colpali", ("A10", "L4")))]
        assert lifecycle.close_calls == 1
        assert result.output == ("vllm_colpali: selected L4 from L4, A10\n")

    def test_undeploy_requires_and_forwards_byte_exact_confirmation(self) -> None:
        lifecycle = _ModalLifecycleStub()
        with patch(
            "cogniverse_cli.main._build_modal_inference_lifecycle",
            return_value=lifecycle,
        ):
            missing = CliRunner().invoke(
                cli,
                ["inference", "modal", "undeploy", "vllm_colpali"],
            )
            confirmed = CliRunner().invoke(
                cli,
                [
                    "inference",
                    "modal",
                    "undeploy",
                    "vllm_colpali",
                    "--confirm-service",
                    "vllm_colpali",
                ],
            )

        assert missing.exit_code == 2
        assert "Missing option '--confirm-service'" in missing.output
        assert confirmed.exit_code == 0, confirmed.output
        assert lifecycle.calls == [("undeploy", ("vllm_colpali", "vllm_colpali"))]
        assert lifecycle.close_calls == 1
        assert confirmed.output == "vllm_colpali: undeployed\n"

    def test_lifecycle_error_is_a_concise_cli_error_without_traceback(self) -> None:
        from cogniverse_cli.modal_inference_lifecycle import ModalLifecycleError

        lifecycle = _ModalLifecycleStub()
        lifecycle.warm = MagicMock(
            side_effect=ModalLifecycleError("vllm_colpali: probe denied for [redacted]")
        )
        with patch(
            "cogniverse_cli.main._build_modal_inference_lifecycle",
            return_value=lifecycle,
        ):
            result = CliRunner().invoke(
                cli,
                ["inference", "modal", "warm", "vllm_colpali"],
            )

        assert result.exit_code == 1
        assert result.output == ("Error: vllm_colpali: probe denied for [redacted]\n")
        assert "Traceback" not in result.output
        assert lifecycle.close_calls == 1

    def test_factory_reads_the_bearer_key_from_process_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import cogniverse_cli.main as main_module

        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "expected-key")
        with patch(
            "cogniverse_cli.modal_inference_lifecycle.ModalInferenceLifecycle"
        ) as lifecycle_class:
            built = main_module._build_modal_inference_lifecycle()

        assert built is lifecycle_class.return_value
        credentials = lifecycle_class.call_args.kwargs["credentials"]
        assert credentials.bearer_token == "expected-key"
