"""Unit tests for cogniverse_cli.main CLI entrypoint."""

from __future__ import annotations

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
        # up() imports sync_hf_token_to_cluster from cogniverse_cli.secrets and
        # kubectl-applies the HF token; without this, the full-flow tests
        # mutate a live k3d cluster's hf-token Secret on a dev box.
        with patch("cogniverse_cli.secrets.sync_hf_token_to_cluster"):
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
    def test_up_k3d_builtin_llm(
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
    @patch("cogniverse_cli.main._probe_host_llm", return_value=True)
    @patch("cogniverse_cli.main.has_workspace_source", return_value=False)
    @patch("cogniverse_cli.main.resolve_project_root", return_value=Path("/root"))
    @patch("cogniverse_cli.main.cluster_exists", return_value=True)
    @patch("cogniverse_cli.main.check_prerequisites", return_value=[])
    @patch("cogniverse_cli.main.has_existing_k8s", return_value=False)
    def test_up_k3d_auto_detects_host_llm(
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
    def test_up_existing_k8s_uses_prod_values(
        self,
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
            # it there so the test never kubectl-applies to a live cluster.
            stack.enter_context(
                patch("cogniverse_cli.secrets.sync_hf_token_to_cluster")
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
    def test_down_full_teardown(
        self,
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

    @patch("cogniverse_cli.main.cluster_exists", return_value=False)
    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.main.helm_uninstall")
    def test_down_surfaces_namespace_delete_failure(
        self,
        mock_uninstall: MagicMock,
        mock_run: MagicMock,
        mock_exists: MagicMock,
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

    @patch("cogniverse_cli.main.helm_uninstall")
    def test_down_keep_data(self, mock_uninstall: MagicMock) -> None:
        """With --keep-data, only removes the Helm release."""
        runner = CliRunner()
        result = runner.invoke(cli, ["down", "--keep-data"])
        assert result.exit_code == 0
        mock_uninstall.assert_called_once()


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
