"""Unit tests for cogniverse_cli.main CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
        for cmd in ("up", "down", "status", "logs"):
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
        # No LLM overrides when builtin
        call_kwargs = mock_helm.call_args
        assert call_kwargs[1].get("set_values") is None

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
        """When auto-detect finds host LLM on k3d, LLM overrides use host.docker.internal."""
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
        assert "host.docker.internal" in set_vals["llm.external.url"]

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
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "runtime"])
        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "deployment/cogniverse-runtime" in cmd
        assert "-f" not in cmd

    @patch("cogniverse_cli.main.subprocess.run")
    def test_logs_vespa_follow(self, mock_run: MagicMock) -> None:
        """Logs for vespa with -f uses statefulset and follow flag."""
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
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "llm"])
        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "statefulset/cogniverse-llm" in cmd

    @patch("cogniverse_cli.main.subprocess.run")
    def test_logs_argo_uses_argo_namespace(self, mock_run: MagicMock) -> None:
        """Argo logs use the 'argo' namespace, not 'cogniverse'."""
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
