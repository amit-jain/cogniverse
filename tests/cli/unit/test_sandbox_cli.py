"""Unit tests for the sandbox CLI subgroup and cogniverse_cli.sandbox helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from cogniverse_cli import sandbox as sandbox_mod
from cogniverse_cli.main import cli


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestSandboxStatusCommand:
    """Tests for ``cogniverse sandbox status``."""

    @patch("cogniverse_cli.sandbox.openshell_installed", return_value=False)
    def test_status_not_installed(self, mock_installed: MagicMock) -> None:
        """Without the openshell CLI, prints the not-installed message only."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sandbox", "status"])
        assert result.exit_code == 0
        assert result.output.splitlines() == ["openshell CLI not installed"]

    @patch("cogniverse_cli.sandbox.get_active_gateway_dir", return_value=None)
    @patch("cogniverse_cli.sandbox.openshell_installed", return_value=True)
    def test_status_no_active_gateway(
        self, mock_installed: MagicMock, mock_dir: MagicMock
    ) -> None:
        """Installed but no active gateway prints the no-gateway line only."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sandbox", "status"])
        assert result.exit_code == 0
        assert result.output.splitlines() == ["No active openshell gateway"]

    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.sandbox.gateway_running", return_value=True)
    @patch(
        "cogniverse_cli.sandbox.get_active_gateway_dir",
        return_value=Path("/cfg/gateways/cogniverse"),
    )
    @patch("cogniverse_cli.sandbox.openshell_installed", return_value=True)
    def test_status_running_and_synced(
        self,
        mock_installed: MagicMock,
        mock_dir: MagicMock,
        mock_running: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Healthy gateway with the mTLS secret present renders all four lines."""
        mock_run.return_value = _completed(returncode=0)
        runner = CliRunner()
        result = runner.invoke(cli, ["sandbox", "status"])
        assert result.exit_code == 0
        assert result.output.splitlines() == [
            "Active gateway: cogniverse",
            "  Config: /cfg/gateways/cogniverse",
            "  Running: yes",
            "  Synced to cluster: yes",
        ]
        kubectl_cmd = mock_run.call_args[0][0]
        assert kubectl_cmd == [
            "kubectl",
            "get",
            "secret",
            "openshell-mtls",
            "-n",
            "cogniverse",
        ]

    @patch("cogniverse_cli.main.subprocess.run")
    @patch("cogniverse_cli.sandbox.gateway_running", return_value=False)
    @patch(
        "cogniverse_cli.sandbox.get_active_gateway_dir",
        return_value=Path("/cfg/gateways/cogniverse"),
    )
    @patch("cogniverse_cli.sandbox.openshell_installed", return_value=True)
    def test_status_stopped_and_unsynced(
        self,
        mock_installed: MagicMock,
        mock_dir: MagicMock,
        mock_running: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Stopped gateway with no cluster secret renders no/no."""
        mock_run.return_value = _completed(returncode=1)
        runner = CliRunner()
        result = runner.invoke(cli, ["sandbox", "status"])
        assert result.exit_code == 0
        assert result.output.splitlines() == [
            "Active gateway: cogniverse",
            "  Config: /cfg/gateways/cogniverse",
            "  Running: no",
            "  Synced to cluster: no",
        ]


class TestSandboxSyncCommand:
    """Tests for ``cogniverse sandbox sync``."""

    @patch("cogniverse_cli.sandbox.sync_gateway_certs_to_cluster", return_value=True)
    def test_sync_success(self, mock_sync: MagicMock) -> None:
        """Successful sync calls the helper once and prints the restart hint."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sandbox", "sync"])
        assert result.exit_code == 0
        mock_sync.assert_called_once_with()
        assert (
            "Sandbox certs synced. Restart runtime to pick up changes." in result.output
        )

    @patch("cogniverse_cli.sandbox.sync_gateway_certs_to_cluster", return_value=False)
    def test_sync_failure(self, mock_sync: MagicMock) -> None:
        """Failed sync exits nonzero with the click error text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sandbox", "sync"])
        assert result.exit_code == 1
        mock_sync.assert_called_once_with()
        assert "Error: Failed to sync openshell certs" in result.output


class TestOpenshellInstalled:
    """Tests for :func:`cogniverse_cli.sandbox.openshell_installed`."""

    def test_found_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sandbox_mod.shutil, "which", lambda name: "/usr/bin/openshell"
        )
        assert sandbox_mod.openshell_installed() is True

    def test_missing_everywhere(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: None)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert sandbox_mod.openshell_installed() is False

    def test_fallback_install_location(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: None)
        monkeypatch.setenv("HOME", str(tmp_path))
        fallback = tmp_path / ".local" / "bin" / "openshell"
        fallback.parent.mkdir(parents=True)
        fallback.touch()
        assert sandbox_mod.openshell_installed() is True


class TestGetActiveGatewayDir:
    """Tests for :func:`cogniverse_cli.sandbox.get_active_gateway_dir`."""

    @pytest.fixture(autouse=True)
    def _home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setenv("HOME", str(tmp_path))
        self.config_root = tmp_path / ".config" / "openshell"
        return tmp_path

    def test_no_active_file(self) -> None:
        assert sandbox_mod.get_active_gateway_dir() is None

    def test_blank_active_name(self) -> None:
        self.config_root.mkdir(parents=True)
        (self.config_root / "active_gateway").write_text("  \n")
        assert sandbox_mod.get_active_gateway_dir() is None

    def test_named_gateway_dir_missing(self) -> None:
        self.config_root.mkdir(parents=True)
        (self.config_root / "active_gateway").write_text("cogniverse\n")
        assert sandbox_mod.get_active_gateway_dir() is None

    def test_active_gateway_resolves(self) -> None:
        gateway_dir = self.config_root / "gateways" / "cogniverse"
        gateway_dir.mkdir(parents=True)
        (self.config_root / "active_gateway").write_text("cogniverse\n")
        assert sandbox_mod.get_active_gateway_dir() == gateway_dir


class TestGatewayRunning:
    """Tests for :func:`cogniverse_cli.sandbox.gateway_running`."""

    def test_not_installed_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox_mod, "openshell_installed", lambda: False)
        run = MagicMock()
        monkeypatch.setattr(sandbox_mod.subprocess, "run", run)
        assert sandbox_mod.gateway_running() is False
        run.assert_not_called()

    def test_healthy_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_mod, "openshell_installed", lambda: True)
        run = MagicMock(
            return_value=_completed(
                returncode=0,
                stdout="Gateway endpoint: https://localhost:19091\n",
            )
        )
        monkeypatch.setattr(sandbox_mod.subprocess, "run", run)
        assert sandbox_mod.gateway_running() is True
        assert run.call_args[0][0] == ["openshell", "gateway", "info"]

    def test_exit_zero_without_endpoint_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox_mod, "openshell_installed", lambda: True)
        run = MagicMock(return_value=_completed(returncode=0, stdout="no gateway\n"))
        monkeypatch.setattr(sandbox_mod.subprocess, "run", run)
        assert sandbox_mod.gateway_running() is False

    def test_nonzero_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_mod, "openshell_installed", lambda: True)
        run = MagicMock(return_value=_completed(returncode=1))
        monkeypatch.setattr(sandbox_mod.subprocess, "run", run)
        assert sandbox_mod.gateway_running() is False

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_mod, "openshell_installed", lambda: True)
        run = MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd="openshell", timeout=10)
        )
        monkeypatch.setattr(sandbox_mod.subprocess, "run", run)
        assert sandbox_mod.gateway_running() is False

    def test_binary_vanished(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_mod, "openshell_installed", lambda: True)
        run = MagicMock(side_effect=FileNotFoundError("openshell"))
        monkeypatch.setattr(sandbox_mod.subprocess, "run", run)
        assert sandbox_mod.gateway_running() is False


class TestSyncGatewayCertsToCluster:
    """Tests for :func:`cogniverse_cli.sandbox.sync_gateway_certs_to_cluster`."""

    @pytest.fixture(autouse=True)
    def _home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        self.config_root = tmp_path / ".config" / "openshell"

    def _build_gateway(self, *, metadata: bool = True, certs: bool = True) -> Path:
        gateway_dir = self.config_root / "gateways" / "cogniverse"
        gateway_dir.mkdir(parents=True)
        (self.config_root / "active_gateway").write_text("cogniverse\n")
        if metadata:
            (gateway_dir / "metadata.json").write_text(
                json.dumps(
                    {"gateway_endpoint": "https://localhost:19091", "token": "t"}
                )
            )
        if certs:
            mtls_dir = gateway_dir / "mtls"
            mtls_dir.mkdir()
            for name in ("ca.crt", "tls.crt", "tls.key"):
                (mtls_dir / name).write_text(name)
        return gateway_dir

    def test_no_active_gateway_fails(self) -> None:
        assert sandbox_mod.sync_gateway_certs_to_cluster() is False

    def test_missing_metadata_fails(self) -> None:
        self._build_gateway(metadata=False)
        assert sandbox_mod.sync_gateway_certs_to_cluster() is False

    def test_missing_cert_fails(self) -> None:
        gateway_dir = self._build_gateway()
        (gateway_dir / "mtls" / "tls.key").unlink()
        assert sandbox_mod.sync_gateway_certs_to_cluster() is False

    def test_happy_path_applies_secret_and_configmaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway_dir = self._build_gateway()
        mtls_dir = gateway_dir / "mtls"

        kubectl_calls: list[tuple[list[str], str | None]] = []

        def fake_kubectl(args: list[str], *, input_data: str | None = None):
            kubectl_calls.append((args, input_data))
            return _completed(returncode=0, stdout="rendered-yaml")

        metadata_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            metadata_cmds.append(cmd)
            return _completed(returncode=0, stdout="metadata-yaml")

        monkeypatch.setattr(sandbox_mod, "_kubectl", fake_kubectl)
        monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)

        assert sandbox_mod.sync_gateway_certs_to_cluster() is True

        assert kubectl_calls[0] == (["get", "namespace", "cogniverse"], None)
        assert kubectl_calls[1] == (
            [
                "create",
                "secret",
                "generic",
                "openshell-mtls",
                "-n",
                "cogniverse",
                f"--from-file=ca.crt={mtls_dir / 'ca.crt'}",
                f"--from-file=tls.crt={mtls_dir / 'tls.crt'}",
                f"--from-file=tls.key={mtls_dir / 'tls.key'}",
                "--dry-run=client",
                "-o",
                "yaml",
            ],
            None,
        )
        assert kubectl_calls[2] == (["apply", "-f", "-"], "rendered-yaml")

        expected_pod_metadata = json.dumps(
            {"gateway_endpoint": "https://host.docker.internal:19091", "token": "t"}
        )
        assert metadata_cmds == [
            [
                "kubectl",
                "create",
                "configmap",
                "openshell-metadata",
                "-n",
                "cogniverse",
                f"--from-literal=metadata.json={expected_pod_metadata}",
                "--dry-run=client",
                "-o",
                "yaml",
            ]
        ]
        assert kubectl_calls[3] == (["apply", "-f", "-"], "metadata-yaml")

        assert kubectl_calls[4] == (
            [
                "create",
                "configmap",
                "openshell-active",
                "-n",
                "cogniverse",
                "--from-literal=active_gateway=cogniverse",
                "--dry-run=client",
                "-o",
                "yaml",
            ],
            None,
        )
        assert kubectl_calls[5] == (["apply", "-f", "-"], "rendered-yaml")
        assert len(kubectl_calls) == 6

    def test_secret_apply_failure_stops_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._build_gateway()

        def fake_kubectl(args: list[str], *, input_data: str | None = None):
            if args[:1] == ["apply"]:
                return _completed(returncode=1, stderr="denied")
            return _completed(returncode=0, stdout="rendered-yaml")

        run = MagicMock()
        monkeypatch.setattr(sandbox_mod, "_kubectl", fake_kubectl)
        monkeypatch.setattr(sandbox_mod.subprocess, "run", run)

        assert sandbox_mod.sync_gateway_certs_to_cluster() is False
        run.assert_not_called()
