"""Unit tests for cogniverse_cli.cluster lifecycle utilities."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from cogniverse_cli.cluster import (
    DEFAULT_PORTS,
    check_prerequisites,
    cluster_exists,
    create_cluster,
    has_existing_k8s,
)


class TestCheckPrerequisites:
    """Tests for :func:`check_prerequisites`."""

    @patch("cogniverse_cli.cluster.shutil.which")
    def test_check_prerequisites_all_present(self, mock_which: object) -> None:
        """When all tools are on PATH, the missing list is empty."""
        mock_which.return_value = "/usr/local/bin/tool"  # type: ignore[attr-defined]

        result = check_prerequisites(require_k3d=True)

        assert result == []

    @patch("cogniverse_cli.cluster.shutil.which")
    def test_check_prerequisites_missing_k3d(self, mock_which: object) -> None:
        """When k3d is missing, it appears in the result list."""

        def _side_effect(name: str) -> str | None:
            if name == "k3d":
                return None
            return f"/usr/local/bin/{name}"

        mock_which.side_effect = _side_effect  # type: ignore[attr-defined]

        result = check_prerequisites(require_k3d=True)

        assert result == ["k3d"]


class TestClusterExists:
    """Tests for :func:`cluster_exists`."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_cluster_exists_true(self, mock_run: object) -> None:
        """Returns True when k3d reports the cluster exists."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        assert cluster_exists("cogniverse") is True

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_cluster_exists_false(self, mock_run: object) -> None:
        """Returns False when k3d reports the cluster does not exist."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=1
        )

        assert cluster_exists("cogniverse") is False


class TestCreateCluster:
    """Tests for :func:`create_cluster`."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_create_cluster_builds_correct_command(self, mock_run: object) -> None:
        """All default ports produce -p flags in the subprocess command."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        create_cluster("cogniverse")

        call_args = mock_run.call_args  # type: ignore[attr-defined]
        cmd = call_args[0][0]  # positional arg 0 is the command list

        assert cmd[:4] == ["k3d", "cluster", "create", "cogniverse"]

        # Each default port should produce a -p flag
        port_flags = [cmd[i + 1] for i in range(len(cmd)) if cmd[i] == "-p"]
        assert len(port_flags) == len(DEFAULT_PORTS)
        for port in DEFAULT_PORTS:
            assert f"{port}:{port}@loadbalancer" in port_flags

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_host_node_port_pairs_map_asymmetrically(self, mock_run: object) -> None:
        """A "host:node" string entry maps a different host port onto a chart
        NodePort — the e2e stack's scheme (33xxx host side, canonical node
        side) — while plain ints keep the 1:1 mapping."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        create_cluster("cogniverse-e2e", ports=["33000:28000", 8080])

        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        port_flags = [cmd[i + 1] for i in range(len(cmd)) if cmd[i] == "-p"]
        assert port_flags == [
            "33000:28000@loadbalancer",
            "8080:8080@loadbalancer",
        ]

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_env_override_replaces_default_ports(
        self, mock_run: object, monkeypatch
    ) -> None:
        """COGNIVERSE_K3D_PORTS replaces DEFAULT_PORTS entirely."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )
        monkeypatch.setenv("COGNIVERSE_K3D_PORTS", "5000,5001,5002")
        monkeypatch.delenv("COGNIVERSE_K3D_EXTRA_PORTS", raising=False)
        monkeypatch.delenv("COGNIVERSE_K3D_EXCLUDE_PORTS", raising=False)

        create_cluster("cogniverse")

        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        port_flags = [cmd[i + 1] for i in range(len(cmd)) if cmd[i] == "-p"]
        assert sorted(port_flags) == [
            "5000:5000@loadbalancer",
            "5001:5001@loadbalancer",
            "5002:5002@loadbalancer",
        ]

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_extra_ports_env_appends_to_defaults(
        self, mock_run: object, monkeypatch
    ) -> None:
        """COGNIVERSE_K3D_EXTRA_PORTS adds to DEFAULT_PORTS."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )
        monkeypatch.delenv("COGNIVERSE_K3D_PORTS", raising=False)
        monkeypatch.setenv("COGNIVERSE_K3D_EXTRA_PORTS", "9999,7777")
        monkeypatch.delenv("COGNIVERSE_K3D_EXCLUDE_PORTS", raising=False)

        create_cluster("cogniverse")

        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        port_flags = [cmd[i + 1] for i in range(len(cmd)) if cmd[i] == "-p"]
        assert "9999:9999@loadbalancer" in port_flags
        assert "7777:7777@loadbalancer" in port_flags
        for port in DEFAULT_PORTS:
            assert f"{port}:{port}@loadbalancer" in port_flags

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_exclude_ports_env_drops_from_set(
        self, mock_run: object, monkeypatch
    ) -> None:
        """COGNIVERSE_K3D_EXCLUDE_PORTS drops listed ports from the published set."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )
        sample_drop = DEFAULT_PORTS[0]
        monkeypatch.delenv("COGNIVERSE_K3D_PORTS", raising=False)
        monkeypatch.delenv("COGNIVERSE_K3D_EXTRA_PORTS", raising=False)
        monkeypatch.setenv("COGNIVERSE_K3D_EXCLUDE_PORTS", str(sample_drop))

        create_cluster("cogniverse")

        cmd = mock_run.call_args[0][0]  # type: ignore[attr-defined]
        port_flags = [cmd[i + 1] for i in range(len(cmd)) if cmd[i] == "-p"]
        assert f"{sample_drop}:{sample_drop}@loadbalancer" not in port_flags
        assert len(port_flags) == len(DEFAULT_PORTS) - 1


class TestHasExistingK8s:
    """Tests for :func:`has_existing_k8s`."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_has_existing_k8s_true(self, mock_run: object) -> None:
        """Returns True when kubectl cluster-info succeeds."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        assert has_existing_k8s() is True

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_has_existing_k8s_false(self, mock_run: object) -> None:
        """Returns False when kubectl cluster-info fails with non-zero exit."""
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=1
        )

        assert has_existing_k8s() is False


class TestStopStartCluster:
    """stop/start wrap k3d so cluster pause/resume is a first-class CLI
    operation instead of a raw k3d invocation."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_stop_cluster_invokes_k3d_stop(self, mock_run: object) -> None:
        from cogniverse_cli.cluster import stop_cluster

        stop_cluster("cogniverse-e2e")

        args = mock_run.call_args
        assert args.args[0] == ["k3d", "cluster", "stop", "cogniverse-e2e"]
        assert args.kwargs["check"] is True

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_start_cluster_invokes_k3d_start(self, mock_run: object) -> None:
        from cogniverse_cli.cluster import start_cluster

        start_cluster()

        args = mock_run.call_args
        assert args.args[0] == ["k3d", "cluster", "start", "cogniverse"]
        assert args.kwargs["check"] is True

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_list_cluster_states_parses_running_counts(self, mock_run: object) -> None:
        from cogniverse_cli.cluster import list_cluster_states

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '[{"name": "cogniverse", "serversRunning": 0, "serversCount": 1},'
                ' {"name": "cogniverse-e2e", "serversRunning": 1, "serversCount": 1}]'
            ),
        )

        states = list_cluster_states()

        assert states == [
            {"name": "cogniverse", "servers_running": 0, "servers_count": 1},
            {"name": "cogniverse-e2e", "servers_running": 1, "servers_count": 1},
        ]
