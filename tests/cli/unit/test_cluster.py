"""Unit tests for cogniverse_cli.cluster lifecycle utilities."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import cogniverse_cli.cluster as cluster
import pytest
from cogniverse_cli.cluster import (
    DEFAULT_PORTS,
    check_prerequisites,
    cluster_exists,
    create_cluster,
    delete_cluster,
    has_existing_k8s,
    install_missing_prerequisites,
    install_prerequisite,
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


class TestClusterExistsBinaryGuard:
    def test_missing_k3d_reads_as_no_cluster(self):
        """A missing k3d binary means "no cluster" (same guard as
        has_existing_k8s) so `up` reaches its install prompt — it
        previously died with a FileNotFoundError traceback first."""
        from cogniverse_cli.cluster import cluster_exists

        with patch(
            "cogniverse_cli.cluster.subprocess.run",
            side_effect=FileNotFoundError("k3d"),
        ):
            assert cluster_exists("anything") is False

    def test_hung_k3d_reads_as_no_cluster(self):
        from cogniverse_cli.cluster import cluster_exists

        with patch(
            "cogniverse_cli.cluster.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="k3d", timeout=10),
        ):
            assert cluster_exists("anything") is False


class TestParsePortCsv:
    def test_valid_list_with_blanks(self):
        from cogniverse_cli.cluster import _parse_port_csv

        assert _parse_port_csv("80, 90,,443") == [80, 90, 443]
        assert _parse_port_csv("") == []
        assert _parse_port_csv(None) == []

    def test_non_numeric_entry_aborts_with_clear_message(self):
        from cogniverse_cli.cluster import _parse_port_csv

        with pytest.raises(SystemExit) as se:
            _parse_port_csv("80,abc")
        assert "abc" in str(se.value)
        assert "integer" in str(se.value)


class TestDeleteCluster:
    """`delete_cluster` runs `k3d cluster delete <name>` bounded and
    propagating — a failed delete must not read as success."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_invokes_k3d_delete_for_named_cluster(self, mock_run: object) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        delete_cluster("cogniverse-e2e")

        assert mock_run.call_args.args[0] == [  # type: ignore[attr-defined]
            "k3d",
            "cluster",
            "delete",
            "cogniverse-e2e",
        ]
        assert mock_run.call_args.kwargs["check"] is True  # type: ignore[attr-defined]
        assert mock_run.call_args.kwargs["timeout"] == 60  # type: ignore[attr-defined]

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_defaults_to_the_cogniverse_cluster(self, mock_run: object) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        delete_cluster()

        assert mock_run.call_args.args[0] == [  # type: ignore[attr-defined]
            "k3d",
            "cluster",
            "delete",
            "cogniverse",
        ]

    @patch(
        "cogniverse_cli.cluster.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "k3d"),
    )
    def test_propagates_delete_failure(self, mock_run: object) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            delete_cluster()


class TestStartPortForwards:
    """`start_port_forwards` spawns one detached kubectl port-forward per
    spec and records their PIDs for cross-process cleanup."""

    def test_spawns_kubectl_port_forward_and_writes_pids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        pid_file = tmp_path / "pf.pids"
        monkeypatch.setattr(cluster, "PID_FILE", str(pid_file))
        monkeypatch.setattr(cluster, "_port_forward_procs", [])

        popen_calls: list[tuple[list[str], dict]] = []

        def fake_popen(cmd, **kwargs):
            popen_calls.append((cmd, kwargs))
            proc = MagicMock()
            proc.pid = 4001
            return proc

        monkeypatch.setattr(cluster.subprocess, "Popen", fake_popen)

        cluster.start_port_forwards()

        # One forward per spec; argo-server is the only spec today.
        assert len(popen_calls) == len(cluster.PORT_FORWARD_SPECS)
        cmd, kwargs = popen_calls[0]
        assert cmd[:2] == ["sh", "-c"]
        assert "kubectl port-forward svc/argo-server 2746:2746 -n argo" in cmd[2]
        # Detached so the forwards survive the CLI exiting.
        assert kwargs["start_new_session"] is True
        assert pid_file.read_text() == "4001"
        assert len(cluster._port_forward_procs) == 1


class TestInstallPrerequisite:
    """`install_prerequisite` shells out the platform install command and
    maps its exit code to a success bool."""

    @patch("cogniverse_cli.cluster.subprocess.run")
    @patch("cogniverse_cli.cluster.platform.system", return_value="Linux")
    def test_runs_bash_install_and_reports_success(
        self, mock_sys: object, mock_run: object
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=0
        )

        assert install_prerequisite("k3d") is True

        cmd = mock_run.call_args.args[0]  # type: ignore[attr-defined]
        assert cmd[:2] == ["bash", "-c"]
        assert "k3d" in cmd[2]
        assert mock_run.call_args.kwargs["timeout"] == 300  # type: ignore[attr-defined]
        assert mock_run.call_args.kwargs["check"] is False  # type: ignore[attr-defined]

    @patch("cogniverse_cli.cluster.subprocess.run")
    @patch("cogniverse_cli.cluster.platform.system", return_value="Linux")
    def test_nonzero_exit_reports_failure(
        self, mock_sys: object, mock_run: object
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[], returncode=1
        )
        assert install_prerequisite("helm") is False

    @patch("cogniverse_cli.cluster.platform.system", return_value="Linux")
    def test_unknown_tool_returns_false_without_running(self, mock_sys: object) -> None:
        with patch("cogniverse_cli.cluster.subprocess.run") as mock_run:
            assert install_prerequisite("mystery-tool") is False
            mock_run.assert_not_called()

    @patch(
        "cogniverse_cli.cluster.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="bash", timeout=300),
    )
    @patch("cogniverse_cli.cluster.platform.system", return_value="Linux")
    def test_install_timeout_reports_failure(
        self, mock_sys: object, mock_run: object
    ) -> None:
        assert install_prerequisite("k3d") is False


class TestInstallMissingPrerequisites:
    """`install_missing_prerequisites` never auto-installs docker, and only
    drops a tool from the missing list once it is both installed AND on PATH."""

    def test_docker_is_never_auto_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster, "install_prerequisite", lambda tool: True)
        monkeypatch.setattr(cluster.shutil, "which", lambda tool: "/usr/bin/" + tool)
        assert install_missing_prerequisites(["docker", "k3d"]) == ["docker"]

    def test_failed_install_stays_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster, "install_prerequisite", lambda tool: False)
        monkeypatch.setattr(cluster.shutil, "which", lambda tool: None)
        assert install_missing_prerequisites(["k3d", "helm"]) == ["k3d", "helm"]

    def test_installed_but_not_on_path_stays_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster, "install_prerequisite", lambda tool: True)
        monkeypatch.setattr(cluster.shutil, "which", lambda tool: None)
        assert install_missing_prerequisites(["k3d"]) == ["k3d"]

    def test_successful_install_drops_from_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster, "install_prerequisite", lambda tool: True)
        monkeypatch.setattr(cluster.shutil, "which", lambda tool: "/usr/bin/" + tool)
        assert install_missing_prerequisites(["k3d", "helm"]) == []
