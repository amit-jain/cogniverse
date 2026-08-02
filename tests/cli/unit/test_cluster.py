"""Unit tests for cogniverse_cli.cluster lifecycle utilities."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
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
    start_port_forwards,
    stop_port_forwards,
)


def _install_fake_cluster_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    list_json: str = "[]",
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    k3d = bin_dir / "k3d"
    k3d.write_text(
        """#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
if args[:2] == ["cluster", "list"] and "-o" in args:
    print(os.environ.get("FAKE_K3D_LIST_JSON", "[]"))
raise SystemExit(0)
"""
    )
    k3d.chmod(0o755)

    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        """#!/usr/bin/env python3
import sys

if "configmap" in sys.argv:
    raise SystemExit(1)
raise SystemExit(0)
"""
    )
    kubectl.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_K3D_LIST_JSON", list_json)


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

        cmd = mock_run.call_args_list[0][0][0]  # first call: k3d create

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

        cmd = mock_run.call_args_list[0][0][0]  # first call: k3d create
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

        cmd = mock_run.call_args_list[0][0][0]  # first call: k3d create
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

        cmd = mock_run.call_args_list[0][0][0]  # first call: k3d create
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

        cmd = mock_run.call_args_list[0][0][0]  # first call: k3d create
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

        mock_run.side_effect = [  # type: ignore[attr-defined]
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    '[{"name": "cogniverse", "network": {"name": '
                    '"k3d-cogniverse"}, "nodes": []}]'
                ),
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    '[{"name": "cogniverse", "network": {"name": '
                    '"k3d-cogniverse"}, "nodes": []}]'
                ),
            ),
        ]
        with patch("cogniverse_cli.cluster.pin_coredns_upstreams"):
            start_cluster()

        args = mock_run.call_args_list[1]  # inspection runs before k3d start
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


class TestPinCorednsUpstreams:
    """k3d's CoreDNS forwards to the host's resolv.conf; a dead host resolver
    fails every pod's external DNS (vLLM crashloops, flapping NodePorts).
    The pin rewrites the forward to public upstreams, idempotently, on both
    cluster create and start."""

    K3D_COREFILE = (
        "apiVersion: v1\n"
        "data:\n"
        "  Corefile: |\n"
        "    .:53 {\n"
        "        errors\n"
        "        health\n"
        "        forward . /etc/resolv.conf\n"
        "        cache 30\n"
        "    }\n"
        "kind: ConfigMap\n"
    )

    def test_pinned_corefile_rewrites_host_forward(self) -> None:
        from cogniverse_cli.cluster import pinned_corefile

        patched = pinned_corefile(self.K3D_COREFILE)
        assert patched is not None
        assert "forward . 1.1.1.1 8.8.8.8" in patched
        assert "forward . /etc/resolv.conf" not in patched

    def test_pinned_corefile_none_when_already_pinned(self) -> None:
        from cogniverse_cli.cluster import pinned_corefile

        already = self.K3D_COREFILE.replace(
            "forward . /etc/resolv.conf", "forward . 1.1.1.1 8.8.8.8"
        )
        assert pinned_corefile(already) is None

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_pin_applies_patch_and_restarts_coredns(self, mock_run) -> None:
        from cogniverse_cli.cluster import pin_coredns_upstreams

        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout=self.K3D_COREFILE
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        ]

        assert pin_coredns_upstreams("cogniverse") is True

        calls = [c[0][0] for c in mock_run.call_args_list]
        assert calls[0][:3] == ["kubectl", "--context", "k3d-cogniverse"]
        assert "configmap" in calls[0]
        assert calls[1][-3:] == ["apply", "-f", "-"]
        applied = mock_run.call_args_list[1][1]["input"]
        assert "forward . 1.1.1.1 8.8.8.8" in applied
        assert calls[2][-2:] == ["restart", "deployment/coredns"]

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_pin_noop_when_already_pinned(self, mock_run) -> None:
        from cogniverse_cli.cluster import pin_coredns_upstreams

        already = self.K3D_COREFILE.replace(
            "forward . /etc/resolv.conf", "forward . 1.1.1.1 8.8.8.8"
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=already
        )

        assert pin_coredns_upstreams("cogniverse") is True
        assert len(mock_run.call_args_list) == 1  # get only — no apply/restart

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_pin_false_when_configmap_never_appears(self, mock_run) -> None:
        from cogniverse_cli.cluster import pin_coredns_upstreams

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="not found"
        )

        assert pin_coredns_upstreams("cogniverse", timeout_s=0.01) is False

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_create_cluster_pins_coredns(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=""
        )

        create_cluster("cogniverse")

        calls = [c[0][0] for c in mock_run.call_args_list]
        assert calls[0][:3] == ["k3d", "cluster", "create"]
        assert any("configmap" in c for c in calls[1:])

    @patch("cogniverse_cli.cluster.subprocess.run")
    def test_start_cluster_pins_coredns(self, mock_run) -> None:
        from cogniverse_cli.cluster import start_cluster

        def run(cmd, **kwargs):
            if cmd[:3] == ["k3d", "cluster", "list"]:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=(
                        '[{"name": "cogniverse", "network": {"name": '
                        '"k3d-cogniverse"}, "nodes": []}]'
                    ),
                )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="forward . 1.1.1.1 8.8.8.8",
            )

        mock_run.side_effect = run
        start_cluster("cogniverse")

        calls = [c[0][0] for c in mock_run.call_args_list]
        assert calls[0] == [
            "k3d",
            "cluster",
            "list",
            "cogniverse",
            "-o",
            "json",
        ]
        assert calls[1] == ["k3d", "cluster", "start", "cogniverse"]
        assert calls[2] == calls[0]
        assert any("configmap" in c for c in calls[3:])

    def test_create_cluster_raises_when_coredns_pinning_times_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cogniverse_cli.cluster import ClusterStartError, create_cluster

        _install_fake_cluster_tools(tmp_path, monkeypatch)
        real_pin = cluster.pin_coredns_upstreams
        monkeypatch.setattr(
            cluster,
            "pin_coredns_upstreams",
            lambda name: real_pin(name, timeout_s=0.0),
        )

        with pytest.raises(ClusterStartError, match="Could not pin CoreDNS upstreams"):
            create_cluster(
                "cogniverse",
                ports=[],
                share_hf_cache=False,
                share_host_storage=False,
            )

    def test_start_cluster_raises_when_coredns_pinning_times_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cogniverse_cli.cluster import ClusterStartError, start_cluster

        _install_fake_cluster_tools(
            tmp_path,
            monkeypatch,
            list_json=(
                '[{"name": "cogniverse", "network": {"name": "k3d-cogniverse"},'
                ' "nodes": []}]'
            ),
        )
        real_pin = cluster.pin_coredns_upstreams
        monkeypatch.setattr(
            cluster,
            "pin_coredns_upstreams",
            lambda name: real_pin(name, timeout_s=0.0),
        )

        with pytest.raises(ClusterStartError, match="Could not pin CoreDNS upstreams"):
            start_cluster("cogniverse")


class _FakeProc:
    """Stand-in for a spawned port-forward daemon."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.args = ["sh", "-c", "while true; do kubectl port-forward ...; done"]

    def poll(self) -> None:
        return None


@pytest.fixture
def _isolated_pid_file(tmp_path, monkeypatch):
    """Point PID_FILE at a temp file and clear the in-process daemon list."""
    pid_file = tmp_path / "port-forwards.pids"
    monkeypatch.setattr(cluster, "PID_FILE", str(pid_file))
    cluster._port_forward_procs.clear()
    yield pid_file
    cluster._port_forward_procs.clear()


class _FakeGroups:
    """Simulate process groups: SIGTERM/SIGKILL end a group, signal 0 probes it.

    Groups in ``survive_sigterm`` ignore SIGTERM (like the real dash restart
    loop), so reaping must escalate to SIGKILL to end them.
    """

    def __init__(self, alive, survive_sigterm=()) -> None:
        self.alive = set(alive)
        self.survive_sigterm = set(survive_sigterm)
        self.term: list[int] = []
        self.kill: list[int] = []
        self.on_term = None

    def getpgid(self, pid: int) -> int:
        if pid not in self.alive:
            raise ProcessLookupError
        return pid

    def killpg(self, pgid: int, sig: int) -> None:
        if sig == 0:
            if pgid not in self.alive:
                raise ProcessLookupError
            return
        if sig == signal.SIGTERM:
            self.term.append(pgid)
            if self.on_term is not None:
                self.on_term(pgid)
            if pgid not in self.survive_sigterm:
                self.alive.discard(pgid)
        elif sig == signal.SIGKILL:
            self.kill.append(pgid)
            self.alive.discard(pgid)

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(cluster.os, "getpgid", self.getpgid)
        monkeypatch.setattr(cluster.os, "killpg", self.killpg)


class TestPortForwardReaping:
    """Tests for orphan-free (re)start and stop of port-forward daemons."""

    def test_start_reaps_prior_daemon_before_writing_new_pid(
        self, _isolated_pid_file, monkeypatch
    ) -> None:
        """A prior recorded daemon's process group is signalled before the new
        PID lands on disk, so a second start never orphans the first."""
        pid_file = _isolated_pid_file
        pid_file.write_text("4242")

        groups = _FakeGroups(alive=[4242])
        disk_at_term: list[str] = []
        groups.on_term = lambda pgid: disk_at_term.append(
            pid_file.read_text() if pid_file.exists() else ""
        )
        groups.install(monkeypatch)
        monkeypatch.setattr(
            cluster.subprocess, "Popen", lambda *a, **k: _FakeProc(9999)
        )

        start_port_forwards()

        assert groups.term == [4242]
        assert groups.kill == []  # exited on SIGTERM, no escalation needed
        # The prior PID was still the only thing on disk at reap time.
        assert disk_at_term == ["4242"]
        # The new daemon's PID overwrote it only after the reap.
        assert pid_file.read_text().strip() == "9999"

    def test_reap_escalates_to_sigkill_when_sigterm_ignored(
        self, _isolated_pid_file, monkeypatch
    ) -> None:
        """A daemon that ignores SIGTERM (the real shell loop) is SIGKILLed."""
        pid_file = _isolated_pid_file
        pid_file.write_text("777")
        monkeypatch.setattr(cluster, "_REAP_GRACE_SECONDS", 0.1)

        groups = _FakeGroups(alive=[777], survive_sigterm=[777])
        groups.install(monkeypatch)
        monkeypatch.setattr(
            cluster.subprocess, "Popen", lambda *a, **k: _FakeProc(9999)
        )

        start_port_forwards()

        assert groups.term == [777]
        assert groups.kill == [777]
        assert pid_file.read_text().strip() == "9999"

    def test_start_survives_dead_prior_daemon(
        self, _isolated_pid_file, monkeypatch
    ) -> None:
        """A prior daemon that already exited does not abort the restart."""
        pid_file = _isolated_pid_file
        pid_file.write_text("4242")

        groups = _FakeGroups(alive=[])  # prior daemon already gone
        groups.install(monkeypatch)
        monkeypatch.setattr(
            cluster.subprocess, "Popen", lambda *a, **k: _FakeProc(9999)
        )

        start_port_forwards()

        assert groups.term == []
        assert groups.kill == []
        assert pid_file.read_text().strip() == "9999"

    def test_stop_reaps_recorded_pids_cross_process(
        self, _isolated_pid_file, monkeypatch
    ) -> None:
        """stop reaps daemons recorded in PID_FILE with an empty in-process
        list (the fresh-process case) and removes the file."""
        pid_file = _isolated_pid_file
        pid_file.write_text("111\n222")

        groups = _FakeGroups(alive=[111, 222])
        groups.install(monkeypatch)

        stop_port_forwards()

        assert sorted(groups.term) == [111, 222]
        assert groups.kill == []
        assert not pid_file.exists()
