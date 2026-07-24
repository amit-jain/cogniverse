"""Real-process test that a repeated start reaps the prior port-forward daemon.

No cluster or kubectl is required: ``start_port_forwards`` spawns its real
detached, self-restarting shell loop, and a loop retrying a failing command is
exactly the orphan behavior. We assert the first real process is actually
terminated after the second start (and the second after stop) by reaping its
``Popen`` and confirming its process group is gone.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

import pytest
from cogniverse_cli import cluster as cluster_mod
from cogniverse_cli.cluster import start_port_forwards, stop_port_forwards

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_group_gone(pgid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return
        time.sleep(0.05)
    raise AssertionError(f"process group {pgid} is still alive")


def test_start_reaps_real_prior_daemon(tmp_path, monkeypatch) -> None:
    pid_file = tmp_path / "port-forwards.pids"
    monkeypatch.setattr(cluster_mod, "PID_FILE", str(pid_file))
    # Shorten the SIGTERM grace so the (real) SIGKILL escalation runs quickly.
    monkeypatch.setattr(cluster_mod, "_REAP_GRACE_SECONDS", 0.5)
    cluster_mod._port_forward_procs.clear()

    spawned = []
    try:
        start_port_forwards()
        first = cluster_mod._port_forward_procs[0]
        spawned.append(first)
        first_pgid = os.getpgid(first.pid)
        first_pids = [int(x) for x in pid_file.read_text().split() if x]
        assert first_pids == [first.pid]
        assert _group_alive(first_pgid)

        # Second start must terminate the first real daemon before rebinding.
        start_port_forwards()
        second = cluster_mod._port_forward_procs[0]
        spawned.append(second)
        second_pgid = os.getpgid(second.pid)
        second_pids = [int(x) for x in pid_file.read_text().split() if x]
        assert second.pid != first.pid

        # The first daemon actually exited (killed) — wait reaps it; a still-
        # running daemon would time out here.
        rc = first.wait(timeout=5)
        assert rc is not None
        _wait_group_gone(first_pgid)
        # Only the live daemon's PID remains on disk, and its group is alive.
        assert second_pids == [second.pid]
        assert _group_alive(second_pgid)

        # A fresh teardown process has no in-memory handles — it reaps via the
        # PID file. Simulate that, then assert the second daemon is gone too.
        cluster_mod._port_forward_procs.clear()
        stop_port_forwards()
        rc2 = second.wait(timeout=5)
        assert rc2 is not None
        _wait_group_gone(second_pgid)
        assert not pid_file.exists()
    finally:
        for proc in spawned:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ValueError):
                pass
        cluster_mod._port_forward_procs.clear()
