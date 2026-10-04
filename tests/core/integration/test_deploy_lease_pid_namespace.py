"""A lease holder in another PID namespace is never judged by a local pid probe.

Two processes can share a hostname without sharing a pid space — a container
run with the host's UTS namespace is one. A pid probe from one says nothing
about the other, so a live holder there must not be taken over as "gone".
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from cogniverse_core.registries import schema_deploy_lease
from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
SITE_PACKAGES = REPO / ".venv" / "lib" / "python3.12" / "site-packages"

# Imports first, then reports the next pid its namespace would hand out and
# waits for the test to name a target at or past it. It then forks until a
# child is given that pid, mints a holder there and stays alive holding it.
# Python's start-up and the venv's .pth files spawn processes of their own, so
# the first pid free to a fork here is only known once the container has
# started.
_MINT = """
import os, site, sys, time
site.addsitedir(sys.argv[1])
from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
probe = os.fork()
if probe == 0:
    os._exit(0)
os.waitpid(probe, 0)
print("NEXT " + str(probe + 1), flush=True)
while not os.path.exists("/shared/target"):
    time.sleep(0.1)
target = int(open("/shared/target").read())
while True:
    pid = os.fork()
    if pid == 0:
        if os.getpid() == target:
            print("HOLDER " + SchemaDeployLease(None).holder, flush=True)
            time.sleep(120)
        os._exit(0)
    os.waitpid(pid, 0)
    if pid >= target:
        print("MISSED " + str(pid), flush=True)
        break
"""

# How far past the container's next pid the target sits: room for anything
# else the container's start-up still spawns.
_TARGET_MARGIN = 10


def _free_host_pid(at_least: int) -> int:
    """The lowest pid from ``at_least`` that no host process holds."""
    running = {int(name) for name in os.listdir("/proc") if name.isdigit()}
    return next(pid for pid in range(at_least, 1 << 22) if pid not in running)


def _logged(name: str, prefix: str, deadline: float):
    """The value after ``prefix`` on the container's log line that starts
    with it, and the logs read; None for the value at the deadline or when
    the container stopped."""
    while True:
        logs = subprocess.run(
            ["docker", "logs", name], capture_output=True, text=True, timeout=30
        )
        value = next(
            (
                line.split(" ", 1)[1]
                for line in logs.stdout.splitlines()
                if line.startswith(prefix)
            ),
            None,
        )
        running = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if value is not None or running != "true" or time.monotonic() >= deadline:
            return value, logs
        time.sleep(0.5)


def _record_holder(store):
    entry = store.get_config(
        tenant_id="__system__",
        scope=ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )
    return entry.config_value["holder"]


def test_a_live_holder_in_another_pid_namespace_is_not_taken_over(tmp_path):
    script = tmp_path / "mint.py"
    script.write_text(_MINT)
    shared = tmp_path / "shared"
    shared.mkdir()
    name = f"lease-pidns-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    # No --rm: a container that stops early keeps its logs for the assertion.
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "--uts=host",
            "-u",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{REPO}:{REPO}:ro",
            "-v",
            f"{script}:/mint.py:ro",
            "-v",
            f"{shared}:/shared:ro",
            "python:3.12-slim",
            "python",
            "/mint.py",
            str(SITE_PACKAGES),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    try:
        deadline = time.monotonic() + 120
        next_pid, logs = _logged(name, "NEXT ", deadline)
        assert (next_pid or "").isdigit(), logs.stdout + logs.stderr
        target = _free_host_pid(int(next_pid) + _TARGET_MARGIN)
        (shared / "target").write_text(str(target))
        holder, logs = _logged(name, "HOLDER ", deadline)
        assert holder is not None, logs.stdout + logs.stderr
        assert holder.split(":", 1)[0] == socket.gethostname()
        assert f":{target}:" in holder
        with pytest.raises(ProcessLookupError):
            os.kill(target, 0)

        store = InMemoryConfigStore()
        planted = SchemaDeployLease(store, wait_seconds=0)
        planted.holder = holder
        planted.acquire()

        assert schema_deploy_lease._holder_is_gone(holder) is False
        with pytest.raises(TimeoutError):
            SchemaDeployLease(store, wait_seconds=0).acquire()
        assert _record_holder(store) == holder
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
