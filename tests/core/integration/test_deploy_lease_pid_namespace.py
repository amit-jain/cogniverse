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

# Forks inside the container until a child is given ``target`` as its pid,
# then mints a holder there and stays alive holding it.
_MINT = """
import os, site, sys, time
site.addsitedir(sys.argv[2])
from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
target = int(sys.argv[1])
while True:
    pid = os.fork()
    if pid == 0:
        if os.getpid() == target:
            print("HOLDER " + SchemaDeployLease(None).holder, flush=True)
            time.sleep(120)
        os._exit(0)
    os.waitpid(pid, 0)
    if pid >= target:
        break
"""


def _free_host_pid() -> int:
    running = {int(name) for name in os.listdir("/proc") if name.isdigit()}
    return next(pid for pid in range(3, 400) if pid not in running)


def _record_holder(store):
    entry = store.get_config(
        tenant_id="__system__",
        scope=ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )
    return entry.config_value["holder"]


def test_a_live_holder_in_another_pid_namespace_is_not_taken_over(tmp_path):
    target = _free_host_pid()
    script = tmp_path / "mint.py"
    script.write_text(_MINT)
    name = f"lease-pidns-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
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
            "python:3.12-slim",
            "python",
            "/mint.py",
            str(target),
            str(SITE_PACKAGES),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    try:
        holder = None
        deadline = time.monotonic() + 60
        while holder is None and time.monotonic() < deadline:
            logs = subprocess.run(
                ["docker", "logs", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            holder = next(
                (
                    line.split(" ", 1)[1]
                    for line in logs.stdout.splitlines()
                    if line.startswith("HOLDER ")
                ),
                None,
            )
            if holder is None:
                time.sleep(0.5)
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
