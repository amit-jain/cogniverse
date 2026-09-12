"""Dead-owner container reaping for test-spawned Docker sidecars.

Session fixtures tear sidecars down in a ``finally``, but SIGKILL on the
pytest process skips it — orphaned vLLM/Vespa/router containers then hold
model weights and JVM heap in host RAM indefinitely (a day of orphans once
starved the whole host into a freeze). ``reap_dead_owner_containers`` runs
when every pytest session starts and at every spawn, and removes containers
whose labelled owner pid is gone, while never touching a live session's
containers.
"""

import os
import re
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.fixtures.sidecars import reap_at_session_start
from tests.utils.vllm_sidecar import (
    OWNER_LABEL,
    reap_dead_owner_containers,
    reap_dead_owner_networks,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.integration


def _docker_up() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _require_docker():
    if not _docker_up():
        pytest.fail("docker daemon required — reaper tests exercise real docker")


def _run_probe(name: str, owner_pid: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"{OWNER_LABEL}={owner_pid}",
            "busybox",
            "sleep",
            "300",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def _exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", name], capture_output=True, timeout=30
    )
    return result.returncode == 0


def test_reaps_dead_owner_and_keeps_live_owner():
    import os

    dead = "cogniverse-reaper-test-dead"
    live = "cogniverse-reaper-test-live"
    try:
        _run_probe(dead, "999999999")  # no such pid
        _run_probe(live, str(os.getpid()))  # this very test process

        reap_dead_owner_containers()

        assert not _exists(dead), "dead-owner container must be removed"
        assert _exists(live), "live-owner container must be left alone"
    finally:
        subprocess.run(["docker", "rm", "-f", dead], capture_output=True, timeout=30)
        subprocess.run(["docker", "rm", "-f", live], capture_output=True, timeout=30)


def test_reaps_exited_containers_even_with_live_owner():
    import os

    name = "cogniverse-reaper-test-exited"
    try:
        _run_probe(name, str(os.getpid()))
        subprocess.run(
            ["docker", "stop", "-t", "0", name], capture_output=True, timeout=60
        )

        reap_dead_owner_containers()

        assert not _exists(name), "exited container must be removed regardless of owner"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def test_keeps_created_container_while_live_owner_is_starting_it():
    import os

    name = "cogniverse-reaper-test-created"
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        created = subprocess.run(
            [
                "docker",
                "create",
                "--name",
                name,
                "--label",
                f"{OWNER_LABEL}={os.getpid()}",
                "busybox",
                "sleep",
                "300",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert created.returncode == 0, created.stderr

        reap_dead_owner_containers()

        assert _exists(name), (
            "a live session's created container must survive until docker start"
        )
        started = subprocess.run(
            ["docker", "start", name], capture_output=True, text=True, timeout=60
        )
        assert started.returncode == 0, started.stderr
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def test_phoenix_container_carries_owner_label(phoenix_container):
    """The phoenix_container docker-run must carry the owner-pid label so a
    SIGKILLed session's Phoenix container is reaped instead of orphaned."""
    import os

    name = phoenix_container["container_name"]
    label = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            '{{ index .Config.Labels "' + OWNER_LABEL + '" }}',
            name,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert label == str(os.getpid())

    cid = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", name],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    listed = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={OWNER_LABEL}={os.getpid()}",
            "-q",
            "--no-trunc",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert cid in listed, "container not discoverable by the reaper label filter"


def _dead_owner_pid() -> int:
    exited = subprocess.Popen(["true"])
    exited.wait(timeout=30)
    if os.path.exists(f"/proc/{exited.pid}"):
        pytest.fail(f"pid {exited.pid} of an exited process is still in use")
    return exited.pid


def _start_owned(name: str, owner_pid: int) -> str:
    """Start a labelled container and return the id ``docker ps`` lists it by."""
    _run_probe(name, str(owner_pid))
    return subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", name],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()[:12]


def _state(name: str) -> str:
    return subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", name],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


def _sidecar_section(output: str) -> list[str]:
    """The lines of the nested session's ``test sidecars`` summary section."""
    lines = output.splitlines()
    starts = [
        i for i, line in enumerate(lines) if re.fullmatch(r"=+ test sidecars =+", line)
    ]
    if len(starts) != 1:
        pytest.fail(
            f"expected one 'test sidecars' section, found {len(starts)}:\n{output}"
        )
    section = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("="):
            break
        section.append(line)
    return section


@pytest.mark.parametrize(
    ("selection", "rootdir"),
    [
        (
            "tests/common/unit/test_shared_vespa_config.py"
            "::test_shared_vespa_container_uses_bounded_session_storage",
            REPO_ROOT,
        ),
        (
            "tests/ingestion/unit/test_queue_int_env.py::test_defaults_when_unset",
            REPO_ROOT / "tests" / "ingestion",
        ),
    ],
    ids=["repo-rootdir", "ingestion-rootdir"],
)
def test_a_session_that_provisions_nothing_still_reaps_at_start(selection, rootdir):
    """One trivial test file, nothing provisioned, and the dead owner's
    container is gone when the session ends; a live owner's is untouched."""
    suffix = uuid.uuid4().hex[:8]
    dead = f"cogniverse-reaper-test-session-dead-{suffix}"
    live = f"cogniverse-reaper-test-session-live-{suffix}"
    try:
        dead_id = _start_owned(dead, _dead_owner_pid())
        live_id = _start_owned(live, os.getpid())

        result = subprocess.run(
            [sys.executable, "-m", "pytest", selection, "-p", "no:cacheprovider"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr

        assert result.returncode == 0, output
        assert f"rootdir: {rootdir}" in output.splitlines()
        assert not _exists(dead), "the dead owner's container must be reaped"
        assert _state(live) == "running", "a live owner's container must survive"
        reap_lines = [
            line
            for line in _sidecar_section(output)
            if line.startswith("reaped dead-owner containers: ")
        ]
        assert len(reap_lines) == 1, output
        reaped = (
            reap_lines[0].removeprefix("reaped dead-owner containers: ").split(", ")
        )
        # Other sessions' dead owners on this host are reaped by the same pass,
        # so the line is checked for this test's two containers, not for size.
        assert (dead_id in reaped, live_id in reaped) == (True, False)
    finally:
        subprocess.run(["docker", "rm", "-f", dead], capture_output=True, timeout=30)
        subprocess.run(["docker", "rm", "-f", live], capture_output=True, timeout=30)


def test_session_reap_is_a_no_op_without_a_docker_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert reap_at_session_start() is None


def test_unreachable_docker_daemon_raises_with_context_and_is_reported(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'no-daemon.sock'}")
    listing = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"label={OWNER_LABEL}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    detail = "\n".join(part for part in (listing.stdout, listing.stderr) if part)
    assert listing.returncode == 1
    assert str(tmp_path / "no-daemon.sock") in detail
    expected = (
        f"docker could not list containers labelled {OWNER_LABEL}: {detail.strip()}"
    )

    with pytest.raises(RuntimeError) as excinfo:
        reap_dead_owner_containers()

    assert str(excinfo.value) == expected
    assert reap_at_session_start() == f"dead-owner container reap failed: {expected}"


def test_concurrent_reapers_remove_a_dead_owner_container_exactly_once():
    name = f"cogniverse-reaper-test-race-{uuid.uuid4().hex[:8]}"
    workers = 4
    try:
        container_id = _start_owned(name, _dead_owner_pid())
        start = threading.Barrier(workers, timeout=30)

        def reap(_):
            start.wait()
            return reap_dead_owner_containers()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(reap, range(workers)))

        assert sorted(result.count(container_id) for result in results) == [0] * (
            workers - 1
        ) + [1]
        assert not _exists(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def _network_exists(name: str) -> bool:
    listing = subprocess.run(
        ["docker", "network", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return name in listing.stdout.split()


def _create_owned_network(name: str, owner_pid: int) -> None:
    subprocess.run(
        [
            "docker",
            "network",
            "create",
            "--label",
            f"{OWNER_LABEL}={owner_pid}",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


class TestDeadOwnerNetworksAreReclaimed:
    """A stack fixture creates a network per run. Containers carry the owner
    label and are reaped; a network that outlives its teardown is invisible to
    the container reaper and accumulates on the host for good."""

    def test_removes_a_network_whose_owner_pid_is_gone(self):
        name = f"cog-sr-net-reaptest-dead-{uuid.uuid4().hex[:8]}"
        _create_owned_network(name, _dead_owner_pid())
        try:
            removed = reap_dead_owner_networks()
            assert name in removed
            assert _network_exists(name) is False
        finally:
            subprocess.run(
                ["docker", "network", "rm", name], capture_output=True, timeout=30
            )

    def test_keeps_a_network_whose_owner_is_alive(self):
        name = f"cog-sr-net-reaptest-live-{uuid.uuid4().hex[:8]}"
        _create_owned_network(name, os.getpid())
        try:
            removed = reap_dead_owner_networks()
            assert name not in removed
            assert _network_exists(name) is True
        finally:
            subprocess.run(
                ["docker", "network", "rm", name], capture_output=True, timeout=30
            )
