"""Dead-owner container reaping for test-spawned Docker sidecars.

Session fixtures tear sidecars down in a ``finally``, but SIGKILL on the
pytest process skips it — orphaned vLLM/Vespa/router containers then hold
model weights and JVM heap in host RAM indefinitely (a day of orphans once
starved the whole host into a freeze). ``reap_dead_owner_containers`` runs
at every spawn and removes containers whose labelled owner pid is gone,
while never touching a live session's containers.
"""

import subprocess

import pytest

from tests.utils.vllm_sidecar import OWNER_LABEL, reap_dead_owner_containers

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
