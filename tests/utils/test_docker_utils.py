"""Tests for the Vespa test-port allocator and container starters.

``generate_unique_ports`` must hand back a pair that is actually bindable so a
leftover container from a crashed prior run (or a concurrent session) can't
cause an ``address already in use`` failure when the test starts its Vespa
container — the CI flake this allocator was hardened to prevent.
"""

import json
import re
import socket
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.utils import docker_utils, vespa_docker
from tests.utils.docker_utils import _port_is_free, generate_unique_ports


def test_returns_free_pair_with_standard_offset():
    http_port, config_port = generate_unique_ports("tests.unit.docker_utils")
    # config is always http + 10991 (callers re-derive it from http).
    assert config_port == http_port + 10991
    # range keeps config_port < 65535.
    assert 40000 <= http_port <= 54544
    assert config_port < 65535
    # the returned ports are genuinely bindable right now.
    assert _port_is_free(http_port)
    assert _port_is_free(config_port)


def test_port_is_free_detects_a_bound_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        bound = s.getsockname()[1]
        assert _port_is_free(bound) is False
    # once the socket is closed the port frees up again.
    assert _port_is_free(bound) is True


def test_skips_a_busy_candidate(monkeypatch):
    """A candidate whose http or config port is in use is rejected; the next
    free candidate is returned."""
    import random

    candidates = iter([45000, 46000])
    monkeypatch.setattr(random, "randint", lambda a, b: next(candidates))

    def fake_free(port: int) -> bool:
        return port not in (45000, 45000 + 10991)

    monkeypatch.setattr(docker_utils, "_port_is_free", fake_free)

    http_port, config_port = generate_unique_ports("m")
    assert (http_port, config_port) == (46000, 46000 + 10991)


def test_falls_back_to_deterministic_hash_when_nothing_is_free(monkeypatch):
    """If probing never finds a free pair, fall back to the module+PID hash so
    behaviour degrades to the old deterministic allocation rather than hanging."""
    monkeypatch.setattr(docker_utils, "_port_is_free", lambda port: False)

    http_port, config_port = generate_unique_ports("tests.fallback")
    assert config_port == http_port + 10991
    assert 40000 <= http_port < 54544
    # deterministic for a fixed module+PID.
    assert generate_unique_ports("tests.fallback") == (http_port, config_port)


def _remove_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _assert_exact_bindings(name: str, http_port: int, config_port: int) -> None:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .HostConfig.PortBindings}}",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "8080/tcp": [{"HostIp": "", "HostPort": str(http_port)}],
        "19071/tcp": [{"HostIp": "", "HostPort": str(config_port)}],
    }


@pytest.mark.integration
@pytest.mark.requires_docker
def test_container_start_retries_a_real_port_bind_collision(monkeypatch):
    blocker_http, blocker_config = generate_unique_ports("tests.docker.blocker")
    retry_http, retry_config = generate_unique_ports("tests.docker.retry")
    run_id = uuid.uuid4().hex[:10]
    blocker_name = f"port-blocker-{run_id}"
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            blocker_name,
            "-p",
            f"{blocker_http}:8080",
            "-p",
            f"{blocker_config}:19071",
            "busybox:latest",
            "sleep",
            "60",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    candidates = iter([(blocker_http, blocker_config), (retry_http, retry_config)])
    monkeypatch.setattr(
        docker_utils,
        "generate_unique_ports",
        lambda _module_name: next(candidates),
    )

    started_name = None
    try:
        started_name, http_port, config_port = (
            docker_utils.start_docker_container_with_port_retry(
                "tests.docker.retry",
                name_prefix=f"port-retry-{run_id}",
                image="busybox:latest",
                container_ports=(8080, 19071),
                container_command=["sleep", "60"],
                max_attempts=2,
            )
        )

        assert (http_port, config_port) == (retry_http, retry_config)
        _assert_exact_bindings(started_name, retry_http, retry_config)
    finally:
        if started_name is not None:
            _remove_container(started_name)
        _remove_container(blocker_name)


@pytest.mark.integration
@pytest.mark.requires_docker
def test_concurrent_container_starts_recover_from_same_candidate(monkeypatch):
    shared_pair = generate_unique_ports("tests.docker.concurrent.shared")
    original_generate = docker_utils.generate_unique_ports
    first_calls = 0
    first_calls_lock = threading.Lock()
    first_candidate_barrier = threading.Barrier(4, timeout=15)
    run_id = uuid.uuid4().hex[:10]

    def colliding_generate(module_name):
        nonlocal first_calls
        with first_calls_lock:
            call_number = first_calls
            first_calls += 1
        if call_number < 4:
            first_candidate_barrier.wait()
            return shared_pair
        return original_generate(module_name)

    monkeypatch.setattr(
        docker_utils,
        "generate_unique_ports",
        colliding_generate,
    )

    def start_container(worker_id):
        return docker_utils.start_docker_container_with_port_retry(
            f"tests.docker.concurrent.{worker_id}",
            name_prefix=f"port-concurrent-{run_id}",
            image="busybox:latest",
            container_ports=(8080, 19071),
            container_command=["sleep", "60"],
            max_attempts=6,
        )

    started = []
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            started = list(executor.map(start_container, range(4)))

        assert len({(http, config) for _, http, config in started}) == 4
        for name, http_port, config_port in started:
            _assert_exact_bindings(name, http_port, config_port)
    finally:
        for name, _, _ in started:
            _remove_container(name)


@pytest.mark.integration
@pytest.mark.requires_docker
def test_container_start_surfaces_non_allocation_failure():
    run_id = uuid.uuid4().hex[:10]

    with pytest.raises(RuntimeError, match=r"attempt 1/3.*unknown flag"):
        docker_utils.start_docker_container_with_port_retry(
            "tests.docker.failure",
            name_prefix=f"port-failure-{run_id}",
            image="busybox:latest",
            container_ports=(8080, 19071),
            extra_run_args=["--definitely-invalid-option"],
            container_command=["sleep", "60"],
            max_attempts=3,
        )


@pytest.mark.integration
@pytest.mark.requires_docker
def test_container_start_removes_the_container_its_failed_start_created():
    """Docker creates the container before a start-time failure (here a
    missing device); the starter removes it before raising, so the failed
    start leaves nothing behind."""
    run_id = uuid.uuid4().hex[:10]
    name_prefix = f"port-created-{run_id}"
    missing_device = f"/dev/cogniverse-missing-{run_id}"

    with pytest.raises(RuntimeError) as excinfo:
        docker_utils.start_docker_container_with_port_retry(
            "tests.docker.created",
            name_prefix=name_prefix,
            image="busybox:latest",
            container_ports=(8080, 19071),
            extra_run_args=["--device", missing_device],
            container_command=["sleep", "60"],
            max_attempts=3,
        )

    assert re.fullmatch(
        rf"Docker container {name_prefix}-\d+-\d+-\d+ failed on attempt 1/3: "
        r"docker: Error response from daemon: error gathering device "
        rf'information while adding custom device "{missing_device}": no such '
        r"file or directory Run 'docker run --help' for more information",
        str(excinfo.value),
    )
    leftovers = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={name_prefix}",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert leftovers.stdout.split() == []


def _running_containers(field: str = "{{.Names}}") -> set[str]:
    result = subprocess.run(
        ["docker", "ps", "--no-trunc", "--format", field],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return set(result.stdout.split())


def _allocator_returning(monkeypatch, *pairs):
    """Make the port allocator hand out ``pairs`` in order, wherever the
    container starter looks it up."""
    candidates = iter(pairs)
    monkeypatch.setattr(
        docker_utils, "generate_unique_ports", lambda _module_name: next(candidates)
    )
    monkeypatch.setattr(
        vespa_docker,
        "generate_unique_ports",
        lambda _module_name: next(candidates),
        raising=False,
    )


@pytest.mark.integration
@pytest.mark.requires_docker
def test_vespa_container_moves_off_a_port_another_socket_holds(monkeypatch):
    """A port taken between allocation and ``docker run`` (here: a listening
    socket) is retried on a fresh pair rather than failing the fixture."""
    held_http, held_config = generate_unique_ports("tests.vespa.held")
    fresh_http, fresh_config = generate_unique_ports("tests.vespa.fresh")
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("0.0.0.0", held_http))
    holder.listen()
    _allocator_returning(
        monkeypatch, (held_http, held_config), (fresh_http, fresh_config)
    )
    manager = vespa_docker.VespaDockerManager()
    info = None
    try:
        info = manager.start_container("tests.vespa.held")

        assert (info["http_port"], info["config_port"]) == (fresh_http, fresh_config)
        assert info["base_url"] == f"http://localhost:{fresh_http}"
        assert info["container_name"] in _running_containers()
        _assert_exact_bindings(info["container_name"], fresh_http, fresh_config)
        assert holder.getsockname()[1] == held_http
    finally:
        holder.close()
        if info is not None:
            _remove_container(info["container_name"])


@pytest.mark.integration
@pytest.mark.requires_docker
def test_vespa_container_start_leaves_another_live_container_running(monkeypatch):
    """Another session's test container holding the first candidate keeps
    running, and the new container takes a fresh pair."""
    taken_http, taken_config = generate_unique_ports("tests.vespa.taken")
    fresh_http, fresh_config = generate_unique_ports("tests.vespa.fresh")
    other = f"vespa-test-{taken_http}"
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            other,
            "-p",
            f"{taken_http}:8080",
            "-p",
            f"{taken_config}:19071",
            "busybox:latest",
            "sleep",
            "120",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    other_id = result.stdout.strip()
    _allocator_returning(
        monkeypatch, (taken_http, taken_config), (fresh_http, fresh_config)
    )
    manager = vespa_docker.VespaDockerManager()
    info = None
    try:
        info = manager.start_container("tests.vespa.taken")

        assert other_id in _running_containers("{{.ID}}")
        _assert_exact_bindings(other_id, taken_http, taken_config)
        assert (info["http_port"], info["config_port"]) == (fresh_http, fresh_config)
        assert info["container_name"] != other
        _assert_exact_bindings(info["container_name"], fresh_http, fresh_config)
    finally:
        if info is not None:
            _remove_container(info["container_name"])
        _remove_container(other)


@pytest.mark.integration
@pytest.mark.requires_docker
def test_concurrent_vespa_containers_from_one_candidate_get_distinct_ports(
    monkeypatch,
):
    """Two managers released together onto the same candidate pair both
    start, on distinct ports, and neither removes the other's container."""
    shared_pair = generate_unique_ports("tests.vespa.concurrent.shared")
    original_generate = docker_utils.generate_unique_ports
    calls = 0
    calls_lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=15)

    def colliding_generate(module_name):
        nonlocal calls
        with calls_lock:
            call_number = calls
            calls += 1
        if call_number < 2:
            barrier.wait()
            return shared_pair
        return original_generate(module_name)

    monkeypatch.setattr(docker_utils, "generate_unique_ports", colliding_generate)
    monkeypatch.setattr(
        vespa_docker, "generate_unique_ports", colliding_generate, raising=False
    )

    def start(worker_id):
        try:
            return vespa_docker.VespaDockerManager().start_container(
                f"tests.vespa.concurrent.{worker_id}"
            )
        except RuntimeError as exc:
            return exc

    started = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            started = list(executor.map(start, range(2)))

        assert [type(info) for info in started] == [dict, dict]
        assert len({(i["http_port"], i["config_port"]) for i in started}) == 2
        assert shared_pair in {(i["http_port"], i["config_port"]) for i in started}
        running = _running_containers()
        for info in started:
            assert info["container_name"] in running
            _assert_exact_bindings(
                info["container_name"], info["http_port"], info["config_port"]
            )
    finally:
        for info in started:
            if isinstance(info, dict):
                _remove_container(info["container_name"])


@pytest.mark.integration
@pytest.mark.requires_docker
def test_vespa_container_start_raises_when_every_candidate_is_taken(monkeypatch):
    """Exhausted retries raise with Docker's bind error, leaving no container."""
    held_http, held_config = generate_unique_ports("tests.vespa.exhausted")
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("0.0.0.0", held_http))
    holder.listen()
    _allocator_returning(monkeypatch, *[(held_http, held_config)] * 5)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            vespa_docker.VespaDockerManager().start_container("tests.vespa.exhausted")
    finally:
        holder.close()

    message = str(excinfo.value)
    assert message.startswith(
        "Failed to start Vespa container: "
        "Docker container allocation failed after 5 attempts: "
    )
    assert f"0.0.0.0:{held_http}" in message
    assert "address already in use" in message
    leftovers = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=vespa-test-",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert [
        name for name in leftovers.stdout.split() if name.endswith(f"-{held_http}")
    ] == []
