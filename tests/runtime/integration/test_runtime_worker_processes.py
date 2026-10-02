"""The runtime CLI serving from several uvicorn workers.

Each test boots the image's own command (``libs/runtime/Dockerfile`` CMD) as a
real process tree against the shared test Vespa and a test-owned Redis, with
the full lifespan in every worker. Which worker owns a listening socket or a
client connection is read from ``/proc``, never from the runtime's own
answer.

A server-managed conversation needs no LM here: its tenant never deployed the
schema of its one profile, so the summarizer answers with its fixed reply, and
the turns persist in real Mem0 through the workers' own conversation stores.
"""

from __future__ import annotations

import contextlib
import errno
import http.client
import inspect
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from cogniverse_core.conversation import CONVERSATION_AGENT_NAME
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_HISTORY_LOADED,
    CONVERSATION_SAVE_TIMEOUT_S,
    GROUNDING_NO_DEPLOYED_SCHEMA_FOR_PROFILE,
    AnswerGrounding,
)
from cogniverse_runtime.shared_state import SHARED_STATE_REDIS_TIMEOUT_SECONDS
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.docker_utils import generate_unique_ports
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.vespa_test_helpers import (
    DegradeConfigQueries,
    deploy_tenant_schema,
    make_config_manager,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
CLI_LOGGER = "cogniverse_runtime.runtime_cli"
MAIN_LOGGER = "cogniverse_runtime.main"
WORKERS = 2
# A worker loads every agent before it serves; two start side by side.
BOOT_TIMEOUT_S = 600
# Shutdown runs each worker's lifespan drains, all empty here.
STOP_TIMEOUT_S = 120
# How long a new connection may wait in a worker's accept queue.
ACCEPT_TIMEOUT_S = 30
_TCP_LISTEN = "0A"
SHIPPED_PROFILES = json.loads((ROOT / "configs/config.json").read_text())["backend"][
    "profiles"
]
# The profile a conversation tenant configures and never deploys, so the
# summarizer answers that it has nothing to search, with no LM and no encoder.
UNDEPLOYED_PROFILE = "document_text_semantic"


def _records(log: Path, logger: str, level: str) -> list[str]:
    return [
        fields[3]
        for line in log.read_text().splitlines()
        if len(fields := line.split(" - ", 3)) == 4 and fields[1:3] == [logger, level]
    ]


def _started_worker_pids(log: Path) -> list[int]:
    """Server processes uvicorn reports starting, in start order."""
    return [
        int(match.group(1))
        for line in log.read_text().splitlines()
        if (match := re.search(r"Started server process \[(\d+)\]", line))
    ]


def _until(predicate, process, log: Path, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            pytest.fail(
                f"runtime exited with {process.returncode} before {what}:\n"
                f"{log.read_text()[-20000:]}"
            )
        time.sleep(0.2)
    pytest.fail(
        f"runtime did not reach {what} in {timeout}s:\n{log.read_text()[-20000:]}"
    )


def _children(pid: int) -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        if int(stat.rsplit(")", 1)[1].split()[1]) == pid:
            found.append(int(entry))
    return sorted(found)


def _worker_processes(pid: int) -> list[int]:
    """The CLI's spawned worker children (not multiprocessing's tracker)."""
    workers = []
    for child in _children(pid):
        try:
            argv = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(b"spawn_main" in arg for arg in argv):
            workers.append(child)
    return workers


def _socket_inodes(pid: int) -> set[int]:
    inodes = set()
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(int(target[len("socket:[") : -1]))
    return inodes


def _tcp_rows() -> list[tuple[int, int, str, int]]:
    """(local port, remote port, state, inode) for every TCP socket."""
    rows = []
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            rows.append(
                (
                    int(fields[1].rsplit(":", 1)[1], 16),
                    int(fields[2].rsplit(":", 1)[1], 16),
                    fields[3],
                    int(fields[9]),
                )
            )
    return rows


def _owner(inode: int, pids: list[int]) -> int:
    owners = [pid for pid in pids if inode in _socket_inodes(pid)]
    assert len(owners) == 1, (inode, owners)
    return owners[0]


def _listening_owners(port: int, pids: list[int]) -> list[int]:
    return sorted(
        _owner(inode, pids)
        for local, _, state, inode in _tcp_rows()
        if local == port and state == _TCP_LISTEN
    )


def _serving_worker(port: int, connection: http.client.HTTPConnection, pids):
    """The worker holding the server side of an open client connection, once
    a worker has accepted it: until then the kernel lists the server side,
    queued on a listening socket, with no socket inode."""
    client_port = connection.sock.getsockname()[1]
    deadline = time.monotonic() + ACCEPT_TIMEOUT_S
    while True:
        inodes = [
            inode
            for local, remote, state, inode in _tcp_rows()
            if local == port and remote == client_port and state != _TCP_LISTEN
        ]
        assert len(inodes) == 1, (client_port, inodes)
        if inodes[0] != 0:
            return _owner(inodes[0], pids)
        assert time.monotonic() < deadline, f"no worker accepted {client_port}"
        time.sleep(0.05)


def _get(connection: http.client.HTTPConnection, path: str) -> tuple[int, dict]:
    connection.request("GET", path)
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def _memory_kib(pid: int) -> dict[str, int]:
    fields = dict(
        line.split(":", 1)
        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
    )
    return {key: int(fields[key].split()[0]) for key in ("VmRSS", "VmHWM")}


def _refuses(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == errno.ECONNREFUSED


@contextlib.contextmanager
def _runtime(
    tmp_path: Path,
    redis_url: str,
    name: str = "runtime",
    extra_env: dict[str, str] | None = None,
):
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    command = json.loads(
        next(
            line[4:]
            for line in (ROOT / "libs/runtime/Dockerfile").read_text().splitlines()
            if line.startswith("CMD ")
        )
    )
    command[0] = sys.executable
    command[command.index("--port") + 1] = str(port)
    env = dict(
        os.environ,
        UVICORN_WORKERS=str(WORKERS),
        REDIS_URL=redis_url,
        COGNIVERSE_SANDBOX_POLICY="disabled",
        COGNIVERSE_MEMORY_LIFECYCLE_DISABLED="1",
        LOG_LEVEL="INFO",
        PYTHONUNBUFFERED="1",
        **(extra_env or {}),
    )
    log = tmp_path / f"{name}.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
        )
        try:
            yield process, log, port
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    for pid in _children(process.pid):
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGKILL)
                    process.kill()
                    process.wait(timeout=10)


def _serving(process, log: Path) -> list[int]:
    """Wait until every worker finished its lifespan; return the worker pids."""
    _until(
        lambda: (
            _records(log, MAIN_LOGGER, "INFO").count(
                "Cogniverse Runtime started successfully"
            )
            == WORKERS
        ),
        process,
        log,
        BOOT_TIMEOUT_S,
        f"{WORKERS} started workers",
    )
    return _worker_processes(process.pid)


@pytest.fixture
def redis_url(workflow_state_redis_url):
    return workflow_state_redis_url


class TestTwoWorkersServe:
    def test_each_worker_runs_its_own_lifespan_on_its_own_socket(
        self, tmp_path, redis_url, vespa_instance
    ):
        with _runtime(tmp_path, redis_url) as (process, log, port):
            workers = _serving(process, log)

            assert len(workers) == WORKERS
            assert sorted(_started_worker_pids(log)) == workers
            print(
                "resident memory (KiB):",
                {pid: _memory_kib(pid) for pid in [process.pid, *workers]},
            )
            # Each worker listens on its own socket; the CLI process holds none.
            assert _listening_owners(port, [process.pid, *workers]) == workers

            # The backend wait ran once, in the CLI, before any worker started.
            assert _records(log, CLI_LOGGER, "INFO") == [
                "Waiting for backend startup readiness at "
                f"http://localhost:{vespa_instance['http_port']}...",
                "Backend feed endpoint is ready",
                f"Starting {WORKERS} runtime worker processes on 0.0.0.0:{port}",
            ]
            assert _records(log, CLI_LOGGER, "ERROR") == []

            # Each worker built its own A2A protocol under its own pid.
            mounted = [
                re.fullmatch(
                    r"A2A server mounted at /a2a with \d+ skills on replica "
                    r"[^:]+:(\d+):[0-9a-f]{8}",
                    record,
                )
                for record in _records(log, MAIN_LOGGER, "INFO")
                if record.startswith("A2A server mounted at /a2a")
            ]
            assert sorted(int(match.group(1)) for match in mounted) == workers

            # Both wrote the same deployment overrides; the stored row holds them.
            from cogniverse_foundation.config.utils import create_default_config_manager

            stored = create_default_config_manager().get_system_config()
            assert (stored.backend_url, stored.backend_port, stored.redis_url) == (
                "http://localhost",
                vespa_instance["http_port"],
                redis_url,
            )

    def test_concurrent_connections_reach_every_worker(
        self, tmp_path, redis_url, vespa_instance
    ):
        """32 clients connecting at once are spread over both workers, and every
        request on every connection is answered."""
        clients = 32
        with _runtime(tmp_path, redis_url) as (process, log, port):
            workers = _serving(process, log)
            barrier = threading.Barrier(clients)
            connections = [
                http.client.HTTPConnection("127.0.0.1", port, timeout=30)
                for _ in range(clients)
            ]
            answers: list[tuple[int, dict]] = []
            lock = threading.Lock()

            def client(connection):
                barrier.wait()
                for _ in range(3):
                    answer = _get(connection, "/health/live")
                    with lock:
                        answers.append(answer)

            threads = [threading.Thread(target=client, args=(c,)) for c in connections]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
            try:
                assert answers == [(200, {"status": "alive"})] * (3 * clients)
                owners = [_serving_worker(port, c, workers) for c in connections]
            finally:
                for connection in connections:
                    connection.close()

        assert len(owners) == clients
        assert set(owners) == set(workers)


def _post(
    connection: http.client.HTTPConnection, path: str, body: dict
) -> tuple[int, dict]:
    connection.request(
        "POST",
        path,
        body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def _connection_to(
    port: int, worker: int, workers: list[int]
) -> http.client.HTTPConnection:
    """A new client connection ``worker`` holds, read from /proc."""
    spare = []
    try:
        for _ in range(64):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
            connection.connect()
            if _serving_worker(port, connection, workers) == worker:
                return connection
            spare.append(connection)
    finally:
        for connection in spare:
            connection.close()
    raise AssertionError(f"no connection of 64 reached worker {worker}")


def _conversation_rows(
    connection: http.client.HTTPConnection, tenant_id: str, context_id: str
) -> list[dict]:
    status, body = _get(
        connection,
        f"/admin/tenant/{tenant_id}/memories?agent_name={CONVERSATION_AGENT_NAME}"
        "&limit=200",
    )
    assert status == 200, body
    rows = [
        row
        for row in body["memories"]
        if row["metadata"].get("context_id") == context_id
    ]
    return sorted(rows, key=lambda row: float(row["metadata"]["seq"]))


class TestConversationAcrossWorkers:
    def test_each_turn_reads_the_turns_the_other_worker_answered(
        self, tmp_path, redis_url, vespa_instance, shared_vespa, shared_denseon
    ):
        """Consecutive turns of one context alternate between the workers; each
        reads every turn answered before it, though the reply before it came
        back while that turn's save was still landing on the other worker."""
        # A tenant of its own: its memory schema is deployed, and its one
        # profile embeds through DenseOn but its schema is never deployed.
        tenant_id = f"workers{uuid.uuid4().hex[:8]}:unit"
        config_manager = make_config_manager(shared_vespa)
        deploy_tenant_schema(
            shared_vespa,
            tenant_id=tenant_id,
            base_schema_name="agent_memories",
            config_manager=config_manager,
        )
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(
                UNDEPLOYED_PROFILE,
                {
                    **SHIPPED_PROFILES[UNDEPLOYED_PROFILE],
                    "inference_services": {"embedding": "denseon"},
                },
            ),
            tenant_id=tenant_id,
        )
        expected_answer = AnswerGrounding(
            hits=[],
            state=GROUNDING_NO_DEPLOYED_SCHEMA_FOR_PROFILE,
            undeployed_profiles=(UNDEPLOYED_PROFILE,),
        ).unanswerable_text(tenant_id)
        context_id = f"workers-{uuid.uuid4().hex}"
        # Three turns cover both directions (first to second worker and back)
        # and stay below the turn count that files a wiki page.
        queries = [f"summarize turn {index}" for index in range(3)]
        env = {
            "INFERENCE_SERVICE_URLS": json.dumps({"denseon": shared_denseon}),
            "VESPA_CONFIG_PORT": str(vespa_instance["config_port"]),
        }
        with _runtime(tmp_path, redis_url, extra_env=env) as (process, log, port):
            workers = _serving(process, log)
            served = []
            # A connection per turn: a turn waits out the previous turn's save,
            # longer than a worker keeps an idle connection open.
            for index, query in enumerate(queries):
                worker = workers[index % 2]
                connection = _connection_to(port, worker, workers)
                try:
                    status, body = _post(
                        connection,
                        "/agents/summarizer_agent/process",
                        {
                            "agent_name": "summarizer_agent",
                            "query": query,
                            "context": {"tenant_id": tenant_id},
                            "context_id": context_id,
                        },
                    )
                finally:
                    connection.close()
                assert status == 200, (body, log.read_text()[-20000:])
                served.append((worker, body))

            connection = _connection_to(port, workers[0], workers)
            try:
                deadline = time.monotonic() + 2 * CONVERSATION_SAVE_TIMEOUT_S
                rows = _conversation_rows(connection, tenant_id, context_id)
                while len(rows) < 2 * len(queries) and time.monotonic() < deadline:
                    time.sleep(0.5)
                    rows = _conversation_rows(connection, tenant_id, context_id)
            finally:
                connection.close()

        assert [worker for worker, _ in served] == [
            workers[0],
            workers[1],
            workers[0],
        ]
        for index, (_, body) in enumerate(served):
            assert body["conversation"] == {
                "state": CONVERSATION_HISTORY_LOADED,
                "turn_count": 2 * index,
                "reason": None,
            }, body
            assert body["answer"] == expected_answer
        assert [row["memory"] for row in rows] == [
            f"[ctx:{context_id}] [{role}] {text}"
            for query in queries
            for role, text in (("user", query), ("assistant", expected_answer))
        ]
        assert [row["metadata"]["turn_role"] for row in rows] == [
            "user",
            "assistant",
        ] * len(queries)
        # Each turn's rows sit at its ledger position and the one after it,
        # and every turn's position is above the turn before it.
        seqs = [int(row["metadata"]["seq"]) for row in rows]
        assert [reply - user for user, reply in zip(seqs[0::2], seqs[1::2])] == [
            1
        ] * len(queries)
        assert [
            later - earlier >= 2 for earlier, later in zip(seqs[0::2], seqs[2::2])
        ] == [True] * (len(queries) - 1)
        assert _records(log, CLI_LOGGER, "ERROR") == []


class TestSignals:
    def test_a_reload_signal_before_the_lifespan_handler_is_ignored(
        self, tmp_path, redis_url, vespa_instance
    ):
        """A reload request reaching a worker that is still starting must not
        terminate it: the worker's lifespan reads configuration afresh."""
        registered = (
            "SIGUSR1 hot-reload handler registered "
            "(send `kill -USR1 <pid>` to reload config + sandbox policies)"
        )
        with _runtime(tmp_path, redis_url) as (process, log, _):
            _until(
                lambda: len(_started_worker_pids(log)) == WORKERS,
                process,
                log,
                BOOT_TIMEOUT_S,
                "both workers loading the app",
            )
            assert _records(log, MAIN_LOGGER, "INFO").count(registered) == 0
            for pid in _worker_processes(process.pid):
                os.kill(pid, signal.SIGUSR1)
            workers = _serving(process, log)

            assert sorted(_started_worker_pids(log)) == workers
            assert _records(log, MAIN_LOGGER, "INFO").count(registered) == WORKERS
            assert [
                record
                for record in _records(log, MAIN_LOGGER, "INFO")
                if record.startswith("SIGUSR1 received")
            ] == []
            assert _records(log, CLI_LOGGER, "ERROR") == []
            assert process.poll() is None

    def test_sigusr1_to_the_cli_reloads_every_worker(
        self, tmp_path, redis_url, vespa_instance
    ):
        with _runtime(tmp_path, redis_url) as (process, log, _):
            _serving(process, log)
            process.send_signal(signal.SIGUSR1)
            _until(
                lambda: (
                    _records(log, MAIN_LOGGER, "INFO").count("Hot-reload complete")
                    == WORKERS
                ),
                process,
                log,
                60,
                "every worker's hot-reload",
            )
            received = [
                record
                for record in _records(log, MAIN_LOGGER, "INFO")
                if record.startswith("SIGUSR1 received")
            ]
            assert (
                received
                == ["SIGUSR1 received — hot-reloading configuration (count=1)"]
                * WORKERS
            )
            assert process.poll() is None

    def test_sigterm_stops_every_worker_and_exits_zero(
        self, tmp_path, redis_url, vespa_instance
    ):
        with _runtime(tmp_path, redis_url) as (process, log, port):
            workers = _serving(process, log)
            process.send_signal(signal.SIGTERM)
            code = process.wait(timeout=STOP_TIMEOUT_S)

            assert code == 0
            assert (
                _records(log, MAIN_LOGGER, "INFO").count(
                    "Cogniverse Runtime shut down successfully"
                )
                == WORKERS
            )
            assert [pid for pid in workers if Path(f"/proc/{pid}").exists()] == []
            assert _refuses(port)
            assert _records(log, CLI_LOGGER, "ERROR") == []


SYSTEM_CONFIG_ID = "_system:system:system:system_config"


def _system_config_version(http_port: int) -> int:
    store = VespaConfigStore(backend_url="http://localhost", backend_port=http_port)
    try:
        return store.get_config(
            "_system", ConfigScope.SYSTEM, "system", "system_config"
        ).version
    finally:
        store.close()


class TestStartupThroughADegradedStore:
    def test_workers_start_while_every_config_query_answers_degraded(
        self, tmp_path, redis_url, vespa_instance
    ):
        """Every query on the config store answers as Vespa does while its
        content node is outside its ideal state. A worker's startup writes
        read the latest version by visiting the stored versions, so neither
        waits: both workers serve, each write lands above the latest version,
        and the only system-config query each makes is its prune listing."""
        degrade = DegradeConfigQueries()
        before = _system_config_version(vespa_instance["http_port"])
        # The runtime derives the config server's port from the data port.
        data_port, config_port = generate_unique_ports("degraded-store-proxy")
        with (
            InterceptFaultProxy(
                f"http://localhost:{vespa_instance['http_port']}",
                degrade,
                port=data_port,
            ),
            InterceptFaultProxy(
                f"http://localhost:{vespa_instance['config_port']}", port=config_port
            ),
        ):
            env = {"BACKEND_URL": "http://127.0.0.1", "BACKEND_PORT": str(data_port)}
            with _runtime(tmp_path, redis_url, extra_env=env) as (process, log, _):
                workers = _serving(process, log)
                waits = [
                    record
                    for record in _records(log, MAIN_LOGGER, "WARNING")
                    if record.startswith("Config store for the startup ")
                ]

        keep = inspect.signature(VespaConfigStore).parameters["keep_versions"].default
        prune_listing = (
            "select version from config_metadata where config_id contains "
            f'"{SYSTEM_CONFIG_ID}" order by version desc limit {keep + 100}'
        )
        assert len(workers) == WORKERS
        assert waits == []
        assert [
            query for query in degrade.queries if f'"{SYSTEM_CONFIG_ID}"' in query
        ] == [prune_listing] * WORKERS
        assert _system_config_version(vespa_instance["http_port"]) == before + WORKERS
        assert _records(log, CLI_LOGGER, "ERROR") == []


class TestWorkerFailure:
    def test_a_worker_that_dies_stops_the_runtime(
        self, tmp_path, redis_url, vespa_instance
    ):
        """The pod restarts as it would for a single process; the survivor
        shuts down through its lifespan, nothing is respawned."""
        with _runtime(tmp_path, redis_url) as (process, log, port):
            workers = _serving(process, log)
            killed, survivor = workers
            os.kill(killed, signal.SIGKILL)
            code = process.wait(timeout=STOP_TIMEOUT_S)

            assert code == 1
            assert _records(log, CLI_LOGGER, "ERROR") == [
                f"Runtime worker {killed} exited with code -9; stopping the runtime"
            ]
            assert (
                _records(log, MAIN_LOGGER, "INFO").count(
                    "Cogniverse Runtime shut down successfully"
                )
                == 1
            )
            assert sorted(_started_worker_pids(log)) == workers
            assert not Path(f"/proc/{survivor}").exists()
            assert _refuses(port)

    def test_a_worker_whose_startup_fails_stops_the_runtime(
        self, tmp_path, vespa_instance
    ):
        """An unreachable A2A Redis fails every worker's lifespan: the CLI exits
        non-zero after the first, instead of restarting workers forever while
        the port stays closed."""
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            dead_redis = f"redis://127.0.0.1:{reserved.getsockname()[1]}/0"
        with _runtime(tmp_path, dead_redis) as (process, log, port):
            code = process.wait(timeout=BOOT_TIMEOUT_S)

            started = _started_worker_pids(log)
            assert code == 1
            assert len(started) == WORKERS
            # Whichever worker failed first stops the runtime.
            assert _records(log, CLI_LOGGER, "ERROR") in [
                [f"Runtime worker {pid} exited with code 3; stopping the runtime"]
                for pid in started
            ]
            assert (
                _records(log, MAIN_LOGGER, "INFO").count(
                    "Cogniverse Runtime started successfully"
                )
                == 0
            )
            assert _refuses(port)


def _answered_connection_to(port: int, worker: int, workers: list[int]):
    """A connection that ``worker`` serves.

    Which worker accepts a connection is the kernel's choice, read from
    ``/proc`` after a harmless request; a connection another worker took is
    closed and a new one opened.
    """
    for _ in range(200):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        assert _get(connection, "/health/live") == (200, {"status": "alive"})
        if _serving_worker(port, connection, workers) == worker:
            return connection
        connection.close()
    pytest.fail(f"no connection reached worker {worker} in 200 attempts")


def _send(connection, method: str, path: str, body=None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(
        method, path, body=None if body is None else json.dumps(body), headers=headers
    )
    response = connection.getresponse()
    return response.status, json.loads(response.read())


class _Worker:
    """Sends each request on a fresh connection that this worker serves."""

    def __init__(self, port: int, pid: int, workers: list[int]):
        self.port, self.pid, self.workers = port, pid, workers

    def __call__(self, method: str, path: str, body=None) -> tuple[int, dict]:
        connection = _answered_connection_to(self.port, self.pid, self.workers)
        try:
            return _send(connection, method, path, body)
        finally:
            connection.close()


def _at_once(port: int, workers: list[int], clients: int, method, path, body):
    """``clients`` connections send the same request at once; returns each
    answer with the worker that served it."""
    connections = [
        http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        for _ in range(clients)
    ]
    for connection in connections:
        assert _get(connection, "/health/live") == (200, {"status": "alive"})
    barrier = threading.Barrier(clients)
    answers: list = [None] * clients
    owners: list = [None] * clients

    def client(index):
        barrier.wait()
        answers[index] = _send(connections[index], method, path, body)
        # Read while the connection is open: an idle one is closed by the
        # server once its keep-alive lapses.
        owners[index] = _serving_worker(port, connections[index], workers)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        for connection in connections:
            connection.close()
    return list(zip(answers, owners))


def _annotation(span_id: str) -> dict:
    return {
        "span_id": span_id,
        "timestamp": "2026-09-01T11:00:00+00:00",
        "query": "search for video clips of animals",
        "chosen_agent": "search_agent",
        "routing_confidence": 0.42,
        "outcome": "ambiguous",
        "priority": "medium",
        "reason": "two workers",
        "context": {"tags": []},
        "status": "pending",
        "assigned_to": None,
        "assigned_at": None,
        "sla_deadline": None,
        "completed_at": None,
        "label": None,
        "agent_type": "routing",
        "tenant_id": None,
    }


@pytest.fixture(scope="class")
def two_workers(tmp_path_factory, workflow_state_redis_url, vespa_instance):
    """One runtime serving from two workers, and a request sender for each."""
    tmp_path = tmp_path_factory.mktemp("shared_state")
    with _runtime(tmp_path, workflow_state_redis_url) as (process, log, port):
        workers = _serving(process, log)
        assert len(workers) == WORKERS
        yield port, workers, [_Worker(port, pid, workers) for pid in workers], tmp_path


class TestSharedStateAcrossWorkers:
    """What one worker accepts, the other serves."""

    def test_an_agent_registered_on_one_worker_is_served_by_the_other(
        self, two_workers
    ):
        _, _, (first, second), _ = two_workers
        name = f"mp_agent_{uuid.uuid4().hex[:8]}"
        listed_before = first("GET", "/agents/")[1]["agents"]

        registered = first(
            "POST",
            "/agents/register",
            {"name": name, "url": "http://external:9000", "capabilities": ["mp"]},
        )
        info = second("GET", f"/agents/{name}")
        listed = second("GET", "/agents/")
        removed = second("DELETE", f"/agents/{name}")

        assert registered == (
            201,
            {
                "status": "registered",
                "agent": name,
                "url": "http://external:9000",
                "capabilities": ["mp"],
            },
        )
        assert info == (
            200,
            {
                "name": name,
                "url": "http://external:9000",
                "capabilities": ["mp"],
                "health_status": "unknown",
                "health_endpoint": "/health",
                "process_endpoint": "/tasks/send",
            },
        )
        assert listed[0] == 200
        assert listed[1]["count"] == len(listed_before) + 1
        assert set(listed[1]["agents"]) == {*listed_before, name}
        assert removed == (200, {"status": "unregistered", "agent": name})
        assert first("GET", f"/agents/{name}") == (
            404,
            {"detail": f"Agent '{name}' not found"},
        )
        assert first("DELETE", f"/agents/{name}") == (
            404,
            {"detail": f"Agent '{name}' not found"},
        )

    def test_an_annotation_moves_through_its_lifecycle_across_workers(
        self, two_workers
    ):
        _, _, (first, second), _ = two_workers
        span_id = f"mp-span-{uuid.uuid4().hex}"
        request = _annotation(span_id)
        total = first("GET", "/agents/annotations/queue")[1]["statistics"]["total"]

        enqueued = first(
            "POST", "/agents/annotations/queue/enqueue", {"requests": [request]}
        )
        stored = second("GET", f"/agents/annotations/queue/{span_id}")
        assigned = second(
            "POST",
            f"/agents/annotations/queue/{span_id}/assign",
            {"reviewer": "mp-reviewer", "sla_hours": 1},
        )
        seen_assigned = first("GET", f"/agents/annotations/queue/{span_id}")
        completed = first(
            "POST",
            f"/agents/annotations/queue/{span_id}/complete",
            {"reasoning": "two workers"},
        )
        again = second("POST", f"/agents/annotations/queue/{span_id}/complete", {})

        assert enqueued == (
            200,
            {"enqueued": 1, "skipped": 0, "queue_total": total + 1},
        )
        assert stored == (200, request)
        assert assigned[0] == 200
        assert seen_assigned == (200, assigned[1]["annotation"])
        assert assigned[1]["annotation"]["assigned_to"] == "mp-reviewer"
        assert completed[0] == 200
        assert completed[1]["persisted"] is False
        assert completed[1]["annotation"]["status"] == "completed"
        assert again == (
            400,
            {"detail": f"Cannot complete span {span_id}: status is completed"},
        )

    def test_concurrent_enqueues_on_both_workers_add_each_span_once(self, two_workers):
        port, workers, (first, _), _ = two_workers
        spans = [f"mp-batch-{uuid.uuid4().hex}" for _ in range(8)]
        total = first("GET", "/agents/annotations/queue")[1]["statistics"]["total"]

        answers = _at_once(
            port,
            workers,
            32,
            "POST",
            "/agents/annotations/queue/enqueue",
            {"requests": [_annotation(span) for span in spans]},
        )

        assert set(owner for _, owner in answers) == set(workers)
        assert [status for (status, _), _ in answers] == [200] * 32
        assert sum(body["enqueued"] for (_, body), _ in answers) == 8
        assert (
            first("GET", "/agents/annotations/queue")[1]["statistics"]["total"]
            == total + 8
        )

    def test_concurrent_assigns_on_both_workers_admit_exactly_one(self, two_workers):
        port, workers, (first, _), _ = two_workers
        span_id = f"mp-contended-{uuid.uuid4().hex}"
        first(
            "POST",
            "/agents/annotations/queue/enqueue",
            {"requests": [_annotation(span_id)]},
        )

        answers = _at_once(
            port,
            workers,
            32,
            "POST",
            f"/agents/annotations/queue/{span_id}/assign",
            {"reviewer": "contender"},
        )

        assert set(owner for _, owner in answers) == set(workers)
        assert sorted(status for (status, _), _ in answers) == [200] + [400] * 31
        assert {body["detail"] for (status, body), _ in answers if status == 400} == {
            f"Cannot assign span {span_id}: status is assigned"
        }

    def test_concurrent_completions_on_both_workers_admit_exactly_one(
        self, two_workers
    ):
        port, workers, (first, second), _ = two_workers
        span_id = f"mp-complete-{uuid.uuid4().hex}"
        first(
            "POST",
            "/agents/annotations/queue/enqueue",
            {"requests": [_annotation(span_id)]},
        )

        answers = _at_once(
            port,
            workers,
            32,
            "POST",
            f"/agents/annotations/queue/{span_id}/complete",
            {"label": "correct_routing"},
        )
        winners = [body for (status, body), _ in answers if status == 200]
        refusals = {
            (status, body["detail"]) for (status, body), _ in answers if status != 200
        }

        assert set(owner for _, owner in answers) == set(workers)
        assert len(winners) == 1
        assert refusals <= {
            (400, f"Cannot complete span {span_id}: status is completed"),
            (409, f"Span {span_id} is being completed by another request"),
        }
        assert second("GET", f"/agents/annotations/queue/{span_id}") == (
            200,
            winners[0]["annotation"],
        )

    def test_an_ingestion_job_started_on_one_worker_is_read_on_the_other(
        self, two_workers
    ):
        _, _, (first, second), tmp_path = two_workers
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        started = first(
            "POST",
            "/ingestion/start",
            {
                "video_dir": str(video_dir),
                "profile": "mp_unknown_profile",
                "tenant_id": "__system__",
            },
        )
        job_id = started[1]["job_id"]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = second("GET", f"/ingestion/status/{job_id}")
            if status[0] != 200 or status[1]["status"] not in ("started", "processing"):
                break
            time.sleep(0.5)

        assert started == (
            200,
            {
                "job_id": job_id,
                "status": "started",
                "message": "Ingestion job started successfully",
            },
        )
        assert status == (
            200,
            {
                "job_id": job_id,
                "status": "failed",
                "videos_processed": 0,
                "videos_total": 0,
                "errors": [
                    "Profile mp_unknown_profile missing 'strategies' configuration. "
                    "All profiles must use explicit strategy configuration."
                ],
            },
        )
        assert first("GET", f"/ingestion/status/{job_id}") == status


class TestSharedStateOutage:
    def test_a_paused_redis_answers_503_on_every_worker_and_recovers(
        self, tmp_path, own_redis, vespa_instance
    ):
        url, pause, resume = own_redis
        with _runtime(tmp_path, url) as (process, log, port):
            workers = _serving(process, log)
            senders = [_Worker(port, worker, workers) for worker in workers]
            pause()
            started = time.monotonic()
            down = [
                (
                    send("GET", "/agents/"),
                    send("GET", "/agents/annotations/queue"),
                    send("GET", "/ingestion/status/any-job"),
                )
                for send in senders
            ]
            elapsed = time.monotonic() - started
            resume()
            up = [send("GET", "/agents/annotations/queue")[0] for send in senders]

        unavailable = (
            (
                503,
                {
                    "detail": {
                        "error": "agent_registry_unavailable",
                        "message": "The shared agent registry did not answer; retry.",
                        "failure": "AgentRegistryUnavailableError",
                    }
                },
            ),
            (
                503,
                {
                    "detail": {
                        "error": "annotation_queue_unavailable",
                        "message": "The annotation queue did not answer; retry.",
                        "failure": "AnnotationQueueUnavailableError",
                    }
                },
            ),
            (
                503,
                {
                    "detail": {
                        "error": "ingestion_job_store_unavailable",
                        "message": "The ingestion job store did not answer; retry.",
                        "failure": "IngestionJobStoreUnavailableError",
                        "job_id": "any-job",
                    }
                },
            ),
        )
        assert down == [unavailable] * WORKERS
        assert sorted(
            _records(log, "cogniverse_runtime.http_errors", "ERROR")
        ) == sorted(
            [
                "agent_registry_unavailable: AgentRegistryUnavailableError: "
                "shared agent registry unavailable: read version",
                "annotation_queue_unavailable: AnnotationQueueUnavailableError: "
                "annotation queue unavailable: read queue",
                "ingestion_job_store_unavailable: IngestionJobStoreUnavailableError: "
                "ingestion job store unavailable: read job any-job",
            ]
            * WORKERS
        )
        # Six requests, each bounded by the shared-state client's command timeout.
        assert elapsed < 6 * SHARED_STATE_REDIS_TIMEOUT_SECONDS + 10, elapsed
        assert up == [200] * WORKERS
