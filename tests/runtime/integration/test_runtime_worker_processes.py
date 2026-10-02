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
from tests.utils.vespa_test_helpers import deploy_tenant_schema, make_config_manager

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
CLI_LOGGER = "cogniverse_runtime.runtime_cli"
MAIN_LOGGER = "cogniverse_runtime.main"
WORKERS = 2
# A worker loads every agent before it serves; two start side by side.
BOOT_TIMEOUT_S = 600
# Shutdown runs each worker's lifespan drains, all empty here.
STOP_TIMEOUT_S = 120
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
    """The worker holding the server side of an open client connection."""
    client_port = connection.sock.getsockname()[1]
    inodes = [
        inode
        for local, remote, state, inode in _tcp_rows()
        if local == port and remote == client_port and state != _TCP_LISTEN
    ]
    assert len(inodes) == 1, (client_port, inodes)
    return _owner(inodes[0], pids)


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


def _connection_per_worker(
    port: int, workers: list[int]
) -> dict[int, http.client.HTTPConnection]:
    """One open client connection held by each worker, read from /proc."""
    held: dict[int, http.client.HTTPConnection] = {}
    spare = []
    for _ in range(64):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        connection.connect()
        worker = _serving_worker(port, connection, workers)
        if worker in held:
            spare.append(connection)
        else:
            held[worker] = connection
        if len(held) == len(workers):
            break
    for connection in spare:
        connection.close()
    assert sorted(held) == workers
    return held


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
            connections = _connection_per_worker(port, workers)
            served = []
            try:
                for index, query in enumerate(queries):
                    worker = workers[index % 2]
                    status, body = _post(
                        connections[worker],
                        "/agents/summarizer_agent/process",
                        {
                            "agent_name": "summarizer_agent",
                            "query": query,
                            "context": {"tenant_id": tenant_id},
                            "context_id": context_id,
                        },
                    )
                    assert status == 200, (body, log.read_text()[-20000:])
                    served.append((worker, body))

                deadline = time.monotonic() + 2 * CONVERSATION_SAVE_TIMEOUT_S
                rows = _conversation_rows(
                    connections[workers[0]], tenant_id, context_id
                )
                while len(rows) < 2 * len(queries) and time.monotonic() < deadline:
                    time.sleep(0.5)
                    rows = _conversation_rows(
                        connections[workers[0]], tenant_id, context_id
                    )
            finally:
                for connection in connections.values():
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
