"""ACP process contracts over operating-system pipes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

_HANDSHAKE = {
    "protocolVersion": 1,
    "agentCapabilities": {
        "loadSession": False,
        "promptCapabilities": {
            "image": True,
            "audio": False,
            "embeddedContext": True,
        },
    },
    "agentInfo": {
        "name": "cogniverse",
        "title": "Cogniverse",
        "version": version("cogniverse-runtime"),
    },
    "authMethods": [],
}


def _environment(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(tmp_path),
        "COGNIVERSE_ACP_TENANT": "test:acp",
        "COGNIVERSE_ACP_AGENT": "probe_agent",
        "COGNIVERSE_ACP_CODING_AGENT": "probe_agent",
        "LOG_LEVEL": "INFO",
        "BACKEND_URL": "http://127.0.0.1",
        "BACKEND_PORT": "29071",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "COGNIVERSE_RLM_SKIP_DENO_CHECK": "1",
    }
    env.update(overrides)
    return env


@asynccontextmanager
async def _process(tmp_path: Path, **env: str):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "cogniverse_runtime.acp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_environment(tmp_path, **env),
    )
    stderr = asyncio.create_task(process.stderr.read())
    process.acp_stderr = stderr
    try:
        yield process, stderr
    finally:
        if process.returncode is None:
            process.kill()
        await process.stdout.read()
        await process.wait()
        (tmp_path / "stderr.log").write_bytes(await stderr)


async def _send(process, message):
    process.stdin.write(json.dumps(message).encode() + b"\n")
    await process.stdin.drain()


async def _receive(process):
    line = await asyncio.wait_for(process.stdout.readline(), 30)
    assert line != b"", (await process.acp_stderr).decode()
    return json.loads(line)


async def _initialize(process, request_id=1):
    await _send(
        process,
        {"jsonrpc": "2.0", "id": request_id, "method": "initialize", "params": {}},
    )
    assert await _receive(process) == {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": _HANDSHAKE,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        1,
        True,
        "message",
        [],
        [{"id": 2}],
        [{"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}}],
    ],
)
async def test_non_object_request_preserves_process(tmp_path, invalid):
    async with _process(tmp_path) as (process, stderr):
        await _send(process, invalid)
        assert await _receive(process) == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request: expected an object"},
        }
        await _initialize(process)
        process.stdin.close()
        assert await asyncio.wait_for(process.wait(), 10) == 0
        assert b"Traceback" not in await stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [0, 1, 32 * 1024 * 1024])
async def test_input_line_limit_and_exit_status(tmp_path, extra):
    from cogniverse_runtime.acp.__main__ import _STDIN_LINE_LIMIT

    assert _STDIN_LINE_LIMIT == 64 * 1024 * 1024

    async with _process(tmp_path) as (process, stderr):
        await _initialize(process)
        baseline = _rss_kib(process.pid)
        assert 64 * 1024 <= baseline < 1024 * 1024
        peak = baseline
        monitoring = True

        async def monitor():
            nonlocal peak
            while monitoring:
                peak = max(peak, _rss_kib(process.pid))
                await asyncio.sleep(0.005)

        task = asyncio.create_task(monitor())
        prefix = b'{"jsonrpc":"2.0","id":2,"method":"initialize","params":{}}'
        process.stdin.write(prefix)
        remaining = _STDIN_LINE_LIMIT + extra - len(prefix)
        try:
            while remaining:
                size = min(65536, remaining)
                process.stdin.write(b" " * size)
                await process.stdin.drain()
                remaining -= size
            process.stdin.write(b"\n")
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            assert extra == 32 * 1024 * 1024
        if extra == 0:
            assert await _receive(process) == {
                "jsonrpc": "2.0",
                "id": 2,
                "result": _HANDSHAKE,
            }
            await _initialize(process, 3)
            process.stdin.close()
            expected_exit = 0
        else:
            assert await _receive(process) == {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": f"Input line exceeds {_STDIN_LINE_LIMIT} byte limit",
                },
            }
            expected_exit = 1
        assert await asyncio.wait_for(process.wait(), 15) == expected_exit
        monitoring = False
        await task
        print(
            f"ACP_LINE_BYTES={_STDIN_LINE_LIMIT + extra} PEAK_RSS_KIB={peak} BASELINE_RSS_KIB={baseline}"
        )
        assert peak - baseline < 4 * (_STDIN_LINE_LIMIT // 1024)
        assert b"Traceback" not in await stderr


def _rss_kib(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return 0
    return next(
        (int(line.split()[1]) for line in lines if line.startswith("VmRSS:")), 0
    )


def _hook(tmp_path, body):
    source = """
import sys

def trace(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "_run" and frame.f_globals.get("__package__") == "cogniverse_runtime.acp":
        sys.settrace(None)
        namespace = frame.f_globals
        BODY
    return trace

sys.settrace(trace)
"""
    source = textwrap.dedent(source).replace(
        "        BODY", textwrap.indent(textwrap.dedent(body), "        ")
    )
    (tmp_path / "sitecustomize.py").write_text(source)


@pytest.mark.asyncio
async def test_stdout_backpressure_keeps_event_loop_responsive(tmp_path):
    _hook(
        tmp_path,
        """
import asyncio, json, threading, time
from cogniverse_runtime.acp.server import ACPServer
initialize = ACPServer.initialize
def large_reply(self, params):
    result = initialize(self, params)
    if params.get("pressure"):
        result["payload"] = "x" * (2 * 1024 * 1024)
    return result
ACPServer.initialize = large_reply
factory = namespace["_make_writer"]
def instrument(stream):
    writer = factory(stream)
    async def write(message):
        if message.get("id") == 2:
            loop = asyncio.get_running_loop()
            def post():
                due = time.perf_counter()
                def measure():
                    elapsed = (time.perf_counter() - due) * 1000
                    print("LOOP_PROBE_MS=" + str(elapsed), file=sys.stderr, flush=True)
                loop.call_soon_threadsafe(measure)
            threading.Timer(0.05, post).start()
        await writer(message)
    write.close = writer.close
    return write
namespace["_make_writer"] = instrument
""",
    )
    async with _process(tmp_path) as (process, stderr):
        await _initialize(process)
        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {"pressure": True},
            },
        )
        await asyncio.sleep(0.6)
        data = bytearray()
        while not data.endswith(b"\n"):
            data.extend(await asyncio.wait_for(process.stdout.read(65536), 10))
        response = json.loads(data)
        assert response == {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {**_HANDSHAKE, "payload": "x" * (2 * 1024 * 1024)},
        }
        process.stdin.close()
        assert await asyncio.wait_for(process.wait(), 10) == 0
        lines = (await stderr).decode().splitlines()
        delays = [
            float(line.removeprefix("LOOP_PROBE_MS="))
            for line in lines
            if line.startswith("LOOP_PROBE_MS=")
        ]
        assert len(delays) == 1
        print(f"ACP_BACKPRESSURE_PROBE_MS={delays[0]}")
        assert delays[0] < 50


def test_dispatcher_is_built_once_for_concurrent_first_calls():
    from cogniverse_runtime.acp.__main__ import _LazyDispatcher

    barrier = threading.Barrier(8)
    building = threading.Event()
    release = threading.Event()
    builds = []
    dispatcher = object()

    def build():
        builds.append("build")
        building.set()
        assert release.wait(5) is True
        return object(), dispatcher

    provider = _LazyDispatcher(build)

    def request():
        barrier.wait(timeout=5)
        return provider()

    with ThreadPoolExecutor(max_workers=8) as pool:
        requests = [pool.submit(request) for _ in range(8)]
        assert building.wait(5) is True
        release.set()
        results = [request.result(timeout=5) for request in requests]
    assert builds == ["build"]
    assert results == [dispatcher] * 8


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["", "probe_agent"])
async def test_invalid_model_map_is_startup_failure(tmp_path, override):
    config = tmp_path / "config.json"
    config.write_text('{"harness":{"models":')
    async with _process(
        tmp_path,
        COGNIVERSE_ACP_AGENT=override,
        COGNIVERSE_ACP_CODING_AGENT=override,
        COGNIVERSE_CONFIG=str(config),
    ) as (process, stderr):
        await _send(
            process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert await asyncio.wait_for(process.wait(), 30) == 2
        assert await process.stdout.read() == b""
        assert "ACP startup failed" in (await stderr).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["DEBUG", "ERROR"])
async def test_root_log_level_follows_entrypoint_environment(tmp_path, level):
    _hook(
        tmp_path,
        """
import logging
from cogniverse_runtime.acp.server import ACPServer
initialize = ACPServer.initialize
def report(self, params):
    logging.getLogger().debug("ACP_ROOT_DEBUG_SENTINEL")
    logging.getLogger().error("ACP_ROOT_ERROR_SENTINEL")
    return initialize(self, params)
ACPServer.initialize = report
""",
    )
    async with _process(tmp_path, LOG_LEVEL=level) as (process, stderr):
        await _initialize(process)
        process.stdin.close()
        assert await asyncio.wait_for(process.wait(), 10) == 0
        output = (await stderr).decode()
        assert ("ACP_ROOT_DEBUG_SENTINEL" in output) == (level == "DEBUG")
        assert "ACP_ROOT_ERROR_SENTINEL" in output


@pytest.mark.asyncio
async def test_model_map_selects_both_agents_before_handshake(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "harness": {
                    "models": {
                        "cogniverse": "configured_answer",
                        "cogniverse/coding": "configured_coder",
                    }
                }
            }
        )
    )
    _hook(
        tmp_path,
        """
from cogniverse_runtime.acp.server import ACPServer
initialize = ACPServer.initialize
def report(self, params):
    print("ACP_SELECTED_AGENTS=" + self._agent_provider() + "," + self._coding_agent_provider(), file=sys.stderr, flush=True)
    return initialize(self, params)
ACPServer.initialize = report
""",
    )
    async with _process(
        tmp_path,
        COGNIVERSE_ACP_AGENT="",
        COGNIVERSE_ACP_CODING_AGENT="",
        COGNIVERSE_CONFIG=str(config),
    ) as (process, stderr):
        await _initialize(process)
        process.stdin.close()
        assert await asyncio.wait_for(process.wait(), 10) == 0
        selected = [
            line
            for line in (await stderr).decode().splitlines()
            if line.startswith("ACP_SELECTED_AGENTS=")
        ]
        assert selected == ["ACP_SELECTED_AGENTS=configured_answer,configured_coder"]


def test_failed_dispatcher_build_can_be_retried():
    from cogniverse_runtime.acp.__main__ import _LazyDispatcher

    builds = []
    dispatcher = object()

    def build():
        builds.append("build")
        if len(builds) == 1:
            raise ConnectionError("dispatcher dependency 127.0.0.1:29071 refused")
        return object(), dispatcher

    provider = _LazyDispatcher(build)
    with pytest.raises(
        ConnectionError, match="dispatcher dependency 127.0.0.1:29071 refused"
    ):
        provider()
    assert [provider(), provider()] == [dispatcher, dispatcher]
    assert builds == ["build", "build"]


@pytest.mark.asyncio
async def test_cancelled_write_finishes_its_frame_before_the_next_message():
    from cogniverse_runtime.acp.__main__ import _make_writer

    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    writer = _make_writer(os.fdopen(write_fd, "wb", buffering=0))
    first = {"jsonrpc": "2.0", "id": 1, "result": "x" * (2 * 1024 * 1024)}
    second = {"jsonrpc": "2.0", "id": 2, "result": "after cancellation"}
    first_task = asyncio.create_task(writer(first))
    second_task = None
    try:
        async with asyncio.timeout(3):
            while (
                writer._writer is None
                or writer._writer.transport.get_write_buffer_size() < 65536
            ):
                await asyncio.sleep(0.005)
        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task
        second_task = asyncio.create_task(writer(second))
        data = bytearray()
        deadline = asyncio.get_running_loop().time() + 3
        while data.count(b"\n") != 2 and asyncio.get_running_loop().time() < deadline:
            try:
                data.extend(os.read(read_fd, 65536))
            except BlockingIOError:
                await asyncio.sleep(0.001)
        await asyncio.wait_for(second_task, 3)
        expected = b"".join(
            (json.dumps(message, separators=(",", ":")) + "\n").encode()
            for message in [first, second]
        )
        assert data == expected
    finally:
        await writer.close()
        for task in [first_task, second_task]:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        os.close(read_fd)


@pytest.mark.asyncio
async def test_writer_close_cancels_blocked_frame_and_rejects_waiting_writes():
    from cogniverse_runtime.acp.__main__ import _make_writer

    read_fd, write_fd = os.pipe()
    writer = _make_writer(os.fdopen(write_fd, "wb", buffering=0))
    first = asyncio.create_task(writer({"payload": "x" * (2 * 1024 * 1024)}))
    second = None
    try:
        async with asyncio.timeout(3):
            while (
                writer._writer is None
                or writer._writer.transport.get_write_buffer_size() < 65536
            ):
                await asyncio.sleep(0.005)
        second = asyncio.create_task(writer({"payload": "queued"}))
        await asyncio.wait_for(writer.close(), 2)
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(ConnectionError, match="ACP stdout is closed"):
            await second
        assert writer._pending_write is None
    finally:
        await writer.close()
        for task in [first, second]:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        os.close(read_fd)


@pytest.mark.asyncio
async def test_eof_during_stdout_backpressure_exits_without_waiting_for_editor(
    tmp_path,
):
    blocked = tmp_path / "stdout-blocked"
    _hook(
        tmp_path,
        """
import asyncio
from pathlib import Path
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader
from tests.runtime.integration import test_acp_connection as producer
from tests.utils.memory_store import InMemoryConfigStore
from cogniverse_foundation.telemetry import manager as telemetry
from cogniverse_foundation.telemetry.config import TelemetryConfig
telemetry._telemetry_manager = telemetry.TelemetryManager(TelemetryConfig(enabled=False, otlp_enabled=False))
producer.ANSWER = "Z" * (2 * 1024 * 1024)
store = InMemoryConfigStore()
store.initialize()
manager = ConfigManager(store=store)
registry = AgentRegistry(tenant_id="test:acp", config_manager=manager)
registry.register_agent(AgentEndpoint(name="probe_agent", url="http://127.0.0.1:29071", capabilities=["acp_pipe"], streams_answer_tokens=False))
ConfigLoader.AGENT_CLASSES["probe_agent"] = "tests.runtime.integration.test_acp_connection:PipeTurnAgent"
namespace["_build_dispatcher"] = lambda: (manager, AgentDispatcher(agent_registry=registry, config_manager=manager, schema_loader=None))
factory = namespace["_make_writer"]
def instrument(stream):
    writer = factory(stream)
    async def monitor():
        while writer._writer is None or writer._writer.transport.get_write_buffer_size() < 65536:
            await asyncio.sleep(0.005)
        Path(BLOCKED_PATH).write_text("blocked")
    writer._probe_task = asyncio.create_task(monitor())
    return writer
namespace["_make_writer"] = instrument
""".replace("BLOCKED_PATH", repr(str(blocked))),
    )
    async with _process(
        tmp_path,
        PYTHONPATH=os.pathsep.join(
            [str(tmp_path), str(Path(__file__).resolve().parents[3])]
        ),
    ) as (process, stderr):
        await _initialize(process)
        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path)},
            },
        )
        session = (await _receive(process))["result"]["sessionId"]
        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session,
                    "prompt": [{"type": "text", "text": "large response"}],
                },
            },
        )
        async with asyncio.timeout(15):
            while not blocked.exists():
                await asyncio.sleep(0.005)
        assert blocked.read_text() == "blocked"
        process.stdin.close()
        async with asyncio.timeout(3):
            while process.returncode is None:
                await asyncio.sleep(0.005)
        assert process.returncode == 0
        assert b"Traceback" not in await stderr
