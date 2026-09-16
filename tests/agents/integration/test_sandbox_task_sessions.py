"""A coding task owns one sandbox from its first write to its cleanup.

Real ``SandboxManager``, ``SandboxSessionPool``, ``CodingAgent`` and the
shipped ``openshell`` SDK, driven over real gRPC against a gateway this
module serves on loopback. Each sandbox is a separate directory and every
command is a real OS process, so file and process visibility across
sessions is observed rather than asserted about a double.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import grpc
import pytest

from cogniverse_agents.coding_agent import CodingAgent, CodingDeps, CodingInput
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.sandbox_manager import SandboxManager
from cogniverse_runtime.sandbox_pool import SandboxSessionPool
from tests.utils.local_openshell import serve_local_gateway

pytestmark = pytest.mark.integration

FAILING_CODE = "import sys\nprint('controlled failure', file=sys.stderr)\nsys.exit(7)"


def _config_manager() -> ConfigManager:
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


class _LMHandler(BaseHTTPRequestHandler):
    """Chat-completions endpoint with fixed answers for each DSPy signature."""

    def log_message(self, *_args):
        return

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = json.dumps(body["messages"])
        if "is_successful" in prompt:
            output = {
                "reasoning": "Nonzero exit.",
                "is_successful": False,
                "feedback": "Exit 7 is a failure.",
            }
        elif "test_command" in prompt:
            output = {
                "reasoning": "Emit the requested failure.",
                "code": FAILING_CODE,
                "test_command": "python solution.py",
            }
        else:
            output = {"reasoning": "Run it.", "plan": "Run the given program."}
        data = json.dumps(
            {
                "id": "local",
                "object": "chat.completion",
                "created": 1,
                "model": "local-coding-fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(output),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def gateway(tmp_path):
    with serve_local_gateway(tmp_path / "sandboxes") as (served, endpoint):
        (tmp_path / "sandboxes").mkdir(exist_ok=True)
        served.endpoint = endpoint
        yield served


@pytest.fixture
def sandbox_manager(gateway, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", gateway.endpoint)
    monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path / "openshell-config"))
    manager = SandboxManager(policy="required")
    assert manager.available is True
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def coding_lm():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LMHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield dspy.LM(
            "openai/local-coding-fixture",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local",
            cache=False,
            num_retries=0,
            timeout=20,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def _agent(manager, tenant: str) -> CodingAgent:
    return CodingAgent(
        CodingDeps(tenant_id=tenant),
        sandbox_manager=manager,
        config_manager=_config_manager(),
    )


@pytest.mark.asyncio
async def test_concurrent_tasks_run_the_file_they_wrote(sandbox_manager, gateway):
    """Two tasks interleaved between write and run keep their own sandbox."""
    after_write = asyncio.Barrier(2)
    names: list[str] = []

    async def run(tenant: str, value: int) -> dict:
        agent = _agent(sandbox_manager, tenant)
        async with sandbox_manager.task_session("coding_agent", tenant) as session:
            names.append(session.session_name)
            direct_exec = session.exec

            async def interleaved(command, timeout_seconds):
                result = await direct_exec(command, timeout_seconds)
                if timeout_seconds == 30:
                    await after_write.wait()
                return result

            session.exec = interleaved
            result = await agent._execute_in_sandbox(
                "/tmp/coding_shared/solution.py",
                f"print({value})",
                "",
                "python",
                session,
            )
            session.exec = direct_exec
            read_back = await session.exec(
                ["sh", "-c", "cat /tmp/coding_shared/solution.py"], 10
            )
            assert read_back == {
                "stdout": f"print({value})\n",
                "stderr": "",
                "exit_code": 0,
            }
            return result

    first, second = await asyncio.gather(
        run("prodfixagents:alpha", 11), run("prodfixagents:beta", 22)
    )
    assert first == {
        "stdout": "11\n",
        "stderr": "",
        "exit_code": 0,
        "command": "python /tmp/coding_shared/solution.py",
        "success": True,
    }
    assert second["stdout"] == "22\n"
    assert second["exit_code"] == 0
    assert sorted(names) == ["sandbox-1", "sandbox-2"]
    assert sorted(gateway.deleted) == ["sandbox-1", "sandbox-2"]
    assert gateway.live == {}


@pytest.mark.asyncio
async def test_next_task_cannot_see_prior_task_files(sandbox_manager, gateway):
    """Each task gets an empty filesystem."""
    names: list[str] = []
    for tenant in [
        "prodfixagents:alpha",
        "prodfixagents:alpha",
        "prodfixagents:beta",
    ]:
        async with sandbox_manager.task_session("coding_agent", tenant) as session:
            names.append(session.session_name)
            probe = await session.exec(
                [
                    "sh",
                    "-c",
                    "test ! -e /tmp/task-secret "
                    "&& printf secret > /tmp/task-secret "
                    "&& echo fresh",
                ],
                10,
            )
            assert probe == {"stdout": "fresh\n", "stderr": "", "exit_code": 0}
            listing = await session.exec(["sh", "-c", "ls /tmp"], 10)
            assert listing == {
                "stdout": "task-secret\n",
                "stderr": "",
                "exit_code": 0,
            }
    assert names == ["sandbox-1", "sandbox-2", "sandbox-3"]
    assert gateway.deleted == ["sandbox-1", "sandbox-2", "sandbox-3"]
    assert gateway.live == {}


@pytest.mark.asyncio
async def test_write_failure_stops_the_run(sandbox_manager, gateway):
    """A write that cannot land never executes a stale or absent file."""
    agent = _agent(sandbox_manager, "prodfixagents:write")
    async with sandbox_manager.task_session(
        "coding_agent", "prodfixagents:write"
    ) as session:
        seeded = await session.exec(
            [
                "sh",
                "-c",
                "mkdir -p /tmp/coding_blocked "
                "&& printf \"print('stale')\" > /tmp/coding_blocked/solution.py "
                "&& chmod 400 /tmp/coding_blocked/solution.py",
            ],
            10,
        )
        assert seeded["exit_code"] == 0
        before = len(gateway.execs)
        result = await agent._execute_in_sandbox(
            "/tmp/coding_blocked/solution.py",
            "print('must not run')",
            "",
            "python",
            session,
        )
        assert result["success"] is False
        assert result["exit_code"] == 2
        assert result["stdout"] == ""
        assert result["stderr"].endswith("Permission denied\n")
        assert len(gateway.execs) - before == 1
        unchanged = await session.exec(
            ["sh", "-c", "cat /tmp/coding_blocked/solution.py"], 10
        )
        assert unchanged["stdout"] == "print('stale')"


@pytest.mark.asyncio
async def test_readiness_failure_deletes_the_sandbox_it_created(gateway):
    """A sandbox whose readiness check fails is deleted, not orphaned."""
    from openshell import SandboxClient

    client = SandboxClient(endpoint=gateway.endpoint, timeout=5)
    try:
        failing = SandboxSessionPool(client)
        gateway.readiness_error = grpc.StatusCode.UNAVAILABLE
        for _ in range(3):
            with pytest.raises(grpc.RpcError) as raised:
                with failing.task_session():
                    pytest.fail("session yielded")
            assert raised.value.code() is grpc.StatusCode.UNAVAILABLE
        assert gateway.created == ["sandbox-1", "sandbox-2", "sandbox-3"]
        assert gateway.deleted == ["sandbox-1", "sandbox-2", "sandbox-3"]
        assert gateway.live == {}
        assert failing.stats() == {"max_pool_size": 8, "task_sessions": 0}

        gateway.readiness_error = None
        recovered = SandboxSessionPool(client)
        with recovered.task_session() as session:
            result = session.exec(["echo", "recovered"], timeout_seconds=5)
        assert (result.exit_code, result.stdout, result.stderr) == (
            0,
            "recovered\n",
            "",
        )
        recovered.close_all()
        failing.close_all()
        assert gateway.live == {}
    finally:
        client.close()


@pytest.mark.asyncio
async def test_gateway_exec_outage_still_destroys_the_task_sandbox(
    sandbox_manager, gateway
):
    """A boundary failure mid-task raises and leaves no sandbox behind."""
    with pytest.raises(grpc.RpcError) as raised:
        async with sandbox_manager.task_session(
            "coding_agent", "prodfixagents:fault"
        ) as session:
            gateway.exec_error = grpc.StatusCode.UNAVAILABLE
            await session.exec(["echo", "hi"], 5)
    assert raised.value.code() is grpc.StatusCode.UNAVAILABLE
    assert gateway.created == ["sandbox-1"]
    assert gateway.deleted == ["sandbox-1"]
    assert gateway.live == {}


@pytest.mark.asyncio
async def test_failing_coding_task_reports_failure_not_completion(
    sandbox_manager, gateway, coding_lm
):
    """A task whose program exits nonzero is reported as a failure."""
    agent = _agent(sandbox_manager, "prodfixagents:code")
    with dspy.context(lm=coding_lm, adapter=dspy.JSONAdapter()):
        output = await agent.process(
            CodingInput(
                task="Run the program.",
                tenant_id="prodfixagents:code",
                max_iterations=2,
            )
        )
    assert output.success is False
    assert output.iterations_used == 2
    assert [
        (r["exit_code"], r["stdout"], r["stderr"]) for r in output.execution_results
    ] == [(7, "", "controlled failure\n")] * 2
    assert output.error == (
        "Coding task failed after 2 iteration(s): Exit code: 7\n"
        "stderr: controlled failure\n\n"
        "Feedback: Exit 7 is a failure."
    )
    assert output.summary == output.error
    assert gateway.created == ["sandbox-1"]
    assert gateway.deleted == ["sandbox-1"]
    assert gateway.live == {}


@pytest.mark.asyncio
async def test_concurrent_tasks_do_not_exceed_the_sandbox_cap(
    gateway, tmp_path, monkeypatch
):
    """Container creation is bounded by the pool cap, not by request load."""
    from cogniverse_runtime.sandbox_pool import SandboxCapacityError

    monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", gateway.endpoint)
    monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path / "openshell-config"))
    monkeypatch.setenv("COGNIVERSE_SANDBOX_POOL_SIZE", "3")
    manager = SandboxManager(policy="required")
    assert manager.available is True
    attempted = asyncio.Barrier(12)
    live = {"now": 0, "peak": 0}

    async def task(index: int) -> str:
        try:
            async with manager.task_session("coding_agent", f"prodfixagents:t{index}"):
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
                # No lease is released before every task has attempted, so a
                # refusal cannot be an artefact of fast turnover.
                await attempted.wait()
                live["now"] -= 1
                return "leased"
        except SandboxCapacityError as exc:
            assert str(exc) == "Sandbox task capacity reached: 3 sessions in use"
            await attempted.wait()
            return "refused"

    try:
        outcomes = await asyncio.gather(*[task(i) for i in range(12)])
    finally:
        manager.close()

    assert sorted(outcomes) == ["leased"] * 3 + ["refused"] * 9
    assert live["peak"] == 3
    assert gateway.created == ["sandbox-1", "sandbox-2", "sandbox-3"]
    assert sorted(gateway.deleted) == ["sandbox-1", "sandbox-2", "sandbox-3"]
    assert gateway.live == {}


@pytest.mark.asyncio
async def test_runtime_shutdown_deletes_the_live_task_sandbox(sandbox_manager, gateway):
    """Shutdown during a task reclaims its container instead of orphaning it."""
    async with sandbox_manager.task_session("coding_agent", "prodfixagents:shut"):
        await asyncio.to_thread(sandbox_manager.close)
        assert gateway.deleted == ["sandbox-1"]
        assert gateway.live == {}
    assert gateway.created == ["sandbox-1"]
    assert gateway.deleted == ["sandbox-1"]
    assert gateway.live == {}


@pytest.mark.asyncio
async def test_gateway_reconnect_deletes_the_live_task_sandbox(
    sandbox_manager, gateway
):
    """A cert-rotation reconnect reclaims the task's container on the old client."""
    async with sandbox_manager.task_session("coding_agent", "prodfixagents:rotate"):
        assert await asyncio.to_thread(sandbox_manager.reconnect) is True
        assert gateway.deleted == ["sandbox-1"]
        assert gateway.live == {}
    assert gateway.created == ["sandbox-1"]
    assert gateway.deleted == ["sandbox-1"]
    assert gateway.live == {}
