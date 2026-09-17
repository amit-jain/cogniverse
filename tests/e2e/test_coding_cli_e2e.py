"""E2E tests for the Coding Agent CLI — real runtime, real Vespa, real agents.

Tests the CLI implementation functions directly (not via subprocess) against
the live k3d runtime at localhost:33000. Assertions verify real data flow:
- `index_files` ingests into real Vespa
- `stream_coding_response` consumes real A2A SSE
- REPL session state survives multi-turn round-trips
"""

import inspect
import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import pytest
from cogniverse_cli.code import CodingSession
from cogniverse_cli.index import collect_files, index_files
from cogniverse_cli.streaming import (
    CodingStreamError,
    _build_a2a_request,
    _parse_coding_result,
    stream_coding_response,
)

from cogniverse_foundation.config.routed_lm import UpstreamUnavailable
from cogniverse_runtime.sandbox_pool import SandboxSessionPool
from tests.e2e.conftest import KUBECTL_CONTEXT, RUNTIME, TENANT_ID

SEARCH_AGENT_URL = f"{RUNTIME}/agents/search_agent/process"
CODING_AGENT_URL = f"{RUNTIME}/agents/coding_agent/process"

SANDBOX_WAIT_READY_S = (
    inspect.signature(SandboxSessionPool.__init__)
    .parameters["wait_ready_timeout_s"]
    .default
)
"""Budget each task's sandbox gets to become ready. Every task leases a fresh
sandbox, and a cold start measures ~168 s on the gateway host
(``SandboxSessionPool`` docstring), so the probe owes the whole budget."""

SANDBOX_PROBE_EXEC_S = 60
SANDBOX_PROBE_DEADLINE_S = SANDBOX_WAIT_READY_S + SANDBOX_PROBE_EXEC_S + 60
"""The probe's lease wait and its exec, plus a minute for the pod's
interpreter start, session creation and teardown."""


def _assert_coding_output_shape(result):
    assert set(result) == {
        "plan",
        "code_changes",
        "execution_results",
        "summary",
        "iterations_used",
        "files_modified",
        "rlm_synthesis",
        "rlm_telemetry",
        "pending_tool_calls",
        "continuation_state",
        "success",
        "error",
    }, result
    assert result["pending_tool_calls"] == []
    assert result["continuation_state"] == {}
    assert result["success"] is True
    assert result["error"] is None


def _run_prerequisite_command(
    command: list[str], *, timeout: int
) -> subprocess.CompletedProcess:
    command_text = " ".join(command)
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        pytest.fail(
            "coding sandbox prerequisite executable is unavailable; "
            f"command={command_text!r}; error={exc!r}",
            pytrace=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "coding sandbox prerequisite command timed out; "
            f"command={command_text!r}; timeout={exc.timeout}s; "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            pytrace=False,
        )


@pytest.fixture()
def runtime_sandbox_ready() -> None:
    """Provision OpenShell, sync the cluster mounts, and prove runtime exec.

    The runtime mounts the OpenShell files with subPath, so if the gateway
    is recreated the cluster needs a re-sync and the runtime pod needs a
    rollout to pick up the refreshed files.
    """
    context_command = ["kubectl", "config", "current-context"]
    context = _run_prerequisite_command(context_command, timeout=10)
    assert context.returncode == 0, (
        "could not resolve kubectl context for coding sandbox setup; "
        f"command={' '.join(context_command)!r}; returncode={context.returncode}; "
        f"stdout={context.stdout!r}; stderr={context.stderr!r}"
    )
    assert context.stdout.strip() == KUBECTL_CONTEXT, (
        "coding sandbox setup would target the wrong cluster; "
        f"expected_context={KUBECTL_CONTEXT!r}; "
        f"actual_context={context.stdout.strip()!r}; "
        f"command={' '.join(context_command)!r}"
    )

    from cogniverse_cli.sandbox import ensure_sandbox_ready

    try:
        ready = ensure_sandbox_ready(kube_context=KUBECTL_CONTEXT)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        pytest.fail(
            "OpenShell host gateway setup raised before the runtime probe; "
            f"kubectl_context={KUBECTL_CONTEXT!r}; "
            f"operation='ensure_sandbox_ready'; error={exc!r}",
            pytrace=False,
        )
    assert ready, (
        "OpenShell host gateway setup returned false; "
        f"kubectl_context={KUBECTL_CONTEXT!r}; "
        "operation='ensure_sandbox_ready'; expected=True"
    )

    probe_code = (
        "import asyncio, json\n"
        "from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy\n"
        "mgr = SandboxManager(policy=SandboxPolicy.REQUIRED)\n"
        "async def probe():\n"
        "    async with mgr.task_session('coding_agent', 'e2e:coding') as s:\n"
        "        return await s.exec(\n"
        "            ['python3', '-c', \"print('coding-sandbox-ready')\"], "
        f"{SANDBOX_PROBE_EXEC_S}\n"
        "        )\n"
        "out = asyncio.run(probe())\n"
        "print('__SANDBOX_PROBE__' + json.dumps(out))\n"
    )
    probe_command = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "-n",
        "cogniverse",
        "exec",
        "deploy/cogniverse-runtime",
        "-c",
        "runtime",
        "--",
        "python3",
        "-c",
        probe_code,
    ]
    probe = _run_prerequisite_command(probe_command, timeout=SANDBOX_PROBE_DEADLINE_S)
    assert probe.returncode == 0, (
        "runtime pod could not execute the OpenShell prerequisite probe; "
        f"command={' '.join(probe_command)!r}; returncode={probe.returncode}; "
        f"stdout={probe.stdout!r}; stderr={probe.stderr!r}"
    )
    marker = next(
        (
            line.removeprefix("__SANDBOX_PROBE__")
            for line in probe.stdout.splitlines()
            if line.startswith("__SANDBOX_PROBE__")
        ),
        None,
    )
    assert marker is not None, (
        "runtime sandbox probe did not emit its result marker; "
        f"command={' '.join(probe_command)!r}; "
        f"stdout={probe.stdout!r}; stderr={probe.stderr!r}"
    )
    try:
        payload = json.loads(marker)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "runtime sandbox probe emitted malformed JSON; "
            f"marker={marker!r}; stdout={probe.stdout!r}; error={exc!r}",
            pytrace=False,
        )
    assert payload is not None, (
        "runtime SandboxManager returned no execution result; "
        f"kubectl_context={KUBECTL_CONTEXT!r}; payload={payload!r}"
    )
    assert payload.get("exit_code") == 0, (
        "runtime OpenShell prerequisite exec failed; "
        f"kubectl_context={KUBECTL_CONTEXT!r}; payload={payload!r}"
    )
    assert payload.get("stdout") == "coding-sandbox-ready\n", payload
    assert payload.get("stderr") == "", payload


@pytest.mark.e2e
class TestIndexCommand:
    """cogniverse index — real file collection + Vespa ingestion."""

    def test_collect_files_filters_code_by_extension(self):
        """collect_files returns .py/.ts/.go files and skips non-code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("print('hi')")
            (root / "utils.ts").write_text("export const x = 1;")
            (root / "server.go").write_text("package main")
            (root / "readme.md").write_text("# docs")
            (root / "data.csv").write_text("a,b\n1,2")

            files = collect_files(root, "code")
            names = {f.name for f in files}

            assert names == {"main.py", "utils.ts", "server.go"}

    def test_collect_files_respects_gitignore_patterns(self):
        """Files in .venv, __pycache__, node_modules are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "app.py").write_text("x = 1")

            venv = root / ".venv" / "lib" / "python3.12"
            venv.mkdir(parents=True)
            (venv / "site.py").write_text("lib")

            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "app.cpython-312.pyc").write_text("bytecode")

            node = root / "node_modules" / "pkg"
            node.mkdir(parents=True)
            (node / "index.js").write_text("mod")

            files = collect_files(root, "code")
            paths = [str(f) for f in files]

            assert any("app.py" in p for p in paths)
            assert not any(".venv" in p for p in paths)
            assert not any("__pycache__" in p for p in paths)
            assert not any("node_modules" in p for p in paths)

    def test_index_files_uploads_to_runtime_ingestion(self):
        """index_files POSTs to /ingestion/upload and returns real counts.

        Uses the configured document profile (docs type) because the coding
        profile requires a LateOn-Code encoder that is not registered yet. This
        test still exercises the full path: walk → upload → runtime → Vespa.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "readme.md").write_text(
                "# Test Project\n\nThis is a test readme for integration testing."
            )
            (root / "guide.txt").write_text(
                "A guide to the test project with some content."
            )

            summary = index_files(
                root=root,
                content_type="docs",
                tenant_id=TENANT_ID,
                runtime_url=RUNTIME,
            )

        assert summary["files_found"] == 2, (
            f"Expected 2 files, got {summary['files_found']}"
        )

    def test_index_files_empty_directory_returns_zero(self):
        """Empty directory returns zero counts without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = index_files(
                root=Path(tmpdir),
                content_type="code",
                tenant_id=TENANT_ID,
                runtime_url=RUNTIME,
            )
        assert summary["files_found"] == 0
        assert summary["files_indexed"] == 0


@pytest.mark.e2e
class TestA2AStreamingClient:
    """stream_coding_response — real A2A SSE against the live runtime."""

    def test_a2a_request_builder_produces_valid_jsonrpc(self):
        """Request payload has jsonrpc/method/params fields the runtime expects."""
        req = _build_a2a_request(
            "find videos about nature",
            agent_name="search_agent",
            tenant_id=TENANT_ID,
        )
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "message/stream"
        assert "id" in req
        assert req["params"]["metadata"]["agent_name"] == "search_agent"
        assert req["params"]["metadata"]["tenant_id"] == TENANT_ID
        assert req["params"]["metadata"]["stream"] is True
        msg = req["params"]["message"]
        assert msg["kind"] == "message"
        assert msg["role"] == "user"
        assert msg["parts"][0]["kind"] == "text"

    def test_stream_to_search_agent_returns_parsed_result(self):
        """Stream a real search via A2A SSE and verify result structure.

        Uses search_agent because this streaming test does not need the
        coding sandbox. It still exercises the full
        CLI streaming path: build request → POST /a2a → consume SSE → parse.
        """
        result = stream_coding_response(
            query="find videos about outdoor nature scenes",
            agent_name="search_agent",
            tenant_id=TENANT_ID,
            runtime_url=RUNTIME,
        )

        assert result is not None, "Streaming should produce a result"
        assert result.raw, "Result should have raw event data"


@pytest.mark.e2e
class TestCodingAgentDispatch:
    """Verify coding agent dispatch path — request reaches CodingAgent."""

    def test_coding_agent_is_registered_with_coding_capability(self):
        """The coding_agent is registered and advertises the coding capability."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{RUNTIME}/agents/coding_agent")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "coding_agent"
        assert data["capabilities"] == ["coding", "code_generation", "code_search"]

    def test_coding_agent_full_execution_with_sandbox(self, runtime_sandbox_ready):
        """Full plan → code → sandbox execute → evaluate loop.

        Requires the OpenShell gateway running on the host with the runtime
        pod configured to reach it via host.docker.internal. The sandbox
        actually executes the generated code and returns stdout/stderr/exit.
        """
        query = "write a python function that returns the string hello world"
        with httpx.Client(timeout=400.0) as client:
            resp = client.post(
                CODING_AGENT_URL,
                json={
                    "agent_name": "coding_agent",
                    "query": query,
                    "context": {
                        "tenant_id": TENANT_ID,
                        "max_iterations": 1,
                    },
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200, (
            f"Coding agent failed: {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        assert set(data) == {"status", "agent", "message", "result", "answer"}, data
        # The dispatch envelope carries the canonical answer text; for this
        # agent it is the output's summary field.
        assert data["answer"] == data["result"]["summary"]
        assert data["status"] == "success"
        assert data["agent"] == "coding_agent"
        assert data["message"] == f"Coding task complete for '{query}'"

        # CodingOutput.model_dump(): the plan and generated code are LM free
        # text; everything else is fixed by max_iterations=1 and the sandbox
        # exec contract, so it is pinned exactly.
        result = data["result"]
        _assert_coding_output_shape(result)
        assert result["plan"], "Plan should not be empty"
        assert result["iterations_used"] == 1, result
        assert len(result["code_changes"]) == 1, result["code_changes"]
        change = result["code_changes"][0]
        assert set(change) == {"file_path", "content", "change_type"}, change
        assert change["change_type"] == "create", change
        assert change["file_path"].endswith("/solution.py"), change
        assert change["content"], "generated code must not be empty"
        assert result["files_modified"] == [change["file_path"]], result
        assert result["rlm_synthesis"] is None, result
        assert result["rlm_telemetry"] is None, result

        exec_results = result["execution_results"]
        assert len(exec_results) == 1, exec_results
        first_exec = exec_results[0]
        assert set(first_exec) == {
            "stdout",
            "stderr",
            "exit_code",
            "command",
            "success",
        }, first_exec
        assert first_exec["exit_code"] == 0, (
            "coding agent sandbox execution failed; "
            f"stdout={first_exec['stdout']!r}; "
            f"stderr={first_exec['stderr']!r}; result={first_exec!r}"
        )
        assert first_exec["success"] is True, first_exec
        assert first_exec["command"] == f"python {change['file_path']}", first_exec
        assert result["summary"] == (
            "Completed coding task in 1 iteration(s). Generated 1 file(s). "
            "Final execution: exit_code=0"
        ), result["summary"]


@pytest.mark.e2e
class TestCodingSession:
    """CodingSession maintains state across turns with real HTTP."""

    def test_session_apply_writes_real_files(self):
        """Session.apply() writes code_changes to actual filesystem."""
        from cogniverse_cli.streaming import CodingResult

        session = CodingSession(
            tenant_id=TENANT_ID,
            language="python",
            max_iterations=1,
            codebase_path="",
            runtime_url=RUNTIME,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "subdir" / "hello.py"
            session.last_result = CodingResult(
                code_changes=[
                    {
                        "file_path": str(target_file),
                        "content": "def hello():\n    return 'world'\n",
                        "change_type": "new",
                    }
                ],
            )

            count = session.apply()

            assert count == 1
            assert target_file.exists()
            assert target_file.read_text() == "def hello():\n    return 'world'\n"

    def test_session_apply_deletes_existing_file(self):
        from cogniverse_cli.streaming import CodingResult

        session = CodingSession(
            tenant_id=TENANT_ID,
            language="python",
            max_iterations=1,
            codebase_path="",
            runtime_url=RUNTIME,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "old.py"
            target.write_text("obsolete")

            session.last_result = CodingResult(
                code_changes=[
                    {
                        "file_path": str(target),
                        "content": "",
                        "change_type": "delete",
                    }
                ],
            )
            session.apply()

            assert not target.exists()

    def test_parse_coding_result_handles_nested_result(self):
        """Parses the double-nested {result: {result: {...}}} shape."""
        data = {
            "status": "success",
            "agent": "coding_agent",
            "result": {
                "plan": "1. Do thing",
                "code_changes": [
                    {"file_path": "x.py", "content": "pass", "change_type": "new"},
                ],
                "summary": "Did it",
                "iterations_used": 1,
                "files_modified": ["x.py"],
                "execution_results": [],
            },
        }
        result = _parse_coding_result(data)
        assert result.plan == "1. Do thing"
        assert len(result.code_changes) == 1
        assert result.code_changes[0]["file_path"] == "x.py"
        assert result.summary == "Did it"
        assert result.iterations_used == 1


# The program the failing-turn tests ask for: a fixed marker on stderr and a
# fixed nonzero status, so the reported failure can be pinned to what the
# sandbox actually produced.
FAILING_EXIT_CODE = 7
FAILING_STDERR_MARKER = "cogniverse-e2e-sandbox-failure"
FAILING_QUERY = (
    "write a python program whose only statements print the exact text "
    f"{FAILING_STDERR_MARKER} to stderr and then exit the process with "
    f"status code {FAILING_EXIT_CODE}"
)


def _coding_failure_error(iterations: int, execution: dict) -> str:
    """The prefix the agent's failure text carries for this execution.

    Everything up to the evaluator's own feedback is fixed by the exit code
    and the stderr the sandbox returned, so it is pinned against the recorded
    execution rather than restated.
    """
    return (
        f"Coding task failed after {iterations} iteration(s): "
        f"Exit code: {execution['exit_code']}\n"
        f"stderr: {execution['stderr']}\n"
        f"Feedback: "
    )


@pytest.mark.e2e
class TestNonzeroSandboxExitIsAFailure:
    """A program the sandbox ran and rejected is not completed work."""

    def test_nonzero_sandbox_exit_is_reported_as_failure(self, runtime_sandbox_ready):
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(
                CODING_AGENT_URL,
                json={
                    "agent_name": "coding_agent",
                    "query": FAILING_QUERY,
                    "context": {
                        "tenant_id": TENANT_ID,
                        "max_iterations": 1,
                    },
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        # The failure envelope carries the error instead of a message, and
        # nothing renders as an answer.
        assert set(data) == {"status", "agent", "error", "result"}, data
        assert data["status"] == "error", data
        assert data["agent"] == "coding_agent", data

        result = data["result"]
        assert set(result) == {
            "plan",
            "code_changes",
            "execution_results",
            "summary",
            "iterations_used",
            "files_modified",
            "rlm_synthesis",
            "rlm_telemetry",
            "pending_tool_calls",
            "continuation_state",
            "success",
            "error",
        }, result
        assert result["success"] is False, result
        assert result["pending_tool_calls"] == [], result
        assert result["iterations_used"] == 1, result

        exec_results = result["execution_results"]
        assert len(exec_results) == 1, exec_results
        execution = exec_results[0]
        assert set(execution) == {
            "stdout",
            "stderr",
            "exit_code",
            "command",
            "success",
        }, execution
        assert execution["exit_code"] == FAILING_EXIT_CODE, execution
        assert execution["success"] is False, execution
        assert FAILING_STDERR_MARKER in execution["stderr"], execution

        expected_prefix = _coding_failure_error(1, execution)
        assert result["error"].startswith(expected_prefix), result["error"]
        # The failure text replaces the summary; the success template the
        # completing turn produces must not be what this run reports.
        assert result["summary"] == result["error"], result
        assert result["summary"] != (
            f"Completed coding task in 1 iteration(s). "
            f"Generated {len(result['files_modified'])} file(s). "
            f"Final execution: exit_code={execution['exit_code']}"
        ), result["summary"]
        assert data["error"] == result["error"], data


def _a2a_events(request_body: dict) -> list[dict]:
    """Every SSE payload the runtime emitted for one A2A request."""
    events: list[dict] = []
    with httpx.Client(timeout=900.0) as client:
        with client.stream(
            "POST",
            f"{RUNTIME}/a2a/",
            json=request_body,
            headers={"Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200, response.read()[:300]
            for line in response.iter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload:
                        events.append(json.loads(payload))
    return events


ROUTER_ENVOY_DEPLOYMENT = "cogniverse-semantic-router-envoy"


def _router_envoy_kubectl(*args: str) -> str:
    command = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "-n",
        "cogniverse",
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"{' '.join(command)!r} failed with exit {result.returncode}: "
        f"{result.stderr.strip()[:500]}"
    )
    return result.stdout


def _wait_for_router_envoy(*, serving: bool, deadline_s: float) -> None:
    """Block until the router Envoy the runtime's LM calls go through answers
    (``serving``) or refuses, judged by the session readiness gate's probe."""
    from tests.e2e.conftest import _required_e2e_semantic_router_ready

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        ready, _detail = _required_e2e_semantic_router_ready()
        if ready is serving:
            return
        time.sleep(3)
    pytest.fail(
        f"{ROUTER_ENVOY_DEPLOYMENT} was not "
        f"{'serving' if serving else 'refusing'} within {deadline_s:.0f}s",
        pytrace=False,
    )


@pytest.fixture
def router_envoy_scaled_to_zero():
    """Take the Envoy in front of the semantic router down for one test.

    Every runtime LM call goes through it (``SEMANTIC_ROUTER_URL``), so with no
    replica the coding turn's first LM call is refused. The restore runs on
    every outcome and waits for the readiness gate's probe to answer again.
    """
    declared = int(
        _router_envoy_kubectl(
            "get",
            "deployment",
            ROUTER_ENVOY_DEPLOYMENT,
            "-o",
            "jsonpath={.spec.replicas}",
        ).strip()
    )
    _router_envoy_kubectl(
        "scale", "deployment", ROUTER_ENVOY_DEPLOYMENT, "--replicas=0"
    )
    try:
        _wait_for_router_envoy(serving=False, deadline_s=120.0)
        yield
    finally:
        _router_envoy_kubectl(
            "scale", "deployment", ROUTER_ENVOY_DEPLOYMENT, f"--replicas={declared}"
        )
        _wait_for_router_envoy(serving=True, deadline_s=300.0)


@pytest.mark.e2e
class TestFailedCodingTurnIsTerminalFailure:
    """A coding turn whose LM is unreachable terminates as a failed A2A task.

    The runtime's LM calls go through the router Envoy, which the fixture
    scales to zero, so the turn's planning call is refused before any
    generation or sandbox work: the failure does not depend on what an LM
    writes. Each request carries its own token so no cached completion from an
    earlier run can answer it.
    """

    def test_the_stream_ends_in_a_failed_task_and_the_cli_raises(
        self, router_envoy_scaled_to_zero
    ):
        expected_error = {
            "type": "error",
            "agent": "coding_agent",
            "error_type": UpstreamUnavailable.__name__,
            "message": (
                f"CodingAgent streaming failed with {UpstreamUnavailable.__name__}. "
                "See server logs for detail."
            ),
        }
        request_body = _build_a2a_request(
            f"{FAILING_QUERY} (request {uuid.uuid4().hex})",
            agent_name="coding_agent",
            tenant_id=TENANT_ID,
        )
        events = _a2a_events(request_body)

        finals = [
            event
            for event in events
            if event.get("result", {}).get("final") is True
            or event.get("result", {}).get("status", {}).get("state")
            in ("failed", "input-required", "canceled")
        ]
        assert len(finals) == 1, events
        status = finals[0]["result"]["status"]
        # The terminal state is the failure, not the state a turn awaiting
        # more input ends in.
        assert status["state"] == "failed", finals[0]
        parts = status["message"]["parts"]
        assert [part["kind"] for part in parts] == ["text"], parts
        assert json.loads(parts[0]["text"]) == expected_error, parts[0]["text"]

        # The CLI consumes that same terminal event and refuses to present it
        # as a result.
        with pytest.raises(CodingStreamError) as raised:
            stream_coding_response(
                query=f"{FAILING_QUERY} (request {uuid.uuid4().hex})",
                agent_name="coding_agent",
                tenant_id=TENANT_ID,
                runtime_url=RUNTIME,
            )
        assert str(raised.value) == (
            f"coding_agent ({UpstreamUnavailable.__name__}): "
            f"{expected_error['message']}"
        ), str(raised.value)
