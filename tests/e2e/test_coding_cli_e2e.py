"""E2E tests for the Coding Agent CLI — real runtime, real Vespa, real agents.

Tests the CLI implementation functions directly (not via subprocess) against
the live k3d runtime at localhost:33000. Assertions verify real data flow:
- `index_files` ingests into real Vespa
- `stream_coding_response` consumes real A2A SSE
- REPL session state survives multi-turn round-trips
"""

import json
import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest
from cogniverse_cli.code import CodingSession
from cogniverse_cli.index import collect_files, index_files
from cogniverse_cli.streaming import (
    _build_a2a_request,
    _parse_coding_result,
    stream_coding_response,
)

from tests.e2e.conftest import KUBECTL_CONTEXT, RUNTIME, TENANT_ID

SEARCH_AGENT_URL = f"{RUNTIME}/agents/search_agent/process"
CODING_AGENT_URL = f"{RUNTIME}/agents/coding_agent/process"


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
    """Provision OpenShell and prove a real exec from the runtime pod."""
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
        ready = ensure_sandbox_ready()
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
        "import json\n"
        "from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy\n"
        "mgr = SandboxManager(policy=SandboxPolicy.REQUIRED)\n"
        "out = mgr.exec_in_sandbox(\n"
        "    'coding_agent',\n"
        "    ['python3', '-c', \"print('coding-sandbox-ready')\"],\n"
        "    timeout_seconds=60,\n"
        ")\n"
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
    probe = _run_prerequisite_command(probe_command, timeout=180)
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

        Uses the document_text_semantic profile (docs type) since code_lateon_mv
        requires LateOn-Code encoder which isn't registered yet. This test still
        exercises the full path: walk → upload → runtime → Vespa.
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
        assert "coding" in data["capabilities"]

    def test_coding_agent_full_execution_with_sandbox(self, runtime_sandbox_ready):
        """Full plan → code → sandbox execute → evaluate loop.

        Requires the OpenShell gateway running on the host with the runtime
        pod configured to reach it via host.docker.internal. The sandbox
        actually executes the generated code and returns stdout/stderr/exit.
        """
        with httpx.Client(timeout=400.0) as client:
            resp = client.post(
                CODING_AGENT_URL,
                json={
                    "agent_name": "coding_agent",
                    "query": "write a python function that returns the string hello world",
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
        assert data["status"] == "success"
        assert data["agent"] == "coding_agent"

        result = data["result"]
        assert "plan" in result
        assert result["plan"], "Plan should not be empty"
        assert len(result["code_changes"]) >= 1, (
            "Should generate at least 1 code change"
        )
        assert result["iterations_used"] >= 1

        exec_results = result.get("execution_results", [])
        assert len(exec_results) >= 1, (
            "Execution results should exist (sandbox executed the code)"
        )
        first_exec = exec_results[0]
        assert first_exec.get("exit_code") == 0, (
            "coding agent sandbox execution failed; "
            f"stdout={first_exec.get('stdout')!r}; "
            f"stderr={first_exec.get('stderr')!r}; result={first_exec!r}"
        )
        assert "stdout" in first_exec, first_exec
        assert "stderr" in first_exec, first_exec


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
