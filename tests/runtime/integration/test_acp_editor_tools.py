"""Workspace tools exchange JSON-RPC with an editor over operating-system pipes."""

import asyncio
import contextlib
import json
import os
from pathlib import Path

import pytest

from cogniverse_runtime.acp import tools
from cogniverse_runtime.acp.server import ACPError, ACPServer, ClientConnection, serve
from cogniverse_runtime.config_loader import ConfigLoader
from tests.runtime.integration.test_acp_connection import (
    PipeTurnAgent,
    PipeTurnDeps,
    PipeTurnInput,
    PipeTurnOutput,
    running,
)
from tests.runtime.integration.test_acp_connection import (
    dispatcher as connection_dispatcher,
)

dispatcher = connection_dispatcher

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]


@contextlib.asynccontextmanager
async def editor_connection():
    incoming, editor_out = os.pipe()
    editor_in, outgoing = os.pipe()
    loop = asyncio.get_running_loop()
    readers = []
    transports = []
    for fd in (incoming, editor_in):
        reader = asyncio.StreamReader()
        transport, _ = await loop.connect_read_pipe(
            lambda reader=reader: asyncio.StreamReaderProtocol(reader),
            os.fdopen(fd, "rb", buffering=0),
        )
        readers.append(reader)
        transports.append(transport)

    async def write(message):
        os.write(outgoing, (json.dumps(message) + "\n").encode())

    conn = ClientConnection(readers[0], write)
    server = ACPServer(
        dispatcher_provider=lambda: None,
        agent_provider=lambda: "editor-tools",
        coding_agent_provider=lambda: "editor-tools",
        default_tenant="test:editor",
    )
    serving = asyncio.create_task(serve(server, conn))

    async def receive():
        return json.loads(await asyncio.wait_for(readers[1].readline(), 2))

    def reply(request, *, result=None, error=None):
        response = {"jsonrpc": "2.0", "id": request["id"]}
        response["error" if error else "result"] = error if error else result
        os.write(editor_out, (json.dumps(response) + "\n").encode())

    try:
        yield conn, receive, reply
    finally:
        os.close(editor_out)
        await asyncio.wait_for(serving, 2)
        os.close(outgoing)
        for transport in transports:
            transport.close()


@pytest.mark.asyncio
async def test_workspace_files_round_trip_and_concurrent_isolation(tmp_path):
    async with editor_connection() as (conn, receive, reply):

        async def editor():
            requests = []
            for _ in range(4):
                request = await receive()
                requests.append(request["params"])
                path = Path(request["params"]["path"])
                if request["method"] == "fs/write_text_file":
                    path.write_text(request["params"]["content"])
                    reply(request)
                else:
                    reply(request, result={"content": path.read_text()})
            return requests

        async def round_trip(index):
            root = tmp_path / str(index)
            root.mkdir()
            args = {"path": "answer.txt", "content": f"workspace {index}\n"}
            wrote = await tools.execute_tool_call(
                conn, str(index), "write_file", args, workspace_root=root
            )
            read = await tools.execute_tool_call(
                conn,
                str(index),
                "read_file",
                {"path": "answer.txt"},
                workspace_root=root,
            )
            assert wrote == f"wrote {root / 'answer.txt'}"
            assert read == f"workspace {index}\n"
            assert (root / "answer.txt").read_text() == f"workspace {index}\n"

        editor_task = asyncio.create_task(editor())
        try:
            await asyncio.gather(round_trip(1), round_trip(2))
            requests = await editor_task
            assert sorted((r["sessionId"], r["path"]) for r in requests) == [
                ("1", str(tmp_path / "1/answer.txt")),
                ("1", str(tmp_path / "1/answer.txt")),
                ("2", str(tmp_path / "2/answer.txt")),
                ("2", str(tmp_path / "2/answer.txt")),
            ]
        finally:
            editor_task.cancel()
            await asyncio.gather(editor_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", ["../outside.txt", "absolute", "symlink"])
async def test_workspace_escape_is_rejected_before_editor_request(tmp_path, escape):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n")
    (workspace / "link").symlink_to(tmp_path, target_is_directory=True)
    path = {
        "absolute": str(outside),
        "symlink": "link/outside.txt",
    }.get(escape, escape)
    async with editor_connection() as (conn, receive, reply):
        with pytest.raises(ValueError, match="outside workspace root"):
            await asyncio.wait_for(
                tools.execute_tool_call(
                    conn,
                    "workspace",
                    "write_file",
                    {"path": path, "content": "overwrite"},
                    workspace_root=workspace,
                ),
                timeout=0.5,
            )
        assert outside.read_text() == "keep\n"
        assert conn._next_id == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "allowed"),
    [
        ({"outcome": {"outcome": "selected", "optionId": "allow-once"}}, True),
        ({"outcome": {"outcome": "selected", "optionId": "allow-always"}}, True),
        ({"outcome": {"outcome": "selected", "optionId": "reject-once"}}, False),
        ({"outcome": {"outcome": "selected", "optionId": "allowlist-bypass"}}, False),
        ({"outcome": "selected", "optionId": "allow-once"}, False),
        ({"outcome": ["selected", "allow-once"]}, False),
        ({"outcome": {"outcome": "selected", "optionId": ["allow-once"]}}, False),
        (None, False),
    ],
)
async def test_permission_response_accepts_only_offered_allow_options(outcome, allowed):
    async with editor_connection() as (conn, receive, reply):
        pending = asyncio.create_task(
            conn.call(
                "session/request_permission", {"options": tools.permission_options()}
            )
        )
        request = await receive()
        reply(request, result=outcome)
        assert tools.outcome_allows(await pending) is allowed
        assert tools.outcome_allows(
            outcome,
            offered_options=[{"optionId": "allow-once", "kind": "allow_once"}],
        ) is (allowed and outcome["outcome"]["optionId"] == "allow-once")


@pytest.mark.asyncio
async def test_cancelled_terminal_release_has_reported_deadline(tmp_path, caplog):
    async with editor_connection() as (conn, receive, reply):
        pending = asyncio.create_task(
            tools.execute_tool_call(
                conn,
                "workspace",
                "run_command",
                {"command": "sleep 60"},
                workspace_root=tmp_path,
            )
        )
        try:
            request = await receive()
            assert request["params"] == {
                "sessionId": "workspace",
                "command": "sh",
                "args": ["-c", "sleep 60"],
                "cwd": str(tmp_path),
            }
            reply(request, result={"terminalId": "owned-terminal"})
            assert (await receive())["method"] == "terminal/wait_for_exit"
            pending.cancel()
            assert (await receive())["method"] == "terminal/release"
            done, _ = await asyncio.wait({pending}, timeout=0.5)
            assert done == {pending}
            assert pending.cancelled() is True
            assert (
                "terminal/release timed out after 0.1s for owned-terminal"
                in caplog.text
            )
            assert conn._pending == {}
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_editor_file_error_preserves_context(tmp_path):
    async with editor_connection() as (conn, receive, reply):
        pending = asyncio.create_task(
            tools.execute_tool_call(
                conn,
                "workspace",
                "read_file",
                {"path": "answer.txt"},
                workspace_root=tmp_path,
            )
        )
        request = await receive()
        reply(request, error={"code": -32000, "message": "EIO: answer.txt unreadable"})
        with pytest.raises(ACPError, match="EIO: answer.txt unreadable"):
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_release", [True, False])
async def test_terminal_executes_in_workspace_and_reports_release_failure(
    tmp_path, ack_release
):
    async with editor_connection() as (conn, receive, reply):
        pending = asyncio.create_task(
            tools.execute_tool_call(
                conn,
                "workspace",
                "run_command",
                {
                    "command": "printf 'terminal workspace\\n' > result.txt; cat result.txt"
                },
                workspace_root=tmp_path,
            )
        )
        request = await receive()
        params = request["params"]
        process = await asyncio.create_subprocess_exec(
            params["command"],
            *params["args"],
            cwd=params["cwd"],
            stdout=asyncio.subprocess.PIPE,
        )
        reply(request, result={"terminalId": "owned-command"})
        wait = await receive()
        assert wait["method"] == "terminal/wait_for_exit"
        output, _ = await process.communicate()
        reply(wait, result={"exitCode": process.returncode})
        request = await receive()
        assert request["method"] == "terminal/output"
        reply(request, result={"output": output.decode()})
        release = await receive()
        assert release["params"] == {
            "sessionId": "workspace",
            "terminalId": "owned-command",
        }
        if ack_release:
            reply(release)
            assert await pending == "exit_code=0\nterminal workspace\n"
        else:
            with pytest.raises(
                TimeoutError,
                match="terminal/release timed out after 0.1s for owned-command",
            ):
                await asyncio.wait_for(pending, 0.5)
        assert (tmp_path / "result.txt").read_text() == "terminal workspace\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        {"outcome": "selected", "optionId": "allow-once"},
        {"outcome": ["selected", "allow-once"]},
        {"outcome": {"outcome": "selected", "optionId": "allowlist-bypass"}},
        None,
    ],
)
async def test_malformed_permission_denies_tool_and_completes_turn(
    dispatcher, tmp_path, outcome
):
    async with running(dispatcher, tmp_path, caps={"fs": {"writeTextFile": True}}) as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        await ed.prompt(1, sid, "permission")
        request = await ed.until(
            lambda m: m.get("method") == "session/request_permission"
        )
        assert request["params"]["options"] == tools.permission_options()
        await ed.send({"jsonrpc": "2.0", "id": request["id"], "result": outcome})
        assert await ed.until(lambda m: m.get("id") == 1) == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"stopReason": "end_turn"},
        }
        assert [m["method"] for m in ed.messages if "method" in m] == [
            "session/update",
            "session/request_permission",
            "session/update",
            "session/update",
        ]
        assert ed.messages[2]["params"]["update"] == {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call_0",
            "status": "failed",
        }
        denied = json.dumps(
            [{"tool_call_id": "call_0", "content": "permission denied for write_file"}],
            sort_keys=True,
        )
        assert ed.messages[3]["params"]["update"]["content"] == {
            "type": "text",
            "text": denied,
        }
        assert server._sessions[sid]["history"] == [
            {"role": "user", "content": "permission"},
            {"role": "assistant", "content": denied},
        ]
        assert list(tmp_path.iterdir()) == []
        assert conn._pending == {}


class TerminalTurnDeps(PipeTurnDeps):
    pass


class TerminalTurnInput(PipeTurnInput):
    pass


class TerminalTurnAgent(PipeTurnAgent):
    async def _process_impl(self, input):
        return PipeTurnOutput(
            pending_tool_calls=[
                {
                    "id": "terminal_call",
                    "name": "run_command",
                    "arguments": {"command": "sleep 60"},
                }
            ]
        )


@pytest.mark.asyncio
async def test_session_cancel_finishes_when_editor_never_releases_terminal(
    dispatcher, tmp_path, caplog
):
    previous = ConfigLoader.AGENT_CLASSES["acp_chunk_agent"]
    ConfigLoader.AGENT_CLASSES["acp_chunk_agent"] = f"{__name__}:TerminalTurnAgent"
    try:
        async with running(dispatcher, tmp_path, caps={"terminal": True}) as (
            ed,
            conn,
            server,
            sid,
            task,
        ):
            await ed.prompt(1, sid, "terminal")
            permission = await ed.until(
                lambda m: m.get("method") == "session/request_permission"
            )
            await ed.send(
                {
                    "jsonrpc": "2.0",
                    "id": permission["id"],
                    "result": {
                        "outcome": {"outcome": "selected", "optionId": "allow-once"}
                    },
                }
            )
            create = await ed.until(lambda m: m.get("method") == "terminal/create")
            await ed.send(
                {
                    "jsonrpc": "2.0",
                    "id": create["id"],
                    "result": {"terminalId": "session-terminal"},
                }
            )
            await ed.until(lambda m: m.get("method") == "terminal/wait_for_exit")
            await ed.send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": sid},
                }
            )
            release = await ed.until(lambda m: m.get("method") == "terminal/release")
            assert release["params"] == {
                "sessionId": sid,
                "terminalId": "session-terminal",
            }
            async with asyncio.timeout(0.5):
                result = await ed.until(lambda m: m.get("id") == 1)
            assert result == {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"stopReason": "cancelled"},
            }
            assert (
                "terminal/release timed out after 0.1s for session-terminal"
                in caplog.text
            )
            assert conn._pending == {}
            assert server._sessions[sid]["history"] == []
            response = await ed.request(2, "session/new", {"cwd": str(tmp_path)})
            assert response == {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"sessionId": "sess_00000002"},
            }
    finally:
        ConfigLoader.AGENT_CLASSES["acp_chunk_agent"] = previous


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, {}, {"content": None}, {"content": 42}])
async def test_file_read_rejects_malformed_editor_result(tmp_path, result):
    async with editor_connection() as (conn, receive, reply):
        pending = asyncio.create_task(
            tools.execute_tool_call(
                conn,
                "workspace",
                "read_file",
                {"path": "answer.txt"},
                workspace_root=tmp_path,
            )
        )
        request = await receive()
        reply(request, result=result)
        with pytest.raises(ValueError) as error:
            await pending
        assert (
            str(error.value)
            == "fs/read_text_file returned invalid result: content must be a string"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [{}, {"content": None}, {"content": 42}])
async def test_file_write_requires_content_before_request(tmp_path, content):
    async with editor_connection() as (conn, receive, reply):
        with pytest.raises(ValueError) as error:
            await asyncio.wait_for(
                tools.execute_tool_call(
                    conn,
                    "workspace",
                    "write_file",
                    {"path": "answer.txt", **content},
                    workspace_root=tmp_path,
                ),
                0.2,
            )
        assert (
            str(error.value) == "write_file arguments invalid: content must be a string"
        )
        assert conn._next_id == 0
        assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "result", "message"),
    [
        ("terminal/create", None, "terminalId must be a nonempty string"),
        ("terminal/create", {}, "terminalId must be a nonempty string"),
        ("terminal/create", {"terminalId": ""}, "terminalId must be a nonempty string"),
        (
            "terminal/create",
            {"terminalId": 123},
            "terminalId must be a nonempty string",
        ),
        (
            "terminal/wait_for_exit",
            None,
            "exitCode must be an integer or signal a nonempty string",
        ),
        (
            "terminal/wait_for_exit",
            {},
            "exitCode must be an integer or signal a nonempty string",
        ),
        (
            "terminal/wait_for_exit",
            {"exitCode": True},
            "exitCode must be an integer or signal a nonempty string",
        ),
        (
            "terminal/wait_for_exit",
            {"exitCode": "0"},
            "exitCode must be an integer or signal a nonempty string",
        ),
        ("terminal/output", None, "output must be a string"),
        ("terminal/output", {}, "output must be a string"),
        ("terminal/output", {"output": None}, "output must be a string"),
        ("terminal/output", {"output": 42}, "output must be a string"),
    ],
)
async def test_terminal_rejects_malformed_editor_result(
    tmp_path, method, result, message
):
    async with editor_connection() as (conn, receive, reply):
        methods = []

        async def editor():
            while True:
                request = await receive()
                called = request["method"]
                methods.append(called)
                reply(
                    request,
                    result=result
                    if called == method
                    else {
                        "terminal/create": {"terminalId": "owned-terminal"},
                        "terminal/wait_for_exit": {"exitCode": 0},
                        "terminal/output": {"output": "done\n"},
                        "terminal/release": {},
                    }[called],
                )
                if called == "terminal/release":
                    return

        editor_task = asyncio.create_task(editor())
        try:
            with pytest.raises(ValueError) as error:
                await asyncio.wait_for(
                    tools.execute_tool_call(
                        conn,
                        "workspace",
                        "run_command",
                        {"command": "true"},
                        workspace_root=tmp_path,
                    ),
                    1,
                )
            assert str(error.value) == f"{method} returned invalid result: {message}"
            assert (
                methods
                == {
                    "terminal/create": ["terminal/create"],
                    "terminal/wait_for_exit": [
                        "terminal/create",
                        "terminal/wait_for_exit",
                        "terminal/release",
                    ],
                    "terminal/output": [
                        "terminal/create",
                        "terminal/wait_for_exit",
                        "terminal/output",
                        "terminal/release",
                    ],
                }[method]
            )
        finally:
            editor_task.cancel()
            await asyncio.gather(editor_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_terminal_reports_signal_termination(tmp_path):
    async with editor_connection() as (conn, receive, reply):
        pending = asyncio.create_task(
            tools.execute_tool_call(
                conn,
                "workspace",
                "run_command",
                {"command": "kill -TERM $$"},
                workspace_root=tmp_path,
            )
        )
        create = await receive()
        params = create["params"]
        process = await asyncio.create_subprocess_exec(
            params["command"], *params["args"], cwd=params["cwd"]
        )
        reply(create, result={"terminalId": "signalled-terminal"})
        wait = await receive()
        assert await process.wait() == -15
        reply(wait, result={"exitCode": None, "signal": "SIGTERM"})
        output = await receive()
        reply(output, result={"output": ""})
        release = await receive()
        reply(release, result={})
        assert await pending == "exit_signal=SIGTERM\n"
