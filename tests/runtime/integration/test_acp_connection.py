"""ACP sessions, cancellation and dispatcher turns over operating-system pipes."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

import dspy
import pytest

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.acp.server import ACPServer, ClientConnection, serve
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

EVENTS = []
GATE = None
ANSWER = "A" * 128 + "B" * 128 + "C"


class PipeTurnDeps(AgentDeps):
    pass


class PipeTurnInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    attachments: list = []
    external_tools: list = []
    tool_results: list = []
    tool_exchange: list = []
    continuation_state: dict = {}
    max_iterations: int = 0


class PipeTurnOutput(AgentOutput):
    summary: str = ""
    pending_tool_calls: list = []
    continuation_state: dict = {}


class PipeTurnAgent(AgentBase[PipeTurnInput, PipeTurnOutput, PipeTurnDeps]):
    async def _process_impl(self, input):
        EVENTS.append(("start", input.query))
        try:
            if input.query == "wait":
                await GATE.wait()
            if input.query == "lm failure":
                lm = dspy.LM(
                    "openai/acp-unreachable",
                    api_base="http://127.0.0.1:29071/v1",
                    api_key="test",
                    num_retries=0,
                    timeout=1,
                    cache=False,
                )
                await lm.acall(messages=[{"role": "user", "content": "hello"}])
            if input.query in {"permission", "rounds"}:
                step = input.continuation_state.get("step", 0)
                if input.query == "permission" and input.tool_results:
                    return PipeTurnOutput(
                        summary=json.dumps(input.tool_results, sort_keys=True)
                    )
                EVENTS.append(("round", step, input.max_iterations))
                return PipeTurnOutput(
                    pending_tool_calls=[
                        {
                            "id": f"call_{step}",
                            "name": "write_file",
                            "arguments": {"path": "answer.txt", "content": "written\n"},
                        }
                    ],
                    continuation_state={"step": step + 1},
                )
            if input.attachments:
                return PipeTurnOutput(
                    summary=json.dumps(
                        {"query": input.query, "attachments": input.attachments},
                        sort_keys=True,
                    )
                )
            self.emit_progress("token", "SECRET", data={"output_field": "gaps"})
            self.emit_progress("token", "A" * 128, data={"output_field": "summary"})
            return PipeTurnOutput(summary=ANSWER)
        except asyncio.CancelledError:
            EVENTS.append(("cancel", input.query))
            raise


class TruncatedPipeAgent(PipeTurnAgent):
    async def process(self, input, stream=False):
        async def events():
            yield {
                "phase": "token",
                "message": "unfinished",
                "data": {"output_field": "summary"},
            }

        return events()


@pytest.fixture
def dispatcher():
    global GATE
    EVENTS.clear()
    GATE = asyncio.Event()
    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id="acp:test", config_manager=manager)
    for name, streams in [("acp_live_agent", True), ("acp_chunk_agent", False)]:
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url="http://127.0.0.1:29071",
                capabilities=["acp_pipe"],
                streams_answer_tokens=streams,
            )
        )
        ConfigLoader.AGENT_CLASSES[name] = f"{__name__}:PipeTurnAgent"
    registry.register_agent(
        AgentEndpoint(
            name="acp_truncated_agent",
            url="http://127.0.0.1:29071",
            capabilities=["acp_pipe"],
            streams_answer_tokens=True,
        )
    )
    ConfigLoader.AGENT_CLASSES["acp_truncated_agent"] = f"{__name__}:TruncatedPipeAgent"
    yield AgentDispatcher(
        agent_registry=registry, config_manager=manager, schema_loader=None
    )
    for name in ["acp_live_agent", "acp_chunk_agent", "acp_truncated_agent"]:
        ConfigLoader.AGENT_CLASSES.pop(name)


async def pipe():
    read_fd, write_fd = os.pipe()
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    read_transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        os.fdopen(read_fd, "rb", buffering=0),
    )
    protocol = asyncio.streams.FlowControlMixin(loop=loop)
    transport, _ = await loop.connect_write_pipe(
        lambda: protocol, os.fdopen(write_fd, "wb", buffering=0)
    )
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    return reader, writer, read_transport


class Editor:
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.messages = []

    async def send(self, message):
        self.writer.write((json.dumps(message) + "\n").encode())
        await self.writer.drain()

    async def until(self, predicate):
        async with asyncio.timeout(5):
            while True:
                line = await self.reader.readline()
                if not line:
                    raise EOFError("ACP pipe closed")
                message = json.loads(line)
                self.messages.append(message)
                if predicate(message):
                    return message

    async def request(self, rid, method, params):
        await self.send(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        )
        return await self.until(lambda m: m.get("id") == rid)

    async def prompt(self, rid, sid, text):
        await self.send(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": text}],
                },
            }
        )


@asynccontextmanager
async def running(dispatcher, root, *, agent="acp_chunk_agent", caps=None):
    reader, editor_writer, input_transport = await pipe()
    editor_reader, writer, output_transport = await pipe()

    async def write(message):
        writer.write((json.dumps(message) + "\n").encode())
        await writer.drain()

    conn = ClientConnection(reader, write)
    server = ACPServer(
        dispatcher_provider=lambda: dispatcher,
        agent_provider=lambda: agent,
        coding_agent_provider=lambda: "acp_chunk_agent",
        default_tenant="acp:test",
    )
    server.initialize({"clientCapabilities": caps or {}})
    sid = server.session_new({"cwd": str(root)})["sessionId"]
    task = asyncio.create_task(serve(server, conn))
    editor = Editor(editor_reader, editor_writer)
    try:
        yield editor, conn, server, sid, task
    finally:
        editor_writer.close()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
        for handler in list(conn._tasks):
            handler.cancel()
        await asyncio.gather(*list(conn._tasks), return_exceptions=True)
        writer.close()
        input_transport.close()
        output_transport.close()


async def test_cancelled_permission_replies_leave_no_futures(dispatcher, tmp_path):
    async with running(dispatcher, tmp_path, caps={"fs": {"writeTextFile": True}}) as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        late_ids = []
        for rid in range(1, 21):
            await ed.prompt(rid, sid, "permission")
            request = await ed.until(
                lambda m: m.get("method") == "session/request_permission"
            )
            late_ids.append(request["id"])
            await ed.send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": sid},
                }
            )
            assert await ed.until(lambda m: m.get("id") == rid) == {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"stopReason": "cancelled"},
            }
        assert len(conn._pending) == 0
        for reply_id in late_ids + ["unknown", [], {}]:
            await ed.send(
                {
                    "jsonrpc": "2.0",
                    "id": reply_id,
                    "result": {
                        "outcome": {"outcome": "selected", "optionId": "allow-once"}
                    },
                }
            )
        assert await ed.request(99, "session/new", {"cwd": str(tmp_path)}) == {
            "jsonrpc": "2.0",
            "id": 99,
            "result": {"sessionId": "sess_00000002"},
        }
        assert server._sessions[sid]["history"] == []
        assert list(tmp_path.iterdir()) == []
        assert task.done() is False
        print("cancelled_turns=20 pending_futures=0 late_replies_survived=23")


async def test_already_done_reply_is_consumed(dispatcher, tmp_path):
    async with running(dispatcher, tmp_path) as (ed, conn, server, sid, task):
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        conn._pending["finished"] = future
        await ed.send({"jsonrpc": "2.0", "id": "finished", "result": {}})
        assert await ed.request(9, "session/new", {"cwd": str(tmp_path)}) == {
            "jsonrpc": "2.0",
            "id": 9,
            "result": {"sessionId": "sess_00000002"},
        }
        assert conn._pending == {}


async def test_concurrent_prompt_refused_and_cancel_reaches_running_turn(
    dispatcher, tmp_path
):
    async with running(dispatcher, tmp_path, caps={"fs": {"writeTextFile": True}}) as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        await ed.prompt(1, sid, "permission")
        await ed.until(lambda m: m.get("method") == "session/request_permission")
        await ed.prompt(2, sid, "permission")
        assert await ed.until(lambda m: m.get("id") == 2) == {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {
                "code": -32002,
                "message": "session_busy: a prompt is already running",
            },
        }
        await ed.send(
            {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": sid}}
        )
        assert await ed.until(lambda m: m.get("id") == 1) == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"stopReason": "cancelled"},
        }
        assert [event for event in EVENTS if event[0] == "start"] == [
            ("start", "permission")
        ]
        assert server._sessions[sid]["history"] == []
        for rid, query in [(3, "third"), (4, "fourth")]:
            await ed.prompt(rid, sid, query)
            assert (await ed.until(lambda m: m.get("id") == rid))["result"] == {
                "stopReason": "end_turn"
            }
        assert server._sessions[sid]["history"] == [
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": ANSWER},
            {"role": "user", "content": "fourth"},
            {"role": "assistant", "content": ANSWER},
        ]
        print("started_before_cancel=1 second=session_busy history_order=third,fourth")


@pytest.mark.parametrize(
    "agent,expected",
    [
        ("acp_chunk_agent", ["A" * 128 + "B" * 128, "C"]),
        ("acp_live_agent", ["A" * 128, "B" * 128 + "C"]),
    ],
)
async def test_answer_stream_flag_and_field_filter(
    dispatcher, tmp_path, agent, expected
):
    async with running(dispatcher, tmp_path, agent=agent) as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        await ed.prompt(1, sid, "answer")
        assert (await ed.until(lambda m: m.get("id") == 1))["result"] == {
            "stopReason": "end_turn"
        }
        chunks = [
            m["params"]["update"]["content"]["text"]
            for m in ed.messages
            if m.get("method") == "session/update"
        ]
        assert chunks == expected
        assert server._sessions[sid]["history"] == [
            {"role": "user", "content": "answer"},
            {"role": "assistant", "content": ANSWER},
        ]
        print(f"agent={agent} chunk_lengths={[len(c) for c in chunks]}")
        if agent == "acp_live_agent":
            from cogniverse_agents.search_agent import ConversationalQueryRewriteModule

            dispatcher._query_rewriter = ConversationalQueryRewriteModule()
            dispatcher._query_rewriter.set_lm(
                dspy.LM(
                    "openai/acp-unreachable",
                    api_base="http://127.0.0.1:29071/v1",
                    api_key="test",
                    num_retries=0,
                    timeout=1,
                    cache=False,
                )
            )
        await ed.prompt(2, sid, "answer2")
        second = await ed.until(lambda m: m.get("id") == 2)
        if agent == "acp_live_agent":
            assert second == {
                "jsonrpc": "2.0",
                "id": 2,
                "error": {
                    "code": -32603,
                    "message": "litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.",
                },
            }
            assert server._sessions[sid]["history"] == [
                {"role": "user", "content": "answer"},
                {"role": "assistant", "content": ANSWER},
            ]
        else:
            assert second == {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"stopReason": "end_turn"},
            }
            assert server._sessions[sid]["history"] == [
                {"role": "user", "content": "answer"},
                {"role": "assistant", "content": ANSWER},
                {"role": "user", "content": "answer2"},
                {"role": "assistant", "content": ANSWER},
            ]


@pytest.mark.parametrize("caption", ["", "describe"])
@pytest.mark.parametrize("caps", [{}, {"fs": {"readTextFile": True}}])
async def test_image_blocks_reach_agent_attachments(
    dispatcher, tmp_path, caption, caps
):
    async with running(dispatcher, tmp_path, caps=caps) as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        prompt = [{"type": "image", "mimeType": "image/png", "data": "aGVsbG8="}]
        if caption:
            prompt.insert(0, {"type": "text", "text": caption})
        response = await ed.request(
            1, "session/prompt", {"sessionId": sid, "prompt": prompt}
        )
        assert response == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"stopReason": "end_turn"},
        }
        answer = "".join(
            m["params"]["update"]["content"]["text"]
            for m in ed.messages
            if m.get("method") == "session/update"
        )
        assert json.loads(answer) == {
            "query": caption,
            "attachments": ["data:image/png;base64,aGVsbG8="],
        }


@pytest.mark.parametrize("text", [None, 123, ["text"]])
async def test_malformed_text_has_indexed_error(dispatcher, tmp_path, text):
    async with running(dispatcher, tmp_path) as (ed, conn, server, sid, task):
        assert await ed.request(
            1,
            "session/prompt",
            {"sessionId": sid, "prompt": [{"type": "text", "text": text}]},
        ) == {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32602, "message": "prompt[0].text must be a string"},
        }
        assert EVENTS == []


async def test_round_cap_is_shared_with_coding_agent(dispatcher, tmp_path):
    async with running(dispatcher, tmp_path, caps={"fs": {"writeTextFile": True}}) as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        await ed.prompt(1, sid, "rounds")
        for step in range(8):
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
            write = await ed.until(lambda m: m.get("method") == "fs/write_text_file")
            assert write["params"]["content"] == "written\n"
            (tmp_path / write["params"]["path"]).write_text(write["params"]["content"])
            await ed.send({"jsonrpc": "2.0", "id": write["id"], "result": None})
        assert await ed.until(
            lambda m: (
                m.get("id") == 1 or m.get("method") == "session/request_permission"
            )
        ) == {"jsonrpc": "2.0", "id": 1, "result": {"stopReason": "max_turn_requests"}}
        assert [
            m["params"]["path"]
            for m in ed.messages
            if m.get("method") == "fs/write_text_file"
        ] == [str(tmp_path / "answer.txt")] * 8
        assert (tmp_path / "answer.txt").read_text() == "written\n"
        assert [event for event in EVENTS if event[0] == "round"] == [
            ("round", step, 8) for step in range(8)
        ]
        print("workspace_dispatches=8 max_iterations=8 persisted=written")


async def test_dead_lm_is_reported_and_connection_survives(dispatcher, tmp_path):
    async with running(dispatcher, tmp_path) as (ed, conn, server, sid, task):
        await ed.prompt(1, sid, "lm failure")
        response = await ed.until(lambda m: m.get("id") == 1)
        assert response["error"]["code"] == -32603
        assert (
            response["error"]["message"]
            == "litellm.InternalServerError: InternalServerError: OpenAIException - Connection error."
        )
        assert server._sessions[sid]["history"] == []
        assert await ed.request(2, "session/new", {"cwd": str(tmp_path)}) == {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"sessionId": "sess_00000002"},
        }


async def test_truncated_stream_reports_error_without_committing_history(
    dispatcher, tmp_path
):
    async with running(dispatcher, tmp_path, agent="acp_truncated_agent") as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        await ed.prompt(1, sid, "answer")
        assert await ed.until(lambda m: m.get("id") == 1) == {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32603,
                "message": "Agent 'acp_truncated_agent' ended its stream without a final answer",
            },
        }
        assert server._sessions[sid]["history"] == []


async def test_eof_cancels_prompt_and_joins_handlers(dispatcher, tmp_path):
    async with running(dispatcher, tmp_path, caps={"fs": {"writeTextFile": True}}) as (
        ed,
        conn,
        server,
        sid,
        task,
    ):
        await ed.prompt(1, sid, "permission")
        await ed.until(lambda m: m.get("method") == "session/request_permission")
        ed.writer.close()
        assert await asyncio.wait_for(task, 2) == 0
        assert conn._pending == {}
        assert conn._tasks == set()
        assert server._sessions[sid]["history"] == []
