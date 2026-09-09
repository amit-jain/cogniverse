"""Agent Client Protocol (ACP) server over the dispatcher.

Cogniverse as an ACP agent: an editor spawns this as a subprocess and speaks
JSON-RPC 2.0 over stdio, newline-delimited (one compact JSON object per line).
A prompt turn runs through ``AgentDispatcher`` — the same silent-brain path the
``/v1`` surface uses. When the editor advertises fs/terminal capabilities the
turn runs the coding agent's workspace loop and drives the editor's ``fs/*`` /
``terminal/*`` methods over the connection (permission-gated); otherwise it
streams tokens as ``session/update`` / ``agent_message_chunk`` notifications.

stdout is the protocol channel, so nothing here prints to it directly; the
entrypoint redirects application stdout to stderr and hands the connection a
dedicated protocol writer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from importlib.metadata import version
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from cogniverse_agents.coding_agent import WORKSPACE_MAX_ROUNDS
from cogniverse_runtime.acp import STDIN_LINE_LIMIT, tools
from cogniverse_runtime.harness_turn import (
    derive_request_seed,
    extract_answer_text,
    is_answer_field,
    to_openai_tool_calls,
)
from cogniverse_runtime.routers.openai_compat import (
    _validate_tool_calls,
    split_answer_chunks,
    use_token_stream,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

SESSION_BUSY = -32002

Writer = Callable[[Dict[str, Any]], Awaitable[None]]


class ACPError(Exception):
    """A JSON-RPC error to return for the current request."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _agent_version() -> str:
    return version("cogniverse-runtime")


def _prompt_content(blocks: Any) -> tuple[str, list[str]]:
    """Decode ACP text, embedded text resources and inline image attachments."""
    if not isinstance(blocks, list):
        raise ACPError(INVALID_PARAMS, "prompt must be an array")
    text_parts, attachments = [], []
    for index, block in enumerate(blocks):
        where = f"prompt[{index}]"
        if not isinstance(block, dict):
            raise ACPError(INVALID_PARAMS, f"{where} must be an object")
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ACPError(INVALID_PARAMS, f"{where}.text must be a string")
            text_parts.append(text)
        elif kind == "image":
            data, mime = block.get("data"), block.get("mimeType")
            if (
                not isinstance(data, str)
                or not data
                or not isinstance(mime, str)
                or not mime.startswith("image/")
            ):
                raise ACPError(
                    INVALID_PARAMS, f"{where} requires image mimeType and base64 data"
                )
            attachments.append(f"data:{mime};base64,{data}")
        elif kind == "resource":
            resource = block.get("resource")
            if not isinstance(resource, dict) or not isinstance(
                resource.get("text"), str
            ):
                raise ACPError(
                    INVALID_PARAMS, f"{where}.resource.text must be a string"
                )
            text_parts.append(resource["text"])
        else:
            raise ACPError(INVALID_PARAMS, f"{where} has unsupported type {kind!r}")
    return "".join(text_parts), attachments


class ClientConnection:
    """Bidirectional JSON-RPC over the stdio duplex.

    The editor's requests are dispatched to the server; the server can also
    issue its own requests to the editor (``fs/*`` / ``terminal/*`` /
    ``session/request_permission``) and await the responses, which the read
    loop routes back to the awaiting caller by id.
    """

    def __init__(self, reader: Any, write: Writer) -> None:
        self._reader = reader
        self._write = write
        self._next_id = 0
        self._pending: Dict[str, asyncio.Future] = {}
        self._tasks: set = set()
        self._closed = False

    async def call(self, method: str, params: Dict[str, Any]) -> Any:
        if self._closed:
            raise ACPError(INTERNAL_ERROR, "ACP connection closed")
        self._next_id += 1
        request_id = f"acp-c-{self._next_id}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            return await future
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def notify(self, method: str, params: Dict[str, Any]) -> None:
        if self._closed:
            raise ACPError(INTERNAL_ERROR, "ACP connection closed")
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def respond(self, request_id: Any, result: Any) -> None:
        if self._closed:
            return
        await self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def respond_error(self, request_id: Any, code: int, message: str) -> None:
        if self._closed:
            return
        await self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

    def _resolve_response(self, message: Dict[str, Any]) -> bool:
        """Resolve a reply to a server-issued request; True if consumed here."""
        if "method" in message:
            return False
        if "result" not in message and "error" not in message:
            return False
        request_id = message.get("id")
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            logger.warning("Ignoring ACP reply with invalid id %r", request_id)
            return True
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            logger.warning("Ignoring late or unknown ACP reply id %r", request_id)
            return True
        if "error" in message:
            err = message["error"]
            if not isinstance(err, dict):
                future.set_exception(
                    ACPError(INTERNAL_ERROR, "malformed editor error response")
                )
            else:
                future.set_exception(
                    ACPError(
                        err.get("code", INTERNAL_ERROR), str(err.get("message", ""))
                    )
                )
        else:
            future.set_result(message["result"])
        return True

    def fail_pending(self, reason: str) -> None:
        """Fail every outstanding server-issued request. Called when the
        connection closes so a turn awaiting an fs/terminal/permission reply
        (e.g. one the editor abandoned after cancelling) doesn't hang forever."""
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ACPError(INTERNAL_ERROR, reason))
        self._pending.clear()


class ACPServer:
    """Handles ACP methods against a single default agent + tenant."""

    def __init__(
        self,
        *,
        dispatcher_provider: Callable[[], Any],
        agent_provider: Callable[[], str],
        default_tenant: str,
        coding_agent_provider: Callable[[], str],
    ) -> None:
        self._dispatcher_provider = dispatcher_provider
        self._agent_provider = agent_provider
        self._coding_agent_provider = coding_agent_provider
        self._default_tenant = default_tenant
        self._client_capabilities: Dict[str, Any] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_seq = 0

    def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        capabilities = params.get("clientCapabilities", {})
        if not isinstance(capabilities, dict) or not isinstance(
            capabilities.get("fs", {}), dict
        ):
            raise ACPError(INVALID_PARAMS, "clientCapabilities and fs must be objects")
        self._client_capabilities = capabilities
        return {
            "protocolVersion": PROTOCOL_VERSION,
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
                "version": _agent_version(),
            },
            "authMethods": [],
        }

    def session_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cwd = params.get("cwd")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise ACPError(
                INVALID_PARAMS, "cwd must be an absolute workspace directory"
            )
        root = Path(cwd).resolve()
        if not root.is_dir():
            raise ACPError(
                INVALID_PARAMS, "cwd must be an existing workspace directory"
            )
        self._session_seq += 1
        session_id = f"sess_{self._session_seq:08d}"
        self._sessions[session_id] = {"cwd": root, "history": []}
        return {"sessionId": session_id}

    async def session_prompt(
        self, params: Dict[str, Any], conn: ClientConnection
    ) -> Dict[str, Any]:
        session_id = params.get("sessionId")
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            raise ACPError(INVALID_PARAMS, f"unknown sessionId {session_id!r}")

        query, attachments = _prompt_content(params.get("prompt"))
        if not query.strip() and not attachments:
            raise ACPError(INVALID_PARAMS, "prompt has no text or image content")
        running = session.get("prompt_task")
        if running is not None and not running.done():
            raise ACPError(SESSION_BUSY, "session_busy: a prompt is already running")

        session["prompt_task"] = asyncio.current_task()
        try:
            workspace_tools = tools.advertised_tools(self._client_capabilities)
            if workspace_tools:
                return await self._tool_loop(
                    session_id,
                    session["history"],
                    query,
                    workspace_tools,
                    conn,
                    attachments,
                )
            return await self._stream_answer(
                session_id, session["history"], query, conn, attachments
            )
        except asyncio.CancelledError:
            logger.info("ACP session/prompt cancelled for %s", session_id)
            return {"stopReason": "cancelled"}
        finally:
            session["prompt_task"] = None

    def session_cancel(self, params: Dict[str, Any]) -> None:
        """Stop an in-flight prompt turn (ACP session/cancel notification)."""
        session_id = params.get("sessionId")
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            return
        task = session.get("prompt_task")
        if task is not None and not task.done():
            task.cancel()

    async def _emit_chunk(
        self, conn: ClientConnection, session_id: str, message_id: str, text: str
    ) -> None:
        await conn.notify(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "messageId": message_id,
                    "content": {"type": "text", "text": text},
                },
            },
        )

    async def _stream_answer(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
        query: str,
        conn: ClientConnection,
        attachments: list[str],
    ) -> Dict[str, Any]:
        context = {
            "tenant_id": self._default_tenant,
            "conversation_history": history,
            "request_id": derive_request_seed(query, history),
            "attachments": attachments,
        }
        message_id = f"msg_{session_id}_{len(history)}"
        dispatcher = await asyncio.to_thread(self._dispatcher_provider)
        agent = await asyncio.to_thread(self._agent_provider)
        accumulated = ""
        if not await asyncio.to_thread(use_token_stream, dispatcher, agent, False):
            result = await dispatcher.dispatch(
                agent_name=agent, query=query, context=context
            )
            accumulated = extract_answer_text(result)
            for chunk in split_answer_chunks(accumulated):
                await self._emit_chunk(conn, session_id, message_id, chunk)
        else:
            answer_field = None
            completed = False
            async with aclosing(
                dispatcher.dispatch_stream(agent, query, context)
            ) as events:
                async for event in events:
                    if event.get("type") == "error":
                        raise ACPError(
                            INTERNAL_ERROR, str(event.get("message", "dispatch failed"))
                        )
                    if event.get("phase") == "token":
                        field = (event.get("data") or {}).get("output_field")
                        if not isinstance(field, str) or not is_answer_field(field):
                            continue
                        if answer_field is None:
                            answer_field = field
                        elif field != answer_field:
                            continue
                        delta = event.get("message") or ""
                        accumulated += delta
                        if delta:
                            await self._emit_chunk(conn, session_id, message_id, delta)
                    elif event.get("type") == "final":
                        completed = True
                        answer = extract_answer_text(event.get("data") or {})
                        if not answer.startswith(accumulated):
                            raise ACPError(
                                INTERNAL_ERROR,
                                f"Agent '{agent}' streamed text that its final answer does not begin with",
                            )
                        remainder = answer[len(accumulated) :]
                        accumulated = answer
                        if remainder:
                            await self._emit_chunk(
                                conn, session_id, message_id, remainder
                            )
            if not completed:
                raise ACPError(
                    INTERNAL_ERROR,
                    f"Agent '{agent}' ended its stream without a final answer",
                )
        history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": accumulated},
            ]
        )
        return {"stopReason": "end_turn"}

    async def _tool_loop(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
        query: str,
        workspace_tools: List[Dict[str, Any]],
        conn: ClientConnection,
        attachments: list[str],
    ) -> Dict[str, Any]:
        dispatcher = await asyncio.to_thread(self._dispatcher_provider)
        agent = await asyncio.to_thread(self._coding_agent_provider)
        seed = derive_request_seed(query, history)
        base_context = {
            "tenant_id": self._default_tenant,
            "conversation_history": history,
            "request_id": seed,
            "external_tools": workspace_tools,
            "attachments": attachments,
            "workspace_root": str(self._sessions[session_id]["cwd"]),
            "max_iterations": WORKSPACE_MAX_ROUNDS,
        }
        context = dict(base_context)
        tool_exchange: List[Dict[str, Any]] = []
        answer = ""
        stop_reason = "end_turn"

        for _ in range(WORKSPACE_MAX_ROUNDS):
            result = await dispatcher.dispatch(
                agent_name=agent, query=query, context=context
            )
            pending = (
                result.get("pending_tool_calls") if isinstance(result, dict) else None
            )
            if not pending:
                answer = extract_answer_text(result)
                break

            continuation = result.get("continuation_state") or {}
            assistant_tool_calls = _validate_tool_calls(
                {"tool_calls": to_openai_tool_calls(pending)}, f"agent {agent!r}"
            )
            allowed = {tool["function"]["name"] for tool in workspace_tools}
            call_ids = [call["id"] for call in assistant_tool_calls]
            if len(call_ids) != len(set(call_ids)):
                raise ACPError(
                    INTERNAL_ERROR, f"Agent '{agent}' returned duplicate tool call ids"
                )
            for call in pending:
                if call["name"] not in allowed or not isinstance(
                    call.get("arguments"), dict
                ):
                    raise ACPError(
                        INTERNAL_ERROR,
                        f"Agent '{agent}' returned invalid workspace tool {call['name']!r}",
                    )
            results: List[Dict[str, Any]] = []
            for call in pending:
                content = await self._run_one_tool(session_id, call, conn)
                results.append({"tool_call_id": call["id"], "content": content})
            tool_exchange.append(
                {"tool_calls": assistant_tool_calls, "results": results}
            )
            context = dict(base_context)
            context.update(
                {
                    "assistant_tool_calls": assistant_tool_calls,
                    "tool_results": results,
                    "tool_exchange": tool_exchange,
                    "continuation_state": continuation,
                }
            )
        else:
            stop_reason = "max_turn_requests"

        if answer:
            await self._emit_chunk(
                conn, session_id, f"msg_{session_id}_{len(history)}", answer
            )
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        return {"stopReason": stop_reason}

    async def _run_one_tool(
        self, session_id: str, call: Dict[str, Any], conn: ClientConnection
    ) -> str:
        call_id = call["id"]
        name = call["name"]
        arguments = call.get("arguments") or {}
        title = f"{name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
        kind = tools.tool_kind(name)

        async def _update(status: str) -> None:
            await conn.notify(
                "session/update",
                {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": call_id,
                        "status": status,
                    },
                },
            )

        await conn.notify(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": title,
                    "kind": kind,
                    "status": "pending",
                    "rawInput": arguments,
                },
            },
        )

        if name in tools.MUTATING_TOOLS:
            options = tools.permission_options()
            outcome = await conn.call(
                "session/request_permission",
                {
                    "sessionId": session_id,
                    "toolCall": {"toolCallId": call_id, "title": title, "kind": kind},
                    "options": options,
                },
            )
            if not tools.outcome_allows(outcome, offered_options=options):
                await _update("failed")
                return f"permission denied for {name}"

        try:
            content = await tools.execute_tool_call(
                conn,
                session_id,
                name,
                arguments,
                workspace_root=Path(self._sessions[session_id]["cwd"]),
            )
            await _update("completed")
            return content
        except Exception as exc:
            logger.exception("ACP workspace tool %s failed", name)
            await _update("failed")
            return f"{name} failed: {exc}"


async def handle_message(
    server: ACPServer, conn: ClientConnection, message: Dict[str, Any]
) -> None:
    """Route one parsed JSON-RPC request from the editor."""
    message_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})

    if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
        await conn.respond_error(None, INVALID_REQUEST, "invalid request")
        return
    if not isinstance(params, dict):
        if "id" in message:
            await conn.respond_error(
                message_id, INVALID_PARAMS, "params must be an object"
            )
        return

    try:
        if method == "initialize":
            result: Any = server.initialize(params)
        elif method == "session/new":
            result = server.session_new(params)
        elif method == "session/prompt":
            result = await server.session_prompt(params, conn)
        elif method == "session/cancel":
            server.session_cancel(params)
            return
        else:
            if message_id is not None:
                await conn.respond_error(
                    message_id, METHOD_NOT_FOUND, f"method not found: {method}"
                )
            return
        if message_id is not None:
            await conn.respond(message_id, result)
    except ACPError as exc:
        if message_id is not None:
            await conn.respond_error(message_id, exc.code, exc.message)
    except Exception as exc:
        logger.exception("ACP handler failed for %s", method)
        if message_id is not None:
            await conn.respond_error(message_id, INTERNAL_ERROR, str(exc))


async def serve(
    server: ACPServer, conn: ClientConnection, *, line_limit: int = STDIN_LINE_LIMIT
) -> int:
    """Read newline-delimited JSON-RPC from the connection until EOF.

    Editor responses to server-issued requests resolve their awaiting caller;
    editor requests are handled as tasks so the loop keeps reading (a prompt
    turn's own ``fs/*`` / ``terminal/*`` replies arrive on this same loop).
    """
    try:
        while True:
            try:
                line = await conn._reader.readline()
            except ValueError:
                message = f"Input line exceeds {line_limit} byte limit"
                logger.error(message)
                await conn.respond_error(None, INVALID_REQUEST, message)
                return 1
            if not line:
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except (json.JSONDecodeError, UnicodeDecodeError):
                await conn.respond_error(None, PARSE_ERROR, "parse error")
                continue
            if not isinstance(message, dict):
                await conn.respond_error(
                    None, INVALID_REQUEST, "Invalid Request: expected an object"
                )
                continue
            if conn._resolve_response(message):
                continue
            task = asyncio.create_task(handle_message(server, conn, message))
            conn._tasks.add(task)
            task.add_done_callback(conn._tasks.discard)
    finally:
        conn.fail_pending("ACP connection closed")
        tasks = list(conn._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
