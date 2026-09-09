"""Map the coding agent's workspace tool calls onto ACP editor methods.

The coding agent is tool-agnostic: it picks among the ``external_tools`` it is
advertised and emits ``pending_tool_calls`` by name. Here we advertise the
subset the editor's ``clientCapabilities`` support and translate each chosen
call into the ACP client requests (``fs/*`` / ``terminal/*``) the editor
services back over the connection.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
TERMINAL_RELEASE_TIMEOUT = 0.1

READ_FILE = "read_file"
WRITE_FILE = "write_file"
RUN_COMMAND = "run_command"

# Calls that change the workspace and must clear a permission prompt first.
MUTATING_TOOLS = {WRITE_FILE, RUN_COMMAND}

_TOOL_KINDS = {READ_FILE: "read", WRITE_FILE: "edit", RUN_COMMAND: "execute"}


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def advertised_tools(client_capabilities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """OpenAI tool definitions for the editor methods the client supports."""
    fs = client_capabilities.get("fs") or {}
    tools: List[Dict[str, Any]] = []
    if fs.get("readTextFile"):
        tools.append(
            _tool(
                READ_FILE,
                "Read a UTF-8 text file from the workspace.",
                {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace, or an absolute path within it",
                    }
                },
                ["path"],
            )
        )
    if fs.get("writeTextFile"):
        tools.append(
            _tool(
                WRITE_FILE,
                "Create or overwrite a UTF-8 text file in the workspace.",
                {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace, or an absolute path within it",
                    },
                    "content": {"type": "string", "description": "Full file content"},
                },
                ["path", "content"],
            )
        )
    if client_capabilities.get("terminal"):
        tools.append(
            _tool(
                RUN_COMMAND,
                "Run a shell command and return its output and exit code.",
                {"command": {"type": "string", "description": "Shell command line"}},
                ["command"],
            )
        )
    return tools


def tool_kind(name: str) -> str:
    """ACP tool-call ``kind`` for the editor UI."""
    return _TOOL_KINDS.get(name, "other")


def permission_options() -> List[Dict[str, str]]:
    return [
        {"optionId": "allow-once", "name": "Allow", "kind": "allow_once"},
        {"optionId": "allow-always", "name": "Always allow", "kind": "allow_always"},
        {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
    ]


def outcome_allows(
    outcome: Any, *, offered_options: List[Dict[str, str]] | None = None
) -> bool:
    """Accept a selected allow option from the exact options sent to the editor."""
    options = permission_options() if offered_options is None else offered_options
    inner = outcome.get("outcome") if isinstance(outcome, dict) else None
    if not isinstance(inner, dict):
        logger.warning(
            "Denied malformed session/request_permission outcome: %r", outcome
        )
        return False
    selected = inner.get("optionId")
    allowed = {
        option["optionId"]
        for option in options
        if option.get("kind") in {"allow_once", "allow_always"}
    }
    permitted = (
        inner.get("outcome") == "selected"
        and isinstance(selected, str)
        and selected in allowed
    )
    if not permitted:
        logger.warning("Denied session/request_permission outcome: %r", outcome)
    return permitted


def _workspace_path(path: str, workspace_root: Path) -> Path:
    root = workspace_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path {path!r} is outside workspace root {root}")
    return resolved


def _result_text(
    result: Any, method: str, field: str, *, nonempty: bool = False
) -> str:
    value = result.get(field) if isinstance(result, dict) else None
    if not isinstance(value, str) or (nonempty and not value):
        requirement = "a nonempty string" if nonempty else "a string"
        raise ValueError(
            f"{method} returned invalid result: {field} must be {requirement}"
        )
    return value


async def execute_tool_call(
    conn: Any,
    session_id: str,
    name: str,
    arguments: Dict[str, Any],
    *,
    workspace_root: Path,
) -> str:
    """Run one workspace tool call against the editor; return the result text."""
    if name in {READ_FILE, WRITE_FILE}:
        path = str(_workspace_path(arguments["path"], workspace_root))
    if name == READ_FILE:
        res = await conn.call(
            "fs/read_text_file",
            {"sessionId": session_id, "path": path},
        )
        return _result_text(res, "fs/read_text_file", "content")
    if name == WRITE_FILE:
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("write_file arguments invalid: content must be a string")
        await conn.call(
            "fs/write_text_file",
            {
                "sessionId": session_id,
                "path": path,
                "content": content,
            },
        )
        return f"wrote {path}"
    if name == RUN_COMMAND:
        return await _run_command(
            conn, session_id, arguments["command"], workspace_root
        )
    raise ValueError(f"unknown workspace tool {name!r}")


async def _run_command(
    conn: Any, session_id: str, command: str, workspace_root: Path
) -> str:
    created = await conn.call(
        "terminal/create",
        {
            "sessionId": session_id,
            "command": "sh",
            "args": ["-c", command],
            "cwd": str(workspace_root.resolve()),
        },
    )
    terminal_id = _result_text(created, "terminal/create", "terminalId", nonempty=True)
    cancelled = False
    try:
        exit_status = await conn.call(
            "terminal/wait_for_exit",
            {"sessionId": session_id, "terminalId": terminal_id},
        )
        exit_code = (
            exit_status.get("exitCode") if isinstance(exit_status, dict) else None
        )
        signal = exit_status.get("signal") if isinstance(exit_status, dict) else None
        if type(exit_code) is int:
            status = f"exit_code={exit_code}"
        elif exit_code is None and isinstance(signal, str) and signal:
            status = f"exit_signal={signal}"
        else:
            raise ValueError(
                "terminal/wait_for_exit returned invalid result: "
                "exitCode must be an integer or signal a nonempty string"
            )
        out = await conn.call(
            "terminal/output",
            {"sessionId": session_id, "terminalId": terminal_id},
        )
        output = _result_text(out, "terminal/output", "output")
        return f"{status}\n{output}"
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        try:
            async with asyncio.timeout(TERMINAL_RELEASE_TIMEOUT):
                await conn.call(
                    "terminal/release",
                    {"sessionId": session_id, "terminalId": terminal_id},
                )
        except TimeoutError as exc:
            message = (
                f"terminal/release timed out after {TERMINAL_RELEASE_TIMEOUT}s "
                f"for {terminal_id}"
            )
            logger.warning(message)
            if not cancelled:
                raise TimeoutError(message) from exc
