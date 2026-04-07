"""SSE streaming client for A2A agent communication."""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.markdown import Markdown

console = Console()

RUNTIME_URL = "http://localhost:28000"


@dataclass
class CodingResult:
    """Parsed result from a coding agent response."""

    plan: str = ""
    code_changes: List[Dict[str, str]] = field(default_factory=list)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    iterations_used: int = 0
    files_modified: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def _build_a2a_request(
    query: str,
    agent_name: str = "coding_agent",
    tenant_id: str = "default",
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """Build a JSON-RPC 2.0 A2A message/stream request."""
    request_id = uuid.uuid4().hex[:16]
    message_id = uuid.uuid4().hex
    context_id = uuid.uuid4().hex[:16]

    metadata = {
        "agent_name": agent_name,
        "query": query,
        "tenant_id": tenant_id,
        "stream": True,
    }
    if context:
        metadata.update(context)

    message = {
        "kind": "message",
        "messageId": message_id,
        "role": "user",
        "parts": [{"kind": "text", "text": query}],
        "contextId": context_id,
    }
    if conversation_history:
        message["metadata"] = {"conversation_history": conversation_history}

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/stream",
        "params": {
            "metadata": metadata,
            "message": message,
        },
    }


def _parse_coding_result(data: Dict[str, Any]) -> CodingResult:
    """Parse a CodingOutput from the final event data."""
    result = data.get("result", data)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return CodingResult(summary=result, raw=data)

    if isinstance(result, dict) and "result" in result:
        result = result["result"]

    return CodingResult(
        plan=result.get("plan", ""),
        code_changes=result.get("code_changes", []),
        execution_results=result.get("execution_results", []),
        summary=result.get("summary", ""),
        iterations_used=result.get("iterations_used", 0),
        files_modified=result.get("files_modified", []),
        raw=data,
    )


_PHASE_LABELS = {
    "search": "Searching code context",
    "plan": "Planning implementation",
    "generate": "Generating code",
    "execute": "Executing in sandbox",
    "evaluate": "Evaluating output",
    "summarize": "Summarizing results",
    "rlm_synthesis": "RLM synthesis",
    "done": "Complete",
}


def stream_coding_response(
    query: str,
    agent_name: str = "coding_agent",
    tenant_id: str = "default",
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    runtime_url: str = RUNTIME_URL,
) -> Optional[CodingResult]:
    """Send a coding task via A2A SSE and render streaming output.

    Returns the parsed CodingResult on success, None on error.
    """
    request_body = _build_a2a_request(
        query=query,
        agent_name=agent_name,
        tenant_id=tenant_id,
        context=context,
        conversation_history=conversation_history,
    )

    result: Optional[CodingResult] = None

    try:
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0), follow_redirects=True) as client:
            with client.stream(
                "POST",
                f"{runtime_url}/a2a/",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    console.print(
                        f"[red]Runtime error: {response.status_code}[/red]"
                    )
                    return None

                current_phase = ""
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if not data_str:
                        continue

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    result = _handle_event(event, current_phase)
                    if isinstance(result, str):
                        current_phase = result
                        result = None
                    elif isinstance(result, CodingResult):
                        break

    except httpx.ConnectError:
        console.print("[red]Cannot connect to runtime. Run `cogniverse up` first.[/red]")
        return None
    except httpx.ReadTimeout:
        console.print("[yellow]Request timed out (10 min limit).[/yellow]")
        return None

    return result


def _handle_event(event: dict, current_phase: str):
    """Handle a single SSE event. Returns new phase name or CodingResult.

    Event structure:
      {"id": ..., "result": {"status": {"state": "...", "message": {"parts": [{"kind": "text", "text": "{...json...}"}]}}}}
    """
    result_obj = event.get("result", event)
    status = result_obj.get("status", {})
    state = status.get("state", "")
    message = status.get("message", {})

    if not isinstance(message, dict):
        return current_phase

    parts = message.get("parts", [])
    text = ""
    for part in parts:
        if isinstance(part, dict) and (part.get("kind") == "text" or part.get("type") == "text"):
            text = part.get("text", "")
            break

    if not text:
        return current_phase

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = {"text": text}

    event_type = payload.get("type", "")

    if event_type == "status":
        phase = payload.get("phase", "")
        label = _PHASE_LABELS.get(phase, phase)
        if phase != current_phase:
            console.print(f"  [dim]>> {label}...[/dim]")
        return phase

    if event_type == "partial":
        partial_text = payload.get("text", payload.get("data", ""))
        if partial_text:
            console.print(partial_text, end="")
        return current_phase

    if event_type == "final":
        return _parse_coding_result(payload.get("data", payload))

    if state in ("completed", "input-required", "input_required"):
        return _parse_coding_result(payload)

    return current_phase


def render_coding_result(result: CodingResult) -> None:
    """Pretty-print a CodingResult to the terminal."""
    if result.plan:
        console.print()
        console.print("[bold]## Plan[/bold]")
        console.print(Markdown(result.plan))

    for change in result.code_changes:
        file_path = change.get("file_path", "unknown")
        change_type = change.get("change_type", "modify")
        content = change.get("content", "")
        console.print()
        console.print(f"[bold cyan]{file_path}[/bold cyan] ({change_type}):")
        console.print(content)

    for i, exec_result in enumerate(result.execution_results):
        stderr = exec_result.get("stderr", "")
        exit_code = exec_result.get("exit_code", -1)
        if exit_code == 0:
            console.print(f"  [green]Iteration {i + 1}: passed[/green]")
        else:
            console.print(f"  [red]Iteration {i + 1}: failed (exit {exit_code})[/red]")
            if stderr:
                console.print(f"  [dim]{stderr[:300]}[/dim]")

    if result.summary:
        console.print()
        console.print("[bold]## Summary[/bold]")
        console.print(result.summary)

    if result.files_modified:
        console.print(f"  [dim]Files: {', '.join(result.files_modified)}[/dim]")
    if result.iterations_used:
        console.print(f"  [dim]Iterations: {result.iterations_used}[/dim]")
