"""Interactive coding REPL — talk to the coding agent from the terminal."""

from pathlib import Path
from typing import Dict, List, Optional

import httpx
from rich.console import Console
from rich.syntax import Syntax

from cogniverse_cli.streaming import (
    CodingResult,
    render_coding_result,
    stream_coding_response,
)

console = Console()

RUNTIME_URL = "http://localhost:28000"


class CodingSession:
    """Manages state for a multi-turn coding REPL session."""

    def __init__(
        self,
        tenant_id: str,
        language: str,
        max_iterations: int,
        codebase_path: str,
        runtime_url: str,
    ):
        self.tenant_id = tenant_id
        self.language = language
        self.max_iterations = max_iterations
        self.codebase_path = codebase_path
        self.runtime_url = runtime_url
        self.history: List[Dict[str, str]] = []
        self.last_result: Optional[CodingResult] = None

    def send(self, query: str) -> Optional[CodingResult]:
        """Send a coding task and stream the response."""
        context = {
            "language": self.language,
            "max_iterations": self.max_iterations,
            "codebase_path": self.codebase_path,
        }

        result = stream_coding_response(
            query=query,
            agent_name="coding_agent",
            tenant_id=self.tenant_id,
            context=context,
            conversation_history=self.history,
            runtime_url=self.runtime_url,
        )

        self.history.append({"role": "user", "content": query})
        if result:
            self.last_result = result
            self.history.append({"role": "assistant", "content": result.summary})

        return result

    def apply(self) -> int:
        """Write last code changes to local files. Returns count of files written."""
        if not self.last_result or not self.last_result.code_changes:
            console.print("[yellow]No code changes to apply.[/yellow]")
            return 0

        applied = 0
        for change in self.last_result.code_changes:
            file_path = change.get("file_path", "")
            content = change.get("content", "")
            change_type = change.get("change_type", "modify")

            if not file_path:
                continue

            path = Path(file_path)

            if change_type == "delete":
                if path.exists():
                    path.unlink()
                    console.print(f"  [red]Deleted {file_path}[/red]")
                    applied += 1
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            label = "new" if change_type == "new" else "modified"
            console.print(f"  [green]{file_path} ({label})[/green]")
            applied += 1

        if applied:
            console.print(f"Applied {applied} file(s)")
        return applied

    def show_diff(self) -> None:
        """Show diff between proposed changes and local files."""
        if not self.last_result or not self.last_result.code_changes:
            console.print("[yellow]No code changes to diff.[/yellow]")
            return

        for change in self.last_result.code_changes:
            file_path = change.get("file_path", "")
            content = change.get("content", "")
            change_type = change.get("change_type", "modify")

            path = Path(file_path)
            console.print(f"\n[bold]{file_path}[/bold] ({change_type}):")

            if change_type == "new" or not path.exists():
                console.print(Syntax(content, "python", theme="monokai"))
            elif change_type == "delete":
                console.print("[red]File will be deleted[/red]")
            else:
                existing = path.read_text()
                if existing == content:
                    console.print("[dim]No changes[/dim]")
                else:
                    console.print("[red]--- existing[/red]")
                    console.print("[green]+++ proposed[/green]")
                    console.print(Syntax(content, "python", theme="monokai"))

    def show_plan(self) -> None:
        """Re-display the last plan."""
        if not self.last_result or not self.last_result.plan:
            console.print("[yellow]No plan available.[/yellow]")
            return
        console.print("\n[bold]## Plan[/bold]")
        console.print(self.last_result.plan)

    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        self.last_result = None
        console.print("[dim]History cleared.[/dim]")


SLASH_COMMANDS = {
    "/apply": "Write generated code to local files",
    "/diff": "Show diff of proposed changes",
    "/plan": "Re-display the last plan",
    "/language": "Set language (e.g. /language rust)",
    "/codebase": "Set codebase path (e.g. /codebase ./src)",
    "/iterations": "Set max iterations (e.g. /iterations 3)",
    "/clear": "Clear conversation history",
    "/help": "Show available commands",
    "/exit": "Exit the REPL",
}


def _handle_slash_command(session: CodingSession, line: str) -> bool:
    """Handle a slash command. Returns True if REPL should continue."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit":
        return False
    elif cmd == "/apply":
        session.apply()
    elif cmd == "/diff":
        session.show_diff()
    elif cmd == "/plan":
        session.show_plan()
    elif cmd == "/language":
        if arg:
            session.language = arg.strip()
            console.print(f"Language set to [bold]{session.language}[/bold]")
        else:
            console.print(f"Current language: [bold]{session.language}[/bold]")
    elif cmd == "/codebase":
        if arg:
            session.codebase_path = arg.strip()
            console.print(f"Codebase path set to [bold]{session.codebase_path}[/bold]")
        else:
            console.print(
                f"Current codebase: [bold]{session.codebase_path or '(none)'}[/bold]"
            )
    elif cmd == "/iterations":
        if arg:
            try:
                session.max_iterations = int(arg.strip())
                console.print(
                    f"Max iterations set to [bold]{session.max_iterations}[/bold]"
                )
            except ValueError:
                console.print("[red]Invalid number[/red]")
        else:
            console.print(
                f"Current max iterations: [bold]{session.max_iterations}[/bold]"
            )
    elif cmd == "/clear":
        session.clear()
    elif cmd == "/help":
        for cmd_name, desc in SLASH_COMMANDS.items():
            console.print(f"  [bold]{cmd_name:15s}[/bold] {desc}")
    else:
        console.print(
            f"[red]Unknown command: {cmd}[/red]. Type /help for available commands."
        )

    return True


def run_repl(
    tenant_id: str,
    language: str = "python",
    max_iterations: int = 5,
    codebase_path: str = "",
    runtime_url: str = RUNTIME_URL,
) -> None:
    """Run the interactive coding REPL."""
    try:
        resp = httpx.get(f"{runtime_url}/health", timeout=5.0)
        if resp.status_code != 200:
            console.print("[red]Runtime not healthy. Run `cogniverse up` first.[/red]")
            return
    except (httpx.ConnectError, httpx.ReadTimeout):
        console.print(
            "[red]Cannot connect to runtime. Run `cogniverse up` first.[/red]"
        )
        return

    session = CodingSession(
        tenant_id=tenant_id,
        language=language,
        max_iterations=max_iterations,
        codebase_path=codebase_path,
        runtime_url=runtime_url,
    )

    console.print(
        f"[bold]Cogniverse Coding Agent[/bold] (tenant: {tenant_id}, lang: {language})"
    )
    console.print(
        "[dim]Type a coding task, or /help for commands. Ctrl+D to exit.[/dim]"
    )
    console.print()

    while True:
        try:
            line = console.input("[bold green]>>> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not line:
            continue

        if line.startswith("/"):
            if not _handle_slash_command(session, line):
                console.print("[dim]Goodbye.[/dim]")
                break
            continue

        result = session.send(line)
        if result:
            render_coding_result(result)
        console.print()
