"""Admin-side CLI commands.

Currently exposes ``cogniverse admin reconcile-orphans`` which discovers
Vespa-only schema orphans (in Vespa, not in the SchemaRegistry) and
optionally drops them in one redeploy.

Orphans accumulate from interrupted deploy paths — a SIGKILL between
``backend.deploy_schemas`` and ``register_schema``, a power loss
mid-cleanup, or any code path that bypassed ``POST /admin/tenants``
before the recent ``assert_tenant_exists`` guard was added. The
production-safe recovery is operator-triggered, never automatic.
"""

from __future__ import annotations

import sys

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


def cmd_reconcile_orphans(runtime_url: str, *, confirm: bool) -> int:
    """List orphans (default) or drop them when ``confirm`` is True.

    Returns the process exit code (0 success, non-zero on error).
    """
    url = f"{runtime_url.rstrip('/')}/admin/reconcile-orphans"
    params = {"dry_run": "false" if confirm else "true"}
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, params=params)
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach runtime at {runtime_url}: {exc}[/red]")
        return 2

    if resp.status_code != 200:
        console.print(
            f"[red]reconcile-orphans returned {resp.status_code}: "
            f"{resp.text[:500]}[/red]"
        )
        return 3

    data = resp.json()
    orphan_schemas = data.get("orphan_schemas") or []
    orphan_tenants = data.get("orphan_tenants") or []
    unrecovered = data.get("unrecovered_schemas") or []
    deleted = data.get("deleted") or []

    if not orphan_schemas:
        console.print("[green]No orphan schemas found. Cluster is clean.[/green]")
        return 0

    table = Table(title="Orphan schemas (in Vespa, not in SchemaRegistry)")
    table.add_column("Schema", style="cyan")
    table.add_column("Implied tenant", style="yellow")
    for schema in orphan_schemas:
        implied = ""
        for tid in orphan_tenants:
            suffix = "_" + tid.replace(":", "_")
            if schema.endswith(suffix):
                implied = tid
                break
        table.add_row(schema, implied or "[red]<unrecovered>[/red]")
    console.print(table)

    if unrecovered:
        console.print(
            f"\n[yellow]{len(unrecovered)} schema(s) had unknown base "
            f"prefixes and could not be mapped to a tenant. Review and "
            f"add their base names to the KNOWN_BASES list in "
            f"tenant_manager._list_orphan_schemas() if they are real:[/yellow]"
        )
        for s in unrecovered:
            console.print(f"  {s}")

    if confirm:
        if deleted:
            console.print(
                f"\n[green]Dropped {len(deleted)} schema(s) across "
                f"{len(orphan_tenants)} tenant(s).[/green]"
            )
        else:
            console.print(
                "\n[yellow]Confirm requested but nothing was deleted "
                "(possibly all orphans were unrecovered).[/yellow]"
            )
    else:
        console.print(
            f"\n[cyan]Dry run.[/cyan] Re-run with [bold]--confirm[/bold] "
            f"to drop {len(orphan_tenants)} tenant(s) in one redeploy."
        )
    return 0


def run(runtime_url: str, *, confirm: bool) -> None:
    """Entry point used by the click command in main.py."""
    code = cmd_reconcile_orphans(runtime_url, confirm=confirm)
    if code != 0:
        sys.exit(code)
