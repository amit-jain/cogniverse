"""Admin-side CLI commands.

``cogniverse admin reconcile-orphans`` reports two orphan classes and
optionally drops each in one redeploy: registry-orphans (deployed in
Vespa, absent from the SchemaRegistry) and tenant-orphans (deployed and
registered, but the registry row names a tenant with no tenant_metadata
record, so they ride along in every application package).

Registry-orphans accumulate from interrupted deploy paths — a SIGKILL
between ``backend.deploy_schemas`` and ``register_schema``, a power loss
mid-cleanup. Tenant-orphans accumulate from schema auto-deploy paths that
never created a tenant (memory lazy-init, ingestion upload) and from
tenant records removed outside ``DELETE /admin/tenants``. Recovery is
operator-triggered, never automatic.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


def cmd_reconcile_orphans(
    runtime_url: str, *, confirm: bool, tenant_orphans: bool = False
) -> int:
    """List orphans (default) or drop them when ``confirm`` is True.

    ``tenant_orphans`` additionally drops the schemas of tenants with no
    tenant_metadata record; it needs ``confirm`` to take effect and the
    runtime refuses an empty selection with 409.

    Returns the process exit code (0 success, non-zero on error).
    """
    url = f"{runtime_url.rstrip('/')}/admin/reconcile-orphans"
    params = {
        "dry_run": "false" if confirm else "true",
        "remove_tenant_orphans": "true" if tenant_orphans else "false",
        "include_document_counts": "true",
    }
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
    try:
        tenant_orphan_schemas = data["tenant_orphan_schemas"]
        tenant_orphan_tenants = data["tenant_orphan_tenants"]
        tenant_orphans_deleted = data["tenant_orphans_deleted"]
        orphan_details = data["orphan_details"]
    except KeyError as exc:
        console.print(f"[red]reconcile-orphans response is missing {exc}[/red]")
        return 3

    if not orphan_schemas and not tenant_orphan_schemas:
        console.print("[green]No orphan schemas found. Cluster is clean.[/green]")
        return 0

    if tenant_orphan_schemas:
        tenant_table = Table(
            title="Tenant-orphan schemas (registered, but the tenant is gone)"
        )
        tenant_table.add_column("Schema", style="cyan")
        tenant_table.add_column("Owning tenant", style="yellow")
        tenant_table.add_column("Tenant exists?")
        tenant_table.add_column("Documents", justify="right")
        for row in orphan_details:
            tenant_table.add_row(
                row["schema"],
                row["tenant"],
                str(row["tenant_exists"]),
                str(row["document_count"]),
            )
        console.print(tenant_table)
        if tenant_orphans_deleted:
            console.print(
                f"[green]Dropped {len(tenant_orphans_deleted)} tenant-orphan "
                f"schema(s) across {len(tenant_orphan_tenants)} tenant(s).[/green]"
            )
        else:
            console.print(
                "[cyan]Re-run with[/cyan] [bold]--confirm --tenant-orphans[/bold] "
                "[cyan]to drop them.[/cyan]"
            )

    if not orphan_schemas:
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


def run(runtime_url: str, *, confirm: bool, tenant_orphans: bool = False) -> None:
    """Entry point used by the click command in main.py."""
    code = cmd_reconcile_orphans(
        runtime_url, confirm=confirm, tenant_orphans=tenant_orphans
    )
    if code != 0:
        sys.exit(code)


def cmd_create_invite(
    runtime_url: str, tenant_id: str, *, expires_in_hours: int
) -> int:
    """Mint a messaging invite token for ``tenant_id``.

    Prints the token and the exact ``/start <token>`` line the operator hands
    to the user, who sends it to the bot to link their chat account to the
    tenant. Returns the process exit code.
    """
    url = f"{runtime_url.rstrip('/')}/admin/messaging/invite"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                url,
                json={"tenant_id": tenant_id, "expires_in_hours": expires_in_hours},
            )
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach runtime at {runtime_url}: {exc}[/red]")
        return 2

    if resp.status_code != 200:
        console.print(
            f"[red]invite returned {resp.status_code}: {resp.text[:500]}[/red]"
        )
        return 3

    data = resp.json()
    token = data.get("token")
    if not token:
        console.print(f"[red]Runtime returned no token: {data}[/red]")
        return 4

    console.print(
        f"[green]Invite token for [bold]{data.get('tenant_id')}[/bold][/green]"
    )
    console.print(f"  token:      [cyan]{token}[/cyan]")
    console.print(f"  expires in: {expires_in_hours}h")
    console.print(f"\nSend this to the bot:\n  [bold]/start {token}[/bold]")
    return 0


def run_invite(runtime_url: str, tenant_id: str, *, expires_in_hours: int) -> None:
    """Entry point used by the click command in main.py."""
    code = cmd_create_invite(runtime_url, tenant_id, expires_in_hours=expires_in_hours)
    if code != 0:
        sys.exit(code)


def cmd_merge_article_nodes(
    runtime_url: str,
    *,
    apply: bool,
    tenant: Optional[str] = None,
    exclude: Sequence[str] = (),
) -> int:
    """Report (default) or apply the article-node merge per tenant.

    Calls ``POST /admin/graph/merge-article-nodes``. Returns the process exit
    code: 2 when the runtime is unreachable, 3 on a non-200 response.
    """
    url = f"{runtime_url.rstrip('/')}/admin/graph/merge-article-nodes"
    params = [("dry_run", "false" if apply else "true")]
    if tenant:
        params.append(("tenant_id", tenant))
    params.extend(("exclude", article_id) for article_id in exclude)
    try:
        with httpx.Client(timeout=3600.0) as client:
            resp = client.post(url, params=params)
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach runtime at {runtime_url}: {exc}[/red]")
        return 2

    if resp.status_code != 200:
        console.print(
            f"[red]merge-article-nodes returned {resp.status_code}: "
            f"{resp.text[:500]}[/red]"
        )
        return 3

    tenants = resp.json().get("tenants") or []
    merged = 0
    for report in tenants:
        merges = report.get("merges") or []
        skipped = report.get("skipped") or []
        excluded = report.get("excluded") or []
        merged += len(merges)
        if merges:
            table = Table(title=f"Article-node merges for {report['tenant_id']}")
            table.add_column("From", style="cyan")
            table.add_column("Into", style="green")
            table.add_column("Node doc")
            table.add_column("Edges re-pointed", justify="right")
            table.add_column("Edges deduped", justify="right")
            for merge in merges:
                table.add_row(
                    merge["from"],
                    merge["into"],
                    str(merge["node_doc"]),
                    str(merge["edges_repointed"]),
                    str(merge["edges_deduped"]),
                )
            console.print(table)
            console.print(
                f"Edges re-pointed: {report['edges_repointed']}, "
                f"deduped: {report['edges_deduped']}; "
                f"mentions added: {report['mentions_added']}; "
                f"content docs updated: {report['content_docs_updated']}"
            )
        else:
            console.print(f"{report['tenant_id']}: no article nodes to merge")
        if skipped:
            console.print(f"Skipped (no bare twin node): {', '.join(skipped)}")
        if excluded:
            console.print(f"Excluded (never merged): {', '.join(excluded)}")

    if apply:
        console.print(
            f"[green]Merged {merged} article node(s) across "
            f"{len(tenants)} tenant(s).[/green]"
        )
    elif merged:
        console.print("[cyan]Dry run. Re-run with --apply to merge.[/cyan]")
    return 0


def run_merge_article_nodes(
    runtime_url: str,
    *,
    apply: bool,
    tenant: Optional[str] = None,
    exclude: Sequence[str] = (),
) -> None:
    """Entry point used by the click command in main.py."""
    code = cmd_merge_article_nodes(
        runtime_url, apply=apply, tenant=tenant, exclude=exclude
    )
    if code != 0:
        sys.exit(code)
