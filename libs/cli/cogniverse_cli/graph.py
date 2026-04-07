"""CLI subcommands for querying the knowledge graph.

``cogniverse graph stats``     — node/edge counts and top-degree nodes
``cogniverse graph search``    — semantic search over graph nodes
``cogniverse graph neighbors`` — direct neighbors of a node
``cogniverse graph path``      — shortest path between two nodes
"""

from typing import Any, Dict, List

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

RUNTIME_URL = "http://localhost:28000"


def _client(runtime_url: str) -> httpx.Client:
    return httpx.Client(timeout=30.0, base_url=runtime_url)


def cmd_stats(tenant_id: str, runtime_url: str = RUNTIME_URL) -> None:
    with _client(runtime_url) as client:
        resp = client.get("/graph/stats", params={"tenant_id": tenant_id})
    if resp.status_code != 200:
        console.print(f"[red]Graph stats failed: {resp.status_code} {resp.text[:200]}[/red]")
        return

    data = resp.json()
    console.print(f"[bold]Knowledge Graph[/bold] (tenant: {tenant_id})")
    console.print(f"  Nodes: [bold cyan]{data['node_count']}[/bold cyan]")
    console.print(f"  Edges: [bold cyan]{data['edge_count']}[/bold cyan]")

    top = data.get("top_nodes", [])
    if top:
        console.print("\n[bold]Top nodes (by degree):[/bold]")
        table = Table(show_header=True)
        table.add_column("Node", style="cyan")
        table.add_column("Degree", justify="right", style="green")
        for entry in top:
            table.add_row(entry["node_id"], str(entry["degree"]))
        console.print(table)


def cmd_search(
    tenant_id: str,
    query: str,
    top_k: int = 10,
    runtime_url: str = RUNTIME_URL,
) -> None:
    with _client(runtime_url) as client:
        resp = client.get(
            "/graph/search",
            params={"tenant_id": tenant_id, "q": query, "top_k": top_k},
        )
    if resp.status_code != 200:
        console.print(f"[red]Graph search failed: {resp.status_code} {resp.text[:200]}[/red]")
        return

    data = resp.json()
    nodes = data.get("nodes", [])
    if not nodes:
        console.print(f"[yellow]No nodes found for '{query}'[/yellow]")
        return

    console.print(f"[bold]Found {len(nodes)} nodes for '{query}':[/bold]\n")
    for node in nodes:
        name = node.get("name", "?")
        kind = node.get("kind", "")
        desc = node.get("description", "")
        console.print(f"  [cyan]{name}[/cyan] [dim]({kind})[/dim]")
        if desc:
            console.print(f"    {desc[:200]}")


def cmd_neighbors(
    tenant_id: str,
    node: str,
    depth: int = 1,
    runtime_url: str = RUNTIME_URL,
) -> None:
    with _client(runtime_url) as client:
        resp = client.get(
            "/graph/neighbors",
            params={"tenant_id": tenant_id, "node": node, "depth": depth},
        )
    if resp.status_code != 200:
        console.print(f"[red]Graph neighbors failed: {resp.status_code} {resp.text[:200]}[/red]")
        return

    data = resp.json()
    console.print(f"[bold]Neighbors of [cyan]{data['name']}[/cyan][/bold]")
    _render_edges("Outgoing", data.get("out_edges", []), "target_node_id")
    _render_edges("Incoming", data.get("in_edges", []), "source_node_id")


def cmd_path(
    tenant_id: str,
    source: str,
    target: str,
    max_depth: int = 4,
    runtime_url: str = RUNTIME_URL,
) -> None:
    with _client(runtime_url) as client:
        resp = client.get(
            "/graph/path",
            params={
                "tenant_id": tenant_id,
                "source": source,
                "target": target,
                "max_depth": max_depth,
            },
        )
    if resp.status_code != 200:
        console.print(f"[red]Graph path failed: {resp.status_code} {resp.text[:200]}[/red]")
        return

    data = resp.json()
    path = data.get("path")
    if not path:
        console.print(f"[yellow]No path found between '{source}' and '{target}' within depth {max_depth}[/yellow]")
        return

    console.print(f"[bold]Path: [cyan]{source}[/cyan] → [cyan]{target}[/cyan][/bold] (length {data['length']})")
    console.print("  " + " → ".join(path))


def _render_edges(label: str, edges: List[Dict[str, Any]], other_key: str) -> None:
    if not edges:
        console.print(f"  [dim]{label}: none[/dim]")
        return
    console.print(f"  [bold]{label}:[/bold]")
    for edge in edges[:20]:
        other = edge.get(other_key, "?")
        relation = edge.get("relation", "?")
        provenance = edge.get("provenance", "")
        console.print(f"    → [cyan]{other}[/cyan] [dim]({relation}, {provenance})[/dim]")
    if len(edges) > 20:
        console.print(f"    [dim]… and {len(edges) - 20} more[/dim]")
