"""CLI subcommands for querying the knowledge graph.

``cogniverse graph stats``     — node/edge counts and top-degree nodes
``cogniverse graph search``    — semantic search over graph nodes
``cogniverse graph neighbors`` — direct neighbors of a node
``cogniverse graph path``      — shortest path between two nodes

Each command returns a process exit code: 0 on success, 2 when the runtime
is unreachable, 3 on a non-200 or unusable response — matching
``cogniverse admin`` so scripts can branch on failure instead of parsing
a traceback.
"""

from typing import Any, Dict, List, Optional, Tuple

import httpx
from rich.console import Console
from rich.table import Table

from cogniverse_cli.constants import RUNTIME_URL

console = Console()


def _client(runtime_url: str) -> httpx.Client:
    return httpx.Client(timeout=30.0, base_url=runtime_url)


def _fetch(
    path: str, params: Dict[str, Any], runtime_url: str, label: str
) -> Tuple[Optional[Dict[str, Any]], int]:
    """GET a graph route; return (data, 0) or (None, exit_code) with the
    failure already printed."""
    try:
        with _client(runtime_url) as client:
            resp = client.get(path, params=params)
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach runtime at {runtime_url}: {exc}[/red]")
        return None, 2

    if resp.status_code != 200:
        console.print(
            f"[red]{label} failed: {resp.status_code} {resp.text[:200]}[/red]"
        )
        return None, 3

    try:
        data = resp.json()
    except ValueError:
        console.print(f"[red]{label} returned a non-JSON response[/red]")
        return None, 3
    if not isinstance(data, dict):
        console.print(f"[red]{label} returned an unexpected response shape[/red]")
        return None, 3
    return data, 0


def cmd_stats(tenant_id: str, runtime_url: str = RUNTIME_URL) -> int:
    data, code = _fetch(
        "/graph/stats", {"tenant_id": tenant_id}, runtime_url, "Graph stats"
    )
    if data is None:
        return code

    console.print(f"[bold]Knowledge Graph[/bold] (tenant: {tenant_id})")
    console.print(f"  Nodes: [bold cyan]{data.get('node_count', 0)}[/bold cyan]")
    console.print(f"  Edges: [bold cyan]{data.get('edge_count', 0)}[/bold cyan]")

    top = data.get("top_nodes", [])
    if top:
        console.print("\n[bold]Top nodes (by degree):[/bold]")
        table = Table(show_header=True)
        table.add_column("Node", style="cyan")
        table.add_column("Degree", justify="right", style="green")
        for entry in top:
            table.add_row(str(entry.get("node_id", "?")), str(entry.get("degree", "?")))
        console.print(table)
    return 0


def cmd_search(
    tenant_id: str,
    query: str,
    top_k: int = 10,
    runtime_url: str = RUNTIME_URL,
) -> int:
    data, code = _fetch(
        "/graph/search",
        {"tenant_id": tenant_id, "q": query, "top_k": top_k},
        runtime_url,
        "Graph search",
    )
    if data is None:
        return code

    nodes = data.get("nodes", [])
    if not nodes:
        console.print(f"[yellow]No nodes found for '{query}'[/yellow]")
        return 0

    console.print(f"[bold]Found {len(nodes)} nodes for '{query}':[/bold]\n")
    for node in nodes:
        name = node.get("name", "?")
        kind = node.get("kind", "")
        desc = node.get("description", "")
        console.print(f"  [cyan]{name}[/cyan] [dim]({kind})[/dim]")
        if desc:
            console.print(f"    {desc[:200]}")
    return 0


def cmd_neighbors(
    tenant_id: str,
    node: str,
    depth: int = 1,
    runtime_url: str = RUNTIME_URL,
) -> int:
    data, code = _fetch(
        "/graph/neighbors",
        {"tenant_id": tenant_id, "node": node, "depth": depth},
        runtime_url,
        "Graph neighbors",
    )
    if data is None:
        return code

    console.print(f"[bold]Neighbors of [cyan]{data.get('name', node)}[/cyan][/bold]")
    _render_edges("Outgoing", data.get("out_edges", []), "target_node_id")
    _render_edges("Incoming", data.get("in_edges", []), "source_node_id")
    return 0


def cmd_path(
    tenant_id: str,
    source: str,
    target: str,
    max_depth: int = 4,
    runtime_url: str = RUNTIME_URL,
) -> int:
    data, code = _fetch(
        "/graph/path",
        {
            "tenant_id": tenant_id,
            "source": source,
            "target": target,
            "max_depth": max_depth,
        },
        runtime_url,
        "Graph path",
    )
    if data is None:
        return code

    path = data.get("path")
    if not path:
        console.print(
            f"[yellow]No path found between '{source}' and '{target}' "
            f"within depth {max_depth}[/yellow]"
        )
        return 0

    length = data.get("length", len(path) - 1)
    console.print(
        f"[bold]Path: [cyan]{source}[/cyan] → [cyan]{target}[/cyan][/bold] "
        f"(length {length})"
    )
    console.print("  " + " → ".join(str(p) for p in path))
    return 0


def _render_edges(label: str, edges: List[Dict[str, Any]], other_key: str) -> None:
    if not edges:
        console.print(f"  [dim]{label}: none[/dim]")
        return
    console.print(f"  [bold]{label}:[/bold]")
    for edge in edges[:20]:
        other = edge.get(other_key, "?")
        relation = edge.get("relation", "?")
        provenance = edge.get("provenance", "")
        console.print(
            f"    → [cyan]{other}[/cyan] [dim]({relation}, {provenance})[/dim]"
        )
    if len(edges) > 20:
        console.print(f"    [dim]… and {len(edges) - 20} more[/dim]")
