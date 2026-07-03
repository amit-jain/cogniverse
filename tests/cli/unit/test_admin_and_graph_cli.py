"""Coverage for the ``cogniverse_cli`` admin and graph subcommands.

The CLI wrappers were untested — only the runtime HTTP routes had
coverage. These tests pin the CLI's behaviour against the HTTP boundary
using ``httpx.MockTransport`` so the request → parse → stdout pipeline
is exercised end to end without a live runtime.
"""

from __future__ import annotations

import io

import cogniverse_cli.admin as admin_cli
import cogniverse_cli.graph as graph_cli
import httpx
import pytest
from rich.console import Console


@pytest.fixture(autouse=True)
def capture_console(monkeypatch: pytest.MonkeyPatch):
    """Route the rich Console output to a StringIO so we can assert on it."""
    buf = io.StringIO()
    test_console = Console(file=buf, width=200, force_terminal=False, color_system=None)
    monkeypatch.setattr(admin_cli, "console", test_console)
    monkeypatch.setattr(graph_cli, "console", test_console)
    return buf


def _mount_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Replace ``httpx.Client`` so the CLI's ``httpx.Client(...)`` uses our
    ``MockTransport`` instead of opening a socket."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _factory)


# ---------------------------------------------------------------------------
# admin reconcile-orphans
# ---------------------------------------------------------------------------


def test_reconcile_orphans_clean_cluster_returns_0(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"orphan_schemas": []})

    _mount_httpx(monkeypatch, handler)
    rc = admin_cli.cmd_reconcile_orphans("http://runtime.test", confirm=False)
    assert rc == 0
    assert "No orphan schemas" in capture_console.getvalue()


def test_reconcile_orphans_lists_orphans_dry_run(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # CLI must send dry_run=true when confirm=False
        assert request.url.params.get("dry_run") == "true"
        return httpx.Response(
            200,
            json={
                "orphan_schemas": ["agent_memories_acme_acme"],
                "orphan_tenants": ["acme:acme"],
                "unrecovered_schemas": [],
                "deleted": [],
            },
        )

    _mount_httpx(monkeypatch, handler)
    rc = admin_cli.cmd_reconcile_orphans("http://runtime.test", confirm=False)
    assert rc == 0
    out = capture_console.getvalue()
    assert "agent_memories_acme_acme" in out
    assert "Dry run" in out
    # The implied-tenant column must populate from the suffix match.
    assert "acme:acme" in out


def test_reconcile_orphans_confirm_sends_dry_run_false(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    seen_dry_run: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_dry_run.append(request.url.params.get("dry_run", ""))
        return httpx.Response(
            200,
            json={
                "orphan_schemas": ["x_t"],
                "orphan_tenants": ["t"],
                "unrecovered_schemas": [],
                "deleted": ["x_t"],
            },
        )

    _mount_httpx(monkeypatch, handler)
    rc = admin_cli.cmd_reconcile_orphans("http://runtime.test", confirm=True)
    assert rc == 0
    assert seen_dry_run == ["false"]
    assert "Dropped 1 schema" in capture_console.getvalue()


def test_reconcile_orphans_connection_error_returns_2(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _mount_httpx(monkeypatch, handler)
    rc = admin_cli.cmd_reconcile_orphans("http://runtime.test", confirm=False)
    assert rc == 2
    assert "Failed to reach runtime" in capture_console.getvalue()


def test_reconcile_orphans_non_200_returns_3(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _mount_httpx(monkeypatch, handler)
    rc = admin_cli.cmd_reconcile_orphans("http://runtime.test", confirm=False)
    assert rc == 3
    out = capture_console.getvalue()
    assert "500" in out
    assert "boom" in out


# ---------------------------------------------------------------------------
# graph subcommands
# ---------------------------------------------------------------------------


def test_graph_stats_renders_counts_and_top_nodes(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/graph/stats"
        assert request.url.params["tenant_id"] == "acme"
        return httpx.Response(
            200,
            json={
                "node_count": 42,
                "edge_count": 100,
                "top_nodes": [
                    {"node_id": "Alice", "degree": 12},
                    {"node_id": "Bob", "degree": 9},
                ],
            },
        )

    _mount_httpx(monkeypatch, handler)
    graph_cli.cmd_stats("acme", runtime_url="http://runtime.test")
    out = capture_console.getvalue()
    assert "Nodes: 42" in out
    assert "Edges: 100" in out
    assert "Alice" in out
    assert "12" in out


def test_graph_search_renders_nodes(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/graph/search"
        assert request.url.params["q"] == "marie"
        return httpx.Response(
            200,
            json={
                "nodes": [
                    {
                        "name": "Marie Curie",
                        "kind": "person",
                        "description": "scientist",
                    }
                ]
            },
        )

    _mount_httpx(monkeypatch, handler)
    graph_cli.cmd_search("acme", "marie", runtime_url="http://runtime.test")
    out = capture_console.getvalue()
    assert "Marie Curie" in out
    assert "person" in out


def test_graph_search_empty_result_prints_yellow_notice(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nodes": []})

    _mount_httpx(monkeypatch, handler)
    graph_cli.cmd_search("acme", "no-such", runtime_url="http://runtime.test")
    assert "No nodes found" in capture_console.getvalue()


def test_graph_stats_non_200_prints_red_error(
    monkeypatch: pytest.MonkeyPatch, capture_console: io.StringIO
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="vespa down")

    _mount_httpx(monkeypatch, handler)
    graph_cli.cmd_stats("acme", runtime_url="http://runtime.test")
    out = capture_console.getvalue()
    assert "Graph stats failed" in out
    assert "500" in out


class TestGraphUpsertPayloadContract:
    def test_cli_payload_validates_against_route_models(self, tmp_path):
        """The CLI serialises extraction results for /graph/upsert — every
        edge field the route's EdgeDoc requires must be present, or the
        whole upsert 422s and `cogniverse index` silently reports zero
        graph nodes."""
        from cogniverse_cli.index import _build_graph_payload

        from cogniverse_agents.graph.code_extractor import CodeExtractor
        from cogniverse_runtime.routers.graph import EdgeDoc, NodeDoc

        f = tmp_path / "utils.py"
        f.write_text(
            "def make_greeter(name):\n"
            "    return lambda: name\n"
            "\n"
            "class Greeter:\n"
            "    def __init__(self, name):\n"
            "        self.greeter = make_greeter(name)\n"
        )
        result = CodeExtractor().extract(f, "acme", "utils.py")
        assert result is not None and result.edges, "extraction must yield edges"

        payload = _build_graph_payload(result, "acme", "utils.py")

        for node in payload["nodes"]:
            NodeDoc.model_validate(node)
        validated = [EdgeDoc.model_validate(edge) for edge in payload["edges"]]
        assert len(validated) == len(result.edges)
        assert all(e.evidence_span for e in validated)
        assert all(e.modality == "code" for e in validated)
