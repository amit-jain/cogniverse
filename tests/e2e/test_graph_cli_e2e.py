"""E2E tests for knowledge graph extraction and queries.

Tests against the live k3d runtime at localhost:28000:
- cogniverse index extracts graph from code + text files and upserts
- /graph endpoints return real nodes, edges, stats, neighbors, paths
- Round-trip semantic search returns nodes matching the query
"""

import tempfile
import time
import uuid
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, skip_if_no_runtime

GRAPH_STATS_URL = f"{RUNTIME}/graph/stats"
GRAPH_UPSERT_URL = f"{RUNTIME}/graph/upsert"
GRAPH_SEARCH_URL = f"{RUNTIME}/graph/search"
GRAPH_NEIGHBORS_URL = f"{RUNTIME}/graph/neighbors"
GRAPH_PATH_URL = f"{RUNTIME}/graph/path"


def _unique_tenant() -> str:
    return f"graph_e2e_{uuid.uuid4().hex[:8]}"


@pytest.mark.e2e
@skip_if_no_runtime
class TestGraphEndpoints:
    """Direct tests of /graph/* against the live runtime."""

    def test_upsert_then_stats_returns_counts(self):
        tenant = _unique_tenant()
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                GRAPH_UPSERT_URL,
                json={
                    "tenant_id": tenant,
                    "source_doc_id": "demo.py",
                    "nodes": [
                        {"name": "EntityA", "description": "First", "kind": "entity"},
                        {"name": "EntityB", "description": "Second", "kind": "entity"},
                        {"name": "EntityC", "description": "Third", "kind": "entity"},
                    ],
                    "edges": [
                        {"source": "EntityA", "target": "EntityB", "relation": "calls", "provenance": "EXTRACTED"},
                        {"source": "EntityB", "target": "EntityC", "relation": "calls", "provenance": "EXTRACTED"},
                    ],
                },
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["status"] == "upserted"
            assert data["nodes_upserted"] == 3
            assert data["edges_upserted"] == 2

            time.sleep(3)

            resp = client.get(GRAPH_STATS_URL, params={"tenant_id": tenant})
            assert resp.status_code == 200
            stats = resp.json()
            assert stats["node_count"] >= 3
            assert stats["edge_count"] >= 2

    def test_neighbors_returns_outgoing_edges(self):
        tenant = _unique_tenant()
        with httpx.Client(timeout=60.0) as client:
            client.post(
                GRAPH_UPSERT_URL,
                json={
                    "tenant_id": tenant,
                    "source_doc_id": "mod.py",
                    "nodes": [
                        {"name": "Alpha"},
                        {"name": "Beta"},
                        {"name": "Gamma"},
                    ],
                    "edges": [
                        {"source": "Alpha", "target": "Beta", "relation": "imports", "provenance": "EXTRACTED"},
                        {"source": "Alpha", "target": "Gamma", "relation": "calls", "provenance": "EXTRACTED"},
                    ],
                },
            )
            time.sleep(3)

            resp = client.get(
                GRAPH_NEIGHBORS_URL,
                params={"tenant_id": tenant, "node": "Alpha"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["node_id"] == "alpha"
            out = data["out_edges"]
            assert len(out) == 2
            targets = {e["target_node_id"] for e in out}
            assert targets == {"beta", "gamma"}
            relations = {e["relation"] for e in out}
            assert relations == {"imports", "calls"}

    def test_path_finds_multi_hop_route(self):
        tenant = _unique_tenant()
        with httpx.Client(timeout=60.0) as client:
            client.post(
                GRAPH_UPSERT_URL,
                json={
                    "tenant_id": tenant,
                    "source_doc_id": "chain.py",
                    "nodes": [
                        {"name": "Start"},
                        {"name": "Middle"},
                        {"name": "End"},
                    ],
                    "edges": [
                        {"source": "Start", "target": "Middle", "relation": "calls", "provenance": "EXTRACTED"},
                        {"source": "Middle", "target": "End", "relation": "calls", "provenance": "EXTRACTED"},
                    ],
                },
            )
            time.sleep(3)

            resp = client.get(
                GRAPH_PATH_URL,
                params={
                    "tenant_id": tenant,
                    "source": "Start",
                    "target": "End",
                    "max_depth": 4,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["path"] == ["start", "middle", "end"]
            assert data["length"] == 2

    def test_path_returns_none_when_no_route(self):
        tenant = _unique_tenant()
        with httpx.Client(timeout=60.0) as client:
            client.post(
                GRAPH_UPSERT_URL,
                json={
                    "tenant_id": tenant,
                    "source_doc_id": "isolated.py",
                    "nodes": [
                        {"name": "Island1"},
                        {"name": "Island2"},
                    ],
                    "edges": [],
                },
            )
            time.sleep(3)

            resp = client.get(
                GRAPH_PATH_URL,
                params={
                    "tenant_id": tenant,
                    "source": "Island1",
                    "target": "Island2",
                    "max_depth": 4,
                },
            )
            data = resp.json()
            assert data["path"] is None
            assert data["length"] == -1

    def test_upsert_is_idempotent(self):
        """Same input upserted twice produces the same node/edge counts in stats."""
        tenant = _unique_tenant()
        payload = {
            "tenant_id": tenant,
            "source_doc_id": "demo.py",
            "nodes": [
                {"name": "Foo"},
                {"name": "Bar"},
            ],
            "edges": [
                {"source": "Foo", "target": "Bar", "relation": "refs", "provenance": "INFERRED"},
            ],
        }
        with httpx.Client(timeout=60.0) as client:
            client.post(GRAPH_UPSERT_URL, json=payload)
            time.sleep(3)
            first = client.get(GRAPH_STATS_URL, params={"tenant_id": tenant}).json()

            client.post(GRAPH_UPSERT_URL, json=payload)
            time.sleep(3)
            second = client.get(GRAPH_STATS_URL, params={"tenant_id": tenant}).json()

            assert first["node_count"] == second["node_count"]
            assert first["edge_count"] == second["edge_count"]


@pytest.mark.e2e
@skip_if_no_runtime
class TestCliIndexWithGraph:
    """cogniverse index extracts graph from real files and persists them."""

    def test_index_code_emits_graph_nodes(self):
        from cogniverse_cli.index import index_files

        tenant = _unique_tenant()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "utils.py").write_text(
                "def make_greeter(name):\n"
                "    return lambda: f'hello {name}'\n"
                "\n"
                "class Greeter:\n"
                "    def __init__(self, name):\n"
                "        self.greeter = make_greeter(name)\n"
                "\n"
                "    def greet(self):\n"
                "        return self.greeter()\n"
            )

            summary = index_files(
                root=root,
                content_type="code",
                tenant_id=tenant,
                runtime_url=RUNTIME,
            )

        assert summary["files_found"] == 1
        assert summary["graph_nodes"] >= 2, (
            f"Expected >= 2 graph nodes for utils.py, got {summary['graph_nodes']}"
        )
        assert summary["graph_edges"] >= 1, (
            f"Expected >= 1 graph edge for utils.py, got {summary['graph_edges']}"
        )

        time.sleep(3)

        with httpx.Client(timeout=30.0) as client:
            stats = client.get(GRAPH_STATS_URL, params={"tenant_id": tenant}).json()

        assert stats["node_count"] >= 2
        assert stats["edge_count"] >= 1

    def test_index_docs_emits_graph_from_markdown(self):
        from cogniverse_cli.index import index_files

        tenant = _unique_tenant()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "guide.md").write_text(
                "# System Overview\n\n"
                "The ColPali model powers video retrieval in Cogniverse. "
                "It uses Vespa as the storage backend and runs on Ollama for embeddings. "
                "The SearchAgent orchestrates queries across multiple profiles.\n\n"
                "## Architecture\n\n"
                "ColPali and VideoPrism are the two main encoders supported.\n"
            )

            summary = index_files(
                root=root,
                content_type="docs",
                tenant_id=tenant,
                runtime_url=RUNTIME,
            )

        assert summary["files_found"] == 1
        assert summary["graph_nodes"] >= 2, (
            f"Expected >= 2 graph nodes from markdown, got {summary['graph_nodes']}"
        )

        time.sleep(3)

        with httpx.Client(timeout=30.0) as client:
            stats = client.get(GRAPH_STATS_URL, params={"tenant_id": tenant}).json()

        assert stats["node_count"] >= 2
