"""E2E tests for knowledge graph extraction and queries.

Tests against the live k3d runtime at localhost:33000:
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

from tests.e2e.conftest import (
    GLINER_URL,
    RUNTIME,
    register_tenant_and_wait,
)
from tests.e2e.test_api_e2e import PROFILE

GRAPH_STATS_URL = f"{RUNTIME}/graph/stats"
GRAPH_UPSERT_URL = f"{RUNTIME}/graph/upsert"
GRAPH_SEARCH_URL = f"{RUNTIME}/graph/search"
GRAPH_NEIGHBORS_URL = f"{RUNTIME}/graph/neighbors"
GRAPH_PATH_URL = f"{RUNTIME}/graph/path"

STATS_FIELDS = {"tenant_id", "node_count", "edge_count", "top_nodes"}

# Vespa indexes upserted nodes asynchronously; under sweep load the stats
# endpoint lags the upsert by more than a fixed sleep, so every stats read
# polls until the tenant reports exactly the counts its upserts produced.
STATS_DEADLINE_S = 60.0
STATS_POLL_S = 2


def _wait_for_stats(
    client: httpx.Client, tenant: str, *, node_count: int, edge_count: int
) -> dict:
    """Poll /graph/stats until the tenant holds exactly the given counts."""
    deadline = time.monotonic() + STATS_DEADLINE_S
    stats: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(GRAPH_STATS_URL, params={"tenant_id": tenant})
        assert resp.status_code == 200, resp.text[:500]
        stats = resp.json()
        assert set(stats) == STATS_FIELDS, stats
        if (stats["node_count"], stats["edge_count"]) == (node_count, edge_count):
            return stats
        time.sleep(STATS_POLL_S)
    raise AssertionError(
        f"/graph/stats for {tenant} never reached node_count={node_count} "
        f"edge_count={edge_count} within {STATS_DEADLINE_S:.0f}s; last: {stats}"
    )


def _degree_table(stats: dict) -> list[tuple[str, int]]:
    """top_nodes as (node_id, degree) sorted so tie order is irrelevant."""
    return sorted(
        ((node["node_id"], node["degree"]) for node in stats["top_nodes"]),
        key=lambda pair: (-pair[1], pair[0]),
    )


def _unique_tenant() -> str:
    """Mint a fresh tenant id, register it, and wait for full readiness.

    Delegates to ``register_tenant_and_wait`` which polls Vespa's
    config-server schemas list (read-after-write consistent with
    prepareandactivate) AND ``GET /admin/tenants/{tid}`` for the
    tenant_metadata search-side row, with a 10-min hard cap. Bare
    tenant_metadata polling alone overruns under sweep load because
    per-tenant deploy is O(N) in the cluster's existing schema count.
    """
    from tests.e2e.conftest import _MINTED_TENANTS_THIS_TEST

    tid = f"graph_e2e_{uuid.uuid4().hex[:8]}"
    _MINTED_TENANTS_THIS_TEST.append(tid)
    register_tenant_and_wait(tid, created_by="graph_e2e_test")
    return tid


@pytest.mark.e2e
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
                        {
                            "source": "EntityA",
                            "target": "EntityB",
                            "relation": "calls",
                            "evidence_span": "EntityA calls EntityB",
                            "segment_id": "module",
                            "ts_start": 0.0,
                            "ts_end": 0.0,
                            "modality": "code",
                            "provenance": "EXTRACTED",
                        },
                        {
                            "source": "EntityB",
                            "target": "EntityC",
                            "relation": "calls",
                            "evidence_span": "EntityB calls EntityC",
                            "segment_id": "module",
                            "ts_start": 0.0,
                            "ts_end": 0.0,
                            "modality": "code",
                            "provenance": "EXTRACTED",
                        },
                    ],
                },
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data == {
                "status": "upserted",
                "nodes_upserted": 3,
                "edges_upserted": 2,
                "failed_ids": [],
            }, data

            stats = _wait_for_stats(client, tenant, node_count=3, edge_count=2)
            assert stats["node_count"] == 3
            assert stats["edge_count"] == 2
            assert _degree_table(stats) == [
                ("entityb", 2),
                ("entitya", 1),
                ("entityc", 1),
            ], stats["top_nodes"]

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
                        {
                            "source": "Alpha",
                            "target": "Beta",
                            "relation": "imports",
                            "evidence_span": "Alpha imports Beta",
                            "segment_id": "module",
                            "ts_start": 0.0,
                            "ts_end": 0.0,
                            "modality": "code",
                            "provenance": "EXTRACTED",
                        },
                        {
                            "source": "Alpha",
                            "target": "Gamma",
                            "relation": "calls",
                            "evidence_span": "Alpha calls Gamma",
                            "segment_id": "module",
                            "ts_start": 0.0,
                            "ts_end": 0.0,
                            "modality": "code",
                            "provenance": "EXTRACTED",
                        },
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
                        {
                            "source": "Start",
                            "target": "Middle",
                            "relation": "calls",
                            "evidence_span": "Start calls Middle",
                            "segment_id": "module",
                            "ts_start": 0.0,
                            "ts_end": 0.0,
                            "modality": "code",
                            "provenance": "EXTRACTED",
                        },
                        {
                            "source": "Middle",
                            "target": "End",
                            "relation": "calls",
                            "evidence_span": "Middle calls End",
                            "segment_id": "module",
                            "ts_start": 0.0,
                            "ts_end": 0.0,
                            "modality": "code",
                            "provenance": "EXTRACTED",
                        },
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
                {
                    "source": "Foo",
                    "target": "Bar",
                    "relation": "refs",
                    # EdgeDoc requires the evidence anchor (the KG
                    # provenance invariant); a bare edge is rejected 422.
                    "evidence_span": "Foo refs Bar",
                    "segment_id": "seg_0",
                    "ts_start": 0.0,
                    "ts_end": 0.0,
                    "modality": "text",
                    "provenance": "INFERRED",
                },
            ],
        }
        expected_upsert = {
            "status": "upserted",
            "nodes_upserted": 2,
            "edges_upserted": 1,
            "failed_ids": [],
        }
        with httpx.Client(timeout=60.0) as client:
            up1 = client.post(GRAPH_UPSERT_URL, json=payload)
            assert up1.status_code == 200, up1.text[:500]
            assert up1.json() == expected_upsert, up1.json()
            first = _wait_for_stats(client, tenant, node_count=2, edge_count=1)
            assert _degree_table(first) == [("bar", 1), ("foo", 1)], first["top_nodes"]

            up2 = client.post(GRAPH_UPSERT_URL, json=payload)
            assert up2.status_code == 200, up2.text[:500]
            assert up2.json() == expected_upsert, up2.json()
            # Re-upserting the same ids changes nothing: same counts, same
            # degrees, once the second feed has been indexed.
            time.sleep(STATS_POLL_S)
            second = _wait_for_stats(client, tenant, node_count=2, edge_count=1)

            assert first["node_count"] == second["node_count"]
            assert first["edge_count"] == second["edge_count"]
            assert _degree_table(first) == _degree_table(second)


@pytest.mark.e2e
class TestMultimodalGraphExtraction:
    """Graph extraction from multimodal content via the ingestion pipeline.

    Uploads a real video file to /ingestion/upload. The runtime processes
    it through the existing Whisper + VLM pipelines, then reads the
    transcript/descriptions and runs the DocExtractor on them to produce
    graph nodes. Verified by reading the response and the /graph/stats
    endpoint.
    """

    def test_video_upload_produces_graph_nodes(self):
        video_path = Path("tests/system/resources/videos/v_-D1gdv_gQyw.mp4")
        assert video_path.exists(), f"Tracked sample video is missing: {video_path}"

        tenant = _unique_tenant()
        with httpx.Client(timeout=1800.0) as client:
            with open(video_path, "rb") as f:
                # wait=true is needed: graph_nodes/graph_edges are only
                # populated in the synchronous response shape. wait_timeout
                # stays under the k3d serverlb's proxy_timeout (600s) —
                # a longer silent hold gets the TCP stream cut mid-wait.
                resp = client.post(
                    f"{RUNTIME}/ingestion/upload?wait=true&wait_timeout=540",
                    files={"file": (video_path.name, f, "video/mp4")},
                    data={
                        "profile": PROFILE,
                        "backend": "vespa",
                        "tenant_id": tenant,
                    },
                )

        assert resp.status_code == 200, (
            f"Video ingestion returned {resp.status_code}: {resp.text[:200]}"
        )

        data = resp.json()
        assert data["status"] == "success", data

        # The worker runs one merged KG upsert per ingest and stamps its
        # nodes_upserted / edges_upserted on the terminal payload; on this fresh
        # tenant those are exactly the documents /graph/stats can see.
        assert type(data["graph_nodes"]) is int, data
        assert type(data["graph_edges"]) is int, data
        with httpx.Client(timeout=60.0) as client:
            stats = _wait_for_stats(
                client,
                tenant,
                node_count=data["graph_nodes"],
                edge_count=data["graph_edges"],
            )
        assert (stats["node_count"], stats["edge_count"]) == (
            data["graph_nodes"],
            data["graph_edges"],
        ), (stats, data)


@pytest.mark.e2e
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
        # CodeExtractor is tree-sitter driven: utils.py yields the module node
        # plus make_greeter / Greeter / __init__ / greet, four `defines` edges,
        # `__init__ calls make_greeter` and `greet calls greeter` (the attribute
        # call normalises onto the Greeter node id).
        assert summary["graph_errors"] == 0, summary
        assert summary["graph_nodes"] == 5, summary
        assert summary["graph_edges"] == 6, summary

        with httpx.Client(timeout=30.0) as client:
            stats = _wait_for_stats(client, tenant, node_count=5, edge_count=6)

        assert stats["node_count"] == 5
        assert stats["edge_count"] == 6
        assert _degree_table(stats) == [
            ("utils", 4),
            ("greet", 2),
            ("greeter", 2),
            ("init", 2),
            ("make_greeter", 2),
        ], stats["top_nodes"]

    def test_index_docs_emits_graph_from_markdown(self):
        from cogniverse_cli.index import index_files

        tenant = _unique_tenant()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "guide.md").write_text(
                "# System Overview\n\n"
                "The ColPali model powers video retrieval in Cogniverse. "
                "It uses Vespa as the storage backend and runs on the configured LM for embeddings. "
                "The SearchAgent orchestrates queries across multiple profiles.\n\n"
                "## Architecture\n\n"
                "ColPali and X-CLIP are the two main encoders supported.\n"
            )

            summary = index_files(
                root=root,
                content_type="docs",
                tenant_id=tenant,
                runtime_url=RUNTIME,
                gliner_url=GLINER_URL,
            )

        assert summary["files_found"] == 1
        assert summary["graph_errors"] == 0, summary
        assert summary["graph_nodes"] >= 2, (
            f"Expected >= 2 graph nodes from markdown, got {summary['graph_nodes']}"
        )

        # GLiNER picks the entities, so the counts are not pinned; what is
        # pinned is that the CLI's reported upsert counts are exactly what the
        # fresh tenant's /graph/stats can see once Vespa has indexed them.
        with httpx.Client(timeout=30.0) as client:
            stats = _wait_for_stats(
                client,
                tenant,
                node_count=summary["graph_nodes"],
                edge_count=summary["graph_edges"],
            )
        node_count = stats["node_count"]

        assert node_count >= 2, (
            f"After 60s, Vespa /graph/stats still shows {node_count} nodes "
            f"despite POST /graph/upsert reporting {summary['graph_nodes']} "
            f"nodes upserted for tenant={tenant}"
        )
        assert node_count == summary["graph_nodes"], (stats, summary)
        assert stats["edge_count"] == summary["graph_edges"], (stats, summary)
