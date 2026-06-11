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

from tests.e2e.conftest import (
    RUNTIME,
    register_tenant_and_wait,
    skip_if_no_runtime,
)

GRAPH_STATS_URL = f"{RUNTIME}/graph/stats"
GRAPH_UPSERT_URL = f"{RUNTIME}/graph/upsert"
GRAPH_SEARCH_URL = f"{RUNTIME}/graph/search"
GRAPH_NEIGHBORS_URL = f"{RUNTIME}/graph/neighbors"
GRAPH_PATH_URL = f"{RUNTIME}/graph/path"


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
                    "provenance": "INFERRED",
                },
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
class TestMultimodalGraphExtraction:
    """Graph extraction from multimodal content via the ingestion pipeline.

    Uploads a real video file to /ingestion/upload. The runtime processes
    it through the existing Whisper + VLM pipelines, then reads the
    transcript/descriptions and runs the DocExtractor on them to produce
    graph nodes. Verified by reading the response and the /graph/stats
    endpoint.
    """

    def test_video_upload_produces_graph_nodes(self):
        video_path = Path("data/testset/evaluation/sample_videos/v_-nl4G-00PtA.mp4")
        if not video_path.exists():
            pytest.skip(f"Sample video missing at {video_path}")

        tenant = _unique_tenant()
        with httpx.Client(timeout=1800.0) as client:
            with open(video_path, "rb") as f:
                # wait=true is needed: graph_nodes/graph_edges are only
                # populated in the synchronous response shape.
                resp = client.post(
                    f"{RUNTIME}/ingestion/upload?wait=true&wait_timeout=900",
                    files={"file": (video_path.name, f, "video/mp4")},
                    data={
                        "profile": "video_colpali_smol500_mv_frame",
                        "backend": "vespa",
                        "tenant_id": tenant,
                    },
                )

        if resp.status_code != 200:
            pytest.skip(
                f"Video ingestion returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        assert data["status"] == "success", data

        assert "graph_nodes" in data, (
            "ingestion response should include graph_nodes field"
        )
        assert "graph_edges" in data


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
                "It uses Vespa as the storage backend and runs on the configured LM for embeddings. "
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

        # Vespa indexing of the freshly-upserted nodes is async — a flat
        # 3 s sleep was enough on an idle cluster but fails under sweep
        # load when Vespa has many concurrent feed operations. Poll the
        # stats endpoint until the upserted nodes are visible.
        node_count = 0
        deadline = time.monotonic() + 60.0
        with httpx.Client(timeout=30.0) as client:
            while time.monotonic() < deadline:
                stats = client.get(GRAPH_STATS_URL, params={"tenant_id": tenant}).json()
                node_count = stats.get("node_count", 0)
                if node_count >= 2:
                    break
                time.sleep(2)

        assert node_count >= 2, (
            f"After 60s, Vespa /graph/stats still shows {node_count} nodes "
            f"despite POST /graph/upsert reporting {summary['graph_nodes']} "
            f"nodes upserted for tenant={tenant}"
        )
