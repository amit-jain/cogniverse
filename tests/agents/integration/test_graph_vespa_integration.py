"""Integration tests for GraphManager against a real Vespa Docker container.

Backed by the project-wide ``shared_memory_vespa`` container; the
``graph_vespa`` fixture co-deploys the tenant-scoped knowledge_graph schema
through the canonical SchemaRegistry pathway and exercises the full
extract → upsert → query round-trip against real multi-vector embeddings.
"""

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import requests

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Mention,
    Node,
    normalize_name,
)
from tests.utils.docker_utils import generate_unique_ports
from tests.utils.vespa_test_helpers import schema_full_name


def _test_mention(
    source_doc_id: str = "test.py",
    segment_id: str = "module",
    modality: str = "code",
    evidence_span: str = "test evidence",
) -> Mention:
    return Mention(
        source_doc_id=source_doc_id,
        segment_id=segment_id,
        ts_start=0.0,
        ts_end=0.0,
        modality=modality,
        evidence_span=evidence_span,
    )


TENANT_ID = "test_tenant"
# Must equal the document type the canonical SchemaRegistry pathway actually
# deploys (see deploy_tenant_schema → deploy_schema), which canonicalizes the
# tenant id and so double-suffixes: knowledge_graph_test_tenant_test_tenant.
GRAPH_SCHEMA = schema_full_name("knowledge_graph", TENANT_ID)
CONTAINER_NAME = "vespa-graph-integration-tests"

_HTTP_PORT, _CONFIG_PORT = generate_unique_ports(__name__)


def _doc_url(port: int, doc_id: str) -> str:
    return (
        f"http://localhost:{port}/document/v1"
        f"/graph_content/{GRAPH_SCHEMA}/docid/{doc_id}"
    )


def _get_vespa_doc(port: int, doc_id: str, retries: int = 15):
    for _ in range(retries):
        try:
            resp = requests.get(_doc_url(port, doc_id), timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        time.sleep(1)
    return None


def _wait_for_config_port(config_port: int, timeout: int = 120) -> bool:
    for _ in range(timeout):
        try:
            resp = requests.get(
                f"http://localhost:{config_port}/ApplicationStatus", timeout=2
            )
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _wait_for_data_port(http_port: int, timeout: int = 120) -> bool:
    for _ in range(timeout):
        try:
            resp = requests.get(
                f"http://localhost:{http_port}/ApplicationStatus", timeout=5
            )
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _wait_for_schema_ready(
    http_port: int, schema_name: str, timeout: int = 120
) -> bool:
    """Feed a minimal probe document to confirm the schema accepts writes."""
    probe = {
        "fields": {
            "doc_id": "readiness_check",
            "tenant_id": "test",
            "doc_type": "node",
            "name": "probe",
            "description": "probe",
            "kind": "concept",
            "mentions": "[]",
            "degree": 0,
            "source_node_id": "",
            "target_node_id": "",
            "relation": "",
            "provenance": "",
            "source_doc_id": "",
            "confidence": 0.0,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    url = (
        f"http://localhost:{http_port}/document/v1/graph_content/"
        f"{schema_name}/docid/readiness_check"
    )
    for i in range(timeout):
        try:
            resp = requests.post(url, json=probe, timeout=5)
            if resp.status_code in (200, 201):
                requests.delete(url, timeout=5)
                return True
            if i % 10 == 0:
                print(
                    f"   readiness attempt {i + 1}: {resp.status_code} {resp.text[:100]}"
                )
        except Exception as exc:
            if i % 10 == 0:
                print(f"   readiness attempt {i + 1}: {exc}")
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def graph_vespa(shared_memory_vespa):
    """Module-scoped graph-schema fixture backed by the project-wide
    ``shared_vespa`` (re-exported via shared_memory_vespa).

    Deploys ``knowledge_graph_test_tenant`` via SchemaRegistry —
    merge-safe alongside the other tenant schemas already on
    shared_vespa. Yields the same ``{http_port, config_port}`` shape
    consumers expect.
    """
    from tests.utils.vespa_test_helpers import deploy_tenant_schema

    deploy_tenant_schema(
        shared_memory_vespa,
        tenant_id=TENANT_ID,
        base_schema_name="knowledge_graph",
        config_manager=shared_memory_vespa["config_manager"],
    )

    http_port = shared_memory_vespa["http_port"]

    if not _wait_for_schema_ready(http_port, GRAPH_SCHEMA, timeout=120):
        pytest.fail(f"Schema {GRAPH_SCHEMA} not ready within 120s")

    yield {
        "http_port": http_port,
        "config_port": shared_memory_vespa["config_port"],
    }
    # No teardown — shared_vespa owns the container; the deployed
    # knowledge_graph_test_tenant schema stays around until session end.


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def pylate_server():
    """Run the real colbert_pylate sidecar module in a background thread.

    Loads ``lightonai/LateOn`` via PyLate (downloads ~300MB on first
    run, cached in HF home thereafter) and serves the same /pooling
    contract the production sidecar speaks. The graph_manager fixture
    points GraphManager at this URL so upserts exercise the full
    encode → VespaEmbeddingProcessor → Vespa write path against real
    multi-vector embeddings.
    """
    import importlib.util

    import uvicorn

    # Load the production sidecar module by path (no package install).
    spec = importlib.util.spec_from_file_location(
        "pylate_server_under_test",
        "libs/runtime/cogniverse_runtime/sidecars/colbert_pylate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    app = mod.build_app(model_name="lightonai/LateOn", device="cpu", mode="colbert")
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180  # model load on first run is slow
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("pylate /health did not come up within 180s — model load failed")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def graph_manager(graph_vespa, pylate_server):
    """GraphManager wired to the test Vespa via the production VespaBackend
    (BackendRegistry path) and the real PyLate /pooling server."""
    from pathlib import Path

    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    http_port = graph_vespa["http_port"]
    config_port = graph_vespa["config_port"]

    BackendRegistry._backend_instances.clear()
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=http_port,
    )
    config_manager = ConfigManager(store=config_store)
    config_manager.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=http_port)
    )
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=TENANT_ID,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": config_port,
                "port": http_port,
            }
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    manager = GraphManager(
        backend=backend,
        tenant_id=TENANT_ID,
        schema_name=GRAPH_SCHEMA,
        colbert_endpoint_url=pylate_server,
    )
    try:
        yield manager, http_port
    finally:
        BackendRegistry._backend_instances.clear()


@pytest.mark.integration
class TestGraphVespaUpsert:
    """Upsert nodes/edges and verify they're retrievable from real Vespa."""

    def test_upsert_nodes_persists_to_vespa(self, graph_manager):
        manager, port = graph_manager

        result = ExtractionResult(
            source_doc_id="test_upsert.py",
            nodes=[
                Node(
                    tenant_id=TENANT_ID,
                    name="IntegrationAlpha",
                    kind="entity",
                    description="First integration node",
                    mentions=[_test_mention(source_doc_id="test_upsert.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="IntegrationBeta",
                    kind="entity",
                    description="Second integration node",
                    mentions=[_test_mention(source_doc_id="test_upsert.py")],
                ),
            ],
            edges=[],
        )
        counts = manager.upsert(result)
        assert counts["nodes_upserted"] == 2

        doc_id_alpha = "kg_node_test_tenant_integrationalpha"
        doc = _get_vespa_doc(port, doc_id_alpha)
        assert doc is not None, f"Node {doc_id_alpha} not in Vespa after upsert"
        fields = doc.get("fields", {})
        assert fields.get("doc_type") == "node"
        assert fields.get("tenant_id") == TENANT_ID
        assert fields.get("name") == "IntegrationAlpha"
        assert fields.get("kind") == "entity"
        assert fields.get("description") == "First integration node"

    def test_upsert_edges_persists_to_vespa(self, graph_manager):
        manager, port = graph_manager

        result = ExtractionResult(
            source_doc_id="test_edges.py",
            nodes=[
                Node(
                    tenant_id=TENANT_ID,
                    name="EdgeSource",
                    kind="entity",
                    mentions=[_test_mention(source_doc_id="test_edges.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="EdgeTarget",
                    kind="entity",
                    mentions=[_test_mention(source_doc_id="test_edges.py")],
                ),
            ],
            edges=[
                Edge(
                    tenant_id=TENANT_ID,
                    source="EdgeSource",
                    target="EdgeTarget",
                    relation="calls",
                    evidence_span="EdgeSource calls EdgeTarget()",
                    segment_id="function:test",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    provenance="EXTRACTED",
                    source_doc_id="test_edges.py",
                ),
            ],
        )
        counts = manager.upsert(result)
        assert counts["edges_upserted"] == 1

        edges = manager._visit_edges(source_node_id="edgesource")
        assert len(edges) == 1
        edge = edges[0]
        assert edge.get("relation") == "calls"
        assert edge.get("provenance") == "EXTRACTED"
        assert edge.get("target_node_id") == "edgetarget"

    def test_upsert_merges_duplicate_nodes(self, graph_manager):
        """Two nodes with the same normalized name get merged into one document
        whose mentions are the union of both inputs.
        """
        manager, port = graph_manager

        result = ExtractionResult(
            source_doc_id="first.py",
            nodes=[
                Node(
                    tenant_id=TENANT_ID,
                    name="MergeMe",
                    mentions=[
                        _test_mention(source_doc_id="first.py", segment_id="module")
                    ],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="mergeme",
                    mentions=[
                        _test_mention(source_doc_id="second.py", segment_id="module")
                    ],
                ),
            ],
            edges=[],
        )
        counts = manager.upsert(result)
        assert counts["nodes_upserted"] == 1

        doc = _get_vespa_doc(port, "kg_node_test_tenant_mergeme")
        assert doc is not None
        mentions_json = doc.get("fields", {}).get("mentions", "[]")
        mentions = json.loads(mentions_json)
        source_doc_ids = {m["source_doc_id"] for m in mentions}
        assert source_doc_ids == {"first.py", "second.py"}


@pytest.mark.integration
class TestGraphVespaQueries:
    """Test neighbors/path/stats against real Vespa."""

    def test_neighbors_returns_outgoing_and_incoming(self, graph_manager):
        manager, _ = graph_manager

        result = ExtractionResult(
            source_doc_id="neighbors_test.py",
            nodes=[
                Node(
                    tenant_id=TENANT_ID,
                    name="NeighborHub",
                    mentions=[_test_mention(source_doc_id="neighbors_test.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="NeighborOut1",
                    mentions=[_test_mention(source_doc_id="neighbors_test.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="NeighborOut2",
                    mentions=[_test_mention(source_doc_id="neighbors_test.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="NeighborIn1",
                    mentions=[_test_mention(source_doc_id="neighbors_test.py")],
                ),
            ],
            edges=[
                Edge(
                    tenant_id=TENANT_ID,
                    source="NeighborHub",
                    target="NeighborOut1",
                    relation="calls",
                    evidence_span="NeighborHub calls NeighborOut1",
                    segment_id="function:NeighborHub",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="neighbors_test.py",
                ),
                Edge(
                    tenant_id=TENANT_ID,
                    source="NeighborHub",
                    target="NeighborOut2",
                    relation="imports",
                    evidence_span="NeighborHub imports NeighborOut2",
                    segment_id="function:NeighborHub",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="neighbors_test.py",
                ),
                Edge(
                    tenant_id=TENANT_ID,
                    source="NeighborIn1",
                    target="NeighborHub",
                    relation="calls",
                    evidence_span="NeighborIn1 calls NeighborHub",
                    segment_id="function:NeighborIn1",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="neighbors_test.py",
                ),
            ],
        )
        manager.upsert(result)
        time.sleep(2)

        neighbors = manager.get_neighbors("NeighborHub")
        assert neighbors["node_id"] == "neighborhub"

        out_targets = {e["target_node_id"] for e in neighbors["out_edges"]}
        assert out_targets == {"neighborout1", "neighborout2"}

        in_sources = {e["source_node_id"] for e in neighbors["in_edges"]}
        assert in_sources == {"neighborin1"}

    def test_path_finds_route(self, graph_manager):
        manager, _ = graph_manager

        result = ExtractionResult(
            source_doc_id="path_test.py",
            nodes=[
                Node(
                    tenant_id=TENANT_ID,
                    name="PathA",
                    mentions=[_test_mention(source_doc_id="path_test.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="PathB",
                    mentions=[_test_mention(source_doc_id="path_test.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="PathC",
                    mentions=[_test_mention(source_doc_id="path_test.py")],
                ),
                Node(
                    tenant_id=TENANT_ID,
                    name="PathD",
                    mentions=[_test_mention(source_doc_id="path_test.py")],
                ),
            ],
            edges=[
                Edge(
                    tenant_id=TENANT_ID,
                    source="PathA",
                    target="PathB",
                    relation="calls",
                    evidence_span="PathA calls PathB",
                    segment_id="function:PathA",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="path_test.py",
                ),
                Edge(
                    tenant_id=TENANT_ID,
                    source="PathB",
                    target="PathC",
                    relation="calls",
                    evidence_span="PathB calls PathC",
                    segment_id="function:PathB",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="path_test.py",
                ),
                Edge(
                    tenant_id=TENANT_ID,
                    source="PathC",
                    target="PathD",
                    relation="calls",
                    evidence_span="PathC calls PathD",
                    segment_id="function:PathC",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="path_test.py",
                ),
            ],
        )
        manager.upsert(result)
        time.sleep(2)

        path = manager.get_path("PathA", "PathD", max_depth=5)
        assert path == ["patha", "pathb", "pathc", "pathd"]

    def test_path_visits_frontier_concurrently(self, graph_manager, monkeypatch):
        manager, _ = graph_manager

        fanout = ["CcA1", "CcA2", "CcA3", "CcA4"]
        nodes = [
            Node(
                tenant_id=TENANT_ID,
                name=name,
                mentions=[_test_mention(source_doc_id="concur_test.py")],
            )
            for name in ["CcSource", *fanout, "CcTarget"]
        ]
        edges = []
        for fan in fanout:
            edges.append(
                Edge(
                    tenant_id=TENANT_ID,
                    source="CcSource",
                    target=fan,
                    relation="calls",
                    evidence_span=f"CcSource calls {fan}",
                    segment_id="function:CcSource",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="concur_test.py",
                )
            )
            edges.append(
                Edge(
                    tenant_id=TENANT_ID,
                    source=fan,
                    target="CcTarget",
                    relation="calls",
                    evidence_span=f"{fan} calls CcTarget",
                    segment_id=f"function:{fan}",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    source_doc_id="concur_test.py",
                )
            )
        manager.upsert(
            ExtractionResult(source_doc_id="concur_test.py", nodes=nodes, edges=edges)
        )
        time.sleep(2)

        fanout_ids = {normalize_name(name) for name in fanout}
        barrier = threading.Barrier(len(fanout), timeout=20)
        real_visit_edges = manager._visit_edges

        def barrier_gated(source_node_id=None, target_node_id=None):
            # CcSource fans out to four nodes that share one BFS level. A serial
            # implementation visits them one at a time, so the barrier never
            # fills — the first caller blocks until the 20s timeout raises
            # BrokenBarrierError and the test fails. Only concurrent fetching
            # gets all four into the barrier at once.
            if source_node_id in fanout_ids:
                barrier.wait()
            return real_visit_edges(
                source_node_id=source_node_id, target_node_id=target_node_id
            )

        monkeypatch.setattr(manager, "_visit_edges", barrier_gated)

        path = manager.get_path("CcSource", "CcTarget", max_depth=5)
        assert path is not None
        assert path[0] == "ccsource"
        assert path[-1] == "cctarget"
        assert len(path) == 3
        assert path[1] in fanout_ids

    def test_stats_reports_counts_and_top_nodes(self, graph_manager):
        manager, _ = graph_manager
        time.sleep(2)

        stats = manager.get_stats()
        assert stats["node_count"] >= 4
        assert stats["edge_count"] >= 3
        top = stats["top_nodes"]
        assert isinstance(top, list)
        assert len(top) > 0
        assert all("node_id" in entry and "degree" in entry for entry in top)


@pytest.mark.integration
class TestGraphExtractorE2E:
    """Extract from real files → upsert → query round-trip."""

    def test_code_extractor_roundtrip(self, graph_manager):
        manager, port = graph_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "roundtrip.py"
            f.write_text(
                "def alpha():\n"
                "    beta()\n"
                "\n"
                "def beta():\n"
                "    return 42\n"
                "\n"
                "class Runner:\n"
                "    def run(self):\n"
                "        alpha()\n"
            )

            result = manager.extract_file(f, source_doc_id="roundtrip.py")

        assert result is not None
        counts = manager.upsert(result)
        assert counts["nodes_upserted"] >= 3

        alpha_doc = _get_vespa_doc(port, "kg_node_test_tenant_alpha")
        assert alpha_doc is not None
        assert alpha_doc.get("fields", {}).get("kind") == "entity"

        time.sleep(2)
        neighbors = manager.get_neighbors("roundtrip")
        out_targets = {e["target_node_id"] for e in neighbors["out_edges"]}
        assert "alpha" in out_targets or "beta" in out_targets


@pytest.mark.integration
class TestSearchNodesRealVespa:
    """search_nodes runs its real /search/ MaxSim+bm25 query and returns the
    node, rather than silently falling back to the substring _visit. Previously
    this path had only a pylate-sidecar-gated (usually skipped) test plus a unit
    test asserting the YQL string — nothing proved the real query executes."""

    def _feed_node(self, port, doc_id, name, description):
        fields = {
            "doc_id": doc_id,
            "tenant_id": TENANT_ID,
            "doc_type": "node",
            "name": name,
            "description": description,
            "embedding": {"blocks": {"0": [0.1] * 128}},
            "embedding_binary": {"blocks": {"0": [1] * 16}},
        }
        r = requests.post(_doc_url(port, doc_id), json={"fields": fields}, timeout=15)
        assert r.status_code in (200, 201), r.text[:300]

    def test_search_nodes_uses_real_query_not_substring_fallback(
        self, graph_vespa, monkeypatch
    ):
        from types import SimpleNamespace

        port = graph_vespa["http_port"]
        self._feed_node(
            port,
            "kg_node_search_marie",
            "Marie Curie",
            "studied radioactivity and polonium isotopes",
        )
        self._feed_node(
            port,
            "kg_node_search_other",
            "Random Topic",
            "unrelated cooking recipes and gardening",
        )
        time.sleep(2)

        mgr = GraphManager.__new__(GraphManager)
        mgr._schema_name = GRAPH_SCHEMA
        mgr._tenant_id = TENANT_ID
        mgr._backend = SimpleNamespace(_url="http://localhost", _port=port)

        # Controlled query blocks bypass the pylate encoder — the encoder
        # output is exercised elsewhere; here we cover the Vespa query path.
        monkeypatch.setattr(
            mgr,
            "_encode_query_blocks",
            lambda q: ({"0": [0.1] * 128}, {"0": [1] * 16}),
        )

        # The fallback only does substring name matching; "radioactivity" is in
        # the description, not the name, so a fallback would miss it. Asserting
        # no fallback fired proves the real /search/ query (not _visit) returned
        # the hit.
        visit_calls = []
        real_visit = mgr._visit

        def _spy_visit(*args, **kwargs):
            visit_calls.append((args, kwargs))
            return real_visit(*args, **kwargs)

        monkeypatch.setattr(mgr, "_visit", _spy_visit)

        results = mgr.search_nodes("radioactivity polonium", top_k=10)

        assert not visit_calls, (
            "search_nodes fell back to substring _visit instead of running the "
            "real /search/ query"
        )
        names = [r.get("name") for r in results]
        assert "Marie Curie" in names, names

    def test_search_nodes_survives_conflicting_qt_schema(
        self, graph_vespa, shared_memory_vespa, monkeypatch
    ):
        """Other schemas in the shared content cluster declare
        hybrid_binary_bm25 with different query(qt) dims (videoprism
        ``v[768]`` vs the graph's ``v[128]``). Without ``model.restrict``
        the query 400s on the conflicting input declarations and silently
        falls back to substring _visit. Deploys the conflicting schema
        explicitly so the conflict exists even in isolation.
        """
        from types import SimpleNamespace

        from tests.utils.vespa_test_helpers import deploy_tenant_schema

        deploy_tenant_schema(
            shared_memory_vespa,
            tenant_id=TENANT_ID,
            base_schema_name="video_videoprism_base_mv_chunk_30s",
            config_manager=shared_memory_vespa["config_manager"],
        )
        port = graph_vespa["http_port"]
        conflicting = schema_full_name("video_videoprism_base_mv_chunk_30s", TENANT_ID)
        # The graph probe document doesn't fit the videoprism schema, so
        # poll doc-type liveness via GET: 404 = type known + doc absent.
        probe_url = (
            f"http://localhost:{port}/document/v1/graph_content/"
            f"{conflicting}/docid/liveness_probe"
        )
        for _ in range(120):
            if requests.get(probe_url, timeout=5).status_code in (200, 404):
                break
            time.sleep(1)
        else:
            pytest.fail(f"conflicting schema {conflicting} doc type never went live")
        if not _wait_for_schema_ready(port, GRAPH_SCHEMA, timeout=120):
            pytest.fail(f"Schema {GRAPH_SCHEMA} not ready after co-deploy")

        self._feed_node(
            port,
            "kg_node_search_ada",
            "Ada Lovelace",
            "wrote the first analytical engine program",
        )
        time.sleep(2)

        mgr = GraphManager.__new__(GraphManager)
        mgr._schema_name = GRAPH_SCHEMA
        mgr._tenant_id = TENANT_ID
        mgr._backend = SimpleNamespace(_url="http://localhost", _port=port)
        monkeypatch.setattr(
            mgr,
            "_encode_query_blocks",
            lambda q: ({"0": [0.1] * 128}, {"0": [1] * 16}),
        )

        visit_calls = []
        real_visit = mgr._visit

        def _spy_visit(*args, **kwargs):
            visit_calls.append((args, kwargs))
            return real_visit(*args, **kwargs)

        monkeypatch.setattr(mgr, "_visit", _spy_visit)

        results = mgr.search_nodes("analytical engine program", top_k=10)

        assert not visit_calls, (
            "search_nodes fell back to _visit — the conflicting query(qt) "
            "declaration leaked into the graph query (model.restrict missing)"
        )
        names = [r.get("name") for r in results]
        assert "Ada Lovelace" in names, names
