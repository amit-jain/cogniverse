"""Integration tests for GraphManager against a real Vespa Docker container.

Starts its own Vespa, deploys the knowledge_graph_test_tenant schema,
exercises the full extract → upsert → query round-trip, then tears down.
Exactly the same pattern as test_wiki_vespa_integration.py.
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
)
from tests.utils.docker_utils import generate_unique_ports


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
GRAPH_SCHEMA = "knowledge_graph_test_tenant"
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


def _deploy_graph_schema(config_port: int, http_port: int) -> None:
    """Deploy the knowledge_graph_test_tenant schema via ApplicationPackage."""
    from vespa.package import ApplicationPackage

    from cogniverse_vespa.json_schema_parser import JsonSchemaParser
    from cogniverse_vespa.metadata_schemas import (
        create_adapter_registry_schema,
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    metadata_schemas = [
        create_organization_metadata_schema(),
        create_tenant_metadata_schema(),
        create_config_metadata_schema(),
        create_adapter_registry_schema(),
    ]

    parser = JsonSchemaParser()
    schema_file = Path("configs/schemas/knowledge_graph_schema.json")
    with open(schema_file) as f:
        schema_json = json.load(f)
    schema_json["name"] = GRAPH_SCHEMA
    schema_json["document"]["name"] = GRAPH_SCHEMA
    graph_schema = parser.parse_schema(schema_json)

    app_package = ApplicationPackage(
        name="cogniverse", schema=metadata_schemas + [graph_schema]
    )
    mgr = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=config_port,
    )
    mgr._deploy_package(app_package)


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
    """Run the real deploy/pylate/server.py in a background thread.

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
        "pylate_server_under_test", "deploy/pylate/server.py"
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
