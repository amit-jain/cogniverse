"""Integration tests for GraphManager against a real Vespa Docker container.

Starts its own Vespa, deploys the knowledge_graph_test_tenant schema,
exercises the full extract → upsert → query round-trip, then tears down.
Exactly the same pattern as test_wiki_vespa_integration.py.
"""

import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import Edge, ExtractionResult, Node
from tests.utils.docker_utils import generate_unique_ports

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


def _wait_for_schema_ready(http_port: int, schema_name: str, timeout: int = 120) -> bool:
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
            "embedding": [0.01] * 768,
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
                print(f"   readiness attempt {i + 1}: {resp.status_code} {resp.text[:100]}")
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
def graph_vespa():
    """Module-scoped Vespa with knowledge_graph schema deployed."""
    http_port = _HTTP_PORT
    config_port = _CONFIG_PORT

    machine = platform.machine().lower()
    docker_platform = "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"

    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)

    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{http_port}:8080",
            "-p", f"{config_port}:19071",
            "--platform", docker_platform,
            "vespaengine/vespa",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Vespa container: {result.stderr}")

    print(f"\nVespa container started on http={http_port}, config={config_port}")

    if not _wait_for_config_port(config_port, timeout=120):
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail("Vespa config port did not come up within 120s")

    time.sleep(10)

    try:
        _deploy_graph_schema(config_port, http_port)
        print("Graph schema deployed")
    except Exception as exc:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail(f"Schema deployment failed: {exc}")

    if not _wait_for_data_port(http_port, timeout=120):
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail("Vespa data port did not come up within 120s after deployment")

    if not _wait_for_schema_ready(http_port, GRAPH_SCHEMA, timeout=120):
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail(f"Schema {GRAPH_SCHEMA} not ready within 120s")

    yield {"http_port": http_port, "config_port": config_port}

    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)


@pytest.fixture(scope="module")
def graph_manager(graph_vespa):
    """GraphManager wired to the test Vespa."""
    http_port = graph_vespa["http_port"]

    backend = MagicMock()
    backend._url = "http://localhost"
    backend._port = http_port
    backend.search.return_value = []

    manager = GraphManager(
        backend=backend,
        tenant_id=TENANT_ID,
        schema_name=GRAPH_SCHEMA,
    )
    yield manager, http_port


@pytest.mark.integration
class TestGraphVespaUpsert:
    """Upsert nodes/edges and verify they're retrievable from real Vespa."""

    def test_upsert_nodes_persists_to_vespa(self, graph_manager):
        manager, port = graph_manager

        result = ExtractionResult(
            source_doc_id="test_upsert.py",
            nodes=[
                Node(tenant_id=TENANT_ID, name="IntegrationAlpha", kind="entity",
                     description="First integration node"),
                Node(tenant_id=TENANT_ID, name="IntegrationBeta", kind="entity",
                     description="Second integration node"),
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
                Node(tenant_id=TENANT_ID, name="EdgeSource", kind="entity"),
                Node(tenant_id=TENANT_ID, name="EdgeTarget", kind="entity"),
            ],
            edges=[
                Edge(
                    tenant_id=TENANT_ID,
                    source="EdgeSource",
                    target="EdgeTarget",
                    relation="calls",
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
                Node(tenant_id=TENANT_ID, name="MergeMe", mentions=["first.py"]),
                Node(tenant_id=TENANT_ID, name="mergeme", mentions=["second.py"]),
            ],
            edges=[],
        )
        counts = manager.upsert(result)
        assert counts["nodes_upserted"] == 1

        doc = _get_vespa_doc(port, "kg_node_test_tenant_mergeme")
        assert doc is not None
        mentions_json = doc.get("fields", {}).get("mentions", "[]")
        mentions = json.loads(mentions_json)
        assert "first.py" in mentions
        assert "second.py" in mentions


@pytest.mark.integration
class TestGraphVespaQueries:
    """Test neighbors/path/stats against real Vespa."""

    def test_neighbors_returns_outgoing_and_incoming(self, graph_manager):
        manager, _ = graph_manager

        result = ExtractionResult(
            source_doc_id="neighbors_test.py",
            nodes=[
                Node(tenant_id=TENANT_ID, name="NeighborHub"),
                Node(tenant_id=TENANT_ID, name="NeighborOut1"),
                Node(tenant_id=TENANT_ID, name="NeighborOut2"),
                Node(tenant_id=TENANT_ID, name="NeighborIn1"),
            ],
            edges=[
                Edge(tenant_id=TENANT_ID, source="NeighborHub", target="NeighborOut1",
                     relation="calls", source_doc_id="neighbors_test.py"),
                Edge(tenant_id=TENANT_ID, source="NeighborHub", target="NeighborOut2",
                     relation="imports", source_doc_id="neighbors_test.py"),
                Edge(tenant_id=TENANT_ID, source="NeighborIn1", target="NeighborHub",
                     relation="calls", source_doc_id="neighbors_test.py"),
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
                Node(tenant_id=TENANT_ID, name="PathA"),
                Node(tenant_id=TENANT_ID, name="PathB"),
                Node(tenant_id=TENANT_ID, name="PathC"),
                Node(tenant_id=TENANT_ID, name="PathD"),
            ],
            edges=[
                Edge(tenant_id=TENANT_ID, source="PathA", target="PathB",
                     relation="calls", source_doc_id="path_test.py"),
                Edge(tenant_id=TENANT_ID, source="PathB", target="PathC",
                     relation="calls", source_doc_id="path_test.py"),
                Edge(tenant_id=TENANT_ID, source="PathC", target="PathD",
                     relation="calls", source_doc_id="path_test.py"),
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


@pytest.mark.integration
class TestMultimodalGraphExtraction:
    """Feed transcript/VLM text through the extractor and persist to real Vespa.

    These tests exercise the same code path that the ingestion router's
    /ingestion/upload endpoint uses for video/image/audio files — the
    ``_extract_text_for_graph`` helper harvests text from pipeline
    results, then ``DocExtractor.extract_from_text`` turns that text
    into nodes/edges, then ``GraphManager.upsert`` writes them to Vespa.
    """

    def test_video_transcript_extracts_nodes(self, graph_manager):
        from cogniverse_agents.graph.doc_extractor import DocExtractor
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "transcript": {
                "full_text": (
                    "In this video we demonstrate the ColPali model "
                    "running against Vespa for video retrieval. "
                    "The SearchAgent routes queries to the right profile "
                    "and the RoutingAgent handles query enhancement."
                ),
                "segments": [
                    {"text": "ColPali uses late interaction over patches."},
                    {"text": "Vespa is the vector database backend."},
                ],
            },
        }

        harvested = _extract_text_for_graph(pipeline_result)
        assert "ColPali" in harvested
        assert "Vespa" in harvested

        extractor = DocExtractor()
        extractor._gliner_failed = True
        result = extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="video_multimodal_1.mp4",
        )
        counts = manager.upsert(result)
        assert counts["nodes_upserted"] >= 2

        colpali_doc = _get_vespa_doc(port, "kg_node_test_tenant_colpali")
        assert colpali_doc is not None, "ColPali node not persisted to Vespa"
        fields = colpali_doc.get("fields", {})
        assert fields.get("doc_type") == "node"
        assert fields.get("tenant_id") == TENANT_ID
        mentions = json.loads(fields.get("mentions", "[]"))
        assert "video_multimodal_1.mp4" in mentions

    def test_vlm_descriptions_extract_nodes(self, graph_manager):
        from cogniverse_agents.graph.doc_extractor import DocExtractor
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "descriptions": {
                "descriptions": {
                    "frame_1": "A whiteboard diagram showing SearchAgent calling VespaBackend.",
                    "frame_2": {"description": "Code editor with ColPali imports and DSPy modules."},
                    "frame_3": {"text": "Terminal running kubectl logs deployment."},
                }
            }
        }

        harvested = _extract_text_for_graph(pipeline_result)
        assert "SearchAgent" in harvested
        assert "VespaBackend" in harvested
        assert "ColPali" in harvested

        extractor = DocExtractor()
        extractor._gliner_failed = True
        result = extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="image_vlm_1.jpg",
        )
        counts = manager.upsert(result)
        assert counts["nodes_upserted"] >= 3

        time.sleep(2)
        neighbors = manager.get_neighbors("SearchAgent")
        out_targets = {e["target_node_id"] for e in neighbors["out_edges"]}
        in_sources = {e["source_node_id"] for e in neighbors["in_edges"]}
        assert "vespabackend" in out_targets or "vespabackend" in in_sources

    def test_combined_pipeline_result_extracts_all_sources(self, graph_manager):
        from cogniverse_agents.graph.doc_extractor import DocExtractor
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "transcript": {
                "full_text": "The SearchAgent orchestrates queries using RoutingAgent.",
                "segments": [],
            },
            "descriptions": {
                "descriptions": {
                    "frame_1": "A diagram of CodingAgent interacting with OpenShell.",
                }
            },
            "keyframes": {
                "keyframes": [
                    {"ocr_text": "CodingAgent -> OpenShell -> Sandbox"},
                ]
            },
        }

        harvested = _extract_text_for_graph(pipeline_result)
        assert "SearchAgent" in harvested
        assert "RoutingAgent" in harvested
        assert "CodingAgent" in harvested
        assert "OpenShell" in harvested

        extractor = DocExtractor()
        extractor._gliner_failed = True
        result = extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="combined_multimodal_1.mp4",
        )
        counts = manager.upsert(result)
        assert counts["nodes_upserted"] >= 4

        time.sleep(2)
        for expected in ("searchagent", "routingagent", "codingagent", "openshell"):
            doc = _get_vespa_doc(port, f"kg_node_test_tenant_{expected}")
            assert doc is not None, f"Expected node {expected} to persist"

    def test_empty_pipeline_result_produces_no_graph(self, graph_manager):
        """Files with no extractable text (e.g. pure audio without whisper) get 0."""
        from cogniverse_agents.graph.doc_extractor import DocExtractor
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, _ = graph_manager

        harvested = _extract_text_for_graph({"chunks": [{"score": 1.0}]})
        assert harvested == ""

        extractor = DocExtractor()
        extractor._gliner_failed = True
        result = extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="silent.mp4",
        )
        assert len(result.nodes) == 0
