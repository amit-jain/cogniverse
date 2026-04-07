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


def _ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def real_doc_extractor():
    """A DocExtractor with the real GLiNER model loaded.

    Loads `urchade/gliner_large-v2.1` once per module — ~1GB download on
    first run, cached afterwards in HF_HOME. Produces deterministic
    named-entity predictions so tests can assert exact entity names.
    """
    pytest.importorskip("gliner")

    from cogniverse_agents.graph.doc_extractor import DocExtractor

    extractor = DocExtractor()
    gliner = extractor._get_gliner()
    if gliner is None:
        pytest.skip("GLiNER model unavailable — skipping real-extractor tests")
    return extractor


@pytest.mark.integration
@pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama required for real embeddings in graph integration tests",
)
class TestMultimodalGraphExtraction:
    """Real multimodal extraction: real GLiNER, real Ollama embeddings, real Vespa.

    Exercises the exact code path the ingestion router uses:
      pipeline_result → _extract_text_for_graph → DocExtractor (GLiNER)
        → GraphManager.upsert → real Vespa HTTP feed

    Pipeline result dicts match the actual shapes produced by
    AudioTranscriber and VLMDescriptor (see ingestion/processors/*.py).
    Assertions are tight — exact node ids, exact mention lists, exact
    counts where GLiNER is deterministic.
    """

    _TRANSCRIPT_TEXT = (
        "In this tutorial we walk through the ColPali model from LightOn. "
        "ColPali is a multi-vector retrieval system that uses patch-level "
        "embeddings and late interaction for document image search. "
        "It is paired with Vespa as the vector storage backend for "
        "production-scale retrieval workloads."
    )

    def test_whisper_transcript_extracts_real_entities(self, graph_manager, real_doc_extractor):
        """A realistic Whisper transcript fixture round-trips through GLiNER and Vespa.

        Asserts that GLiNER finds the two most prominent named entities
        (ColPali and LightOn — both explicit product/org names in the text),
        that every node has the expected shape, and that the persisted
        Vespa document has a real Ollama embedding (non-zero 768-dim vector).
        """
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "transcript": {
                "full_text": self._TRANSCRIPT_TEXT,
                "segments": [
                    {"text": "The ColPali model was published by LightOn."},
                    {"text": "Vespa is the backend we will use."},
                ],
                "language": "en",
                "duration": 42.0,
            }
        }

        harvested = _extract_text_for_graph(pipeline_result)
        assert self._TRANSCRIPT_TEXT in harvested
        assert "ColPali model was published by LightOn" in harvested
        assert "Vespa is the backend" in harvested

        result = real_doc_extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="tutorial.mp4",
        )

        assert len(result.nodes) >= 3, (
            f"GLiNER should find at least 3 named entities, got {[n.name for n in result.nodes]}"
        )

        names_lower = {n.name.lower() for n in result.nodes}
        assert any("colpali" in n for n in names_lower), (
            f"GLiNER should find ColPali in transcript, got {names_lower}"
        )
        assert any("lighton" in n for n in names_lower), (
            f"GLiNER should find LightOn in transcript, got {names_lower}"
        )

        for node in result.nodes:
            assert node.tenant_id == TENANT_ID
            assert node.mentions == ["tutorial.mp4"]
            assert node.kind == "concept"
            assert node.node_id == node.name.replace(" ", "_").lower().replace("-", "_")

        edges_per_source = {}
        for edge in result.edges:
            assert edge.provenance == "INFERRED"
            assert edge.relation == "mentioned_with"
            assert edge.source_doc_id == "tutorial.mp4"
            edges_per_source.setdefault(edge.source_node_id, []).append(edge.target_node_id)

        counts = manager.upsert(result)
        assert counts["nodes_upserted"] == len(
            {n.node_id for n in result.nodes}
        ), f"Upsert count must match unique node count, got {counts}"
        assert counts["edges_upserted"] == len(result.edges)

        colpali_node = next(n for n in result.nodes if "colpali" in n.name.lower())
        doc = _get_vespa_doc(port, colpali_node.doc_id)
        assert doc is not None, f"ColPali node {colpali_node.doc_id} not in Vespa"

        fields = doc["fields"]
        assert fields["tenant_id"] == TENANT_ID
        assert fields["doc_type"] == "node"
        assert fields["kind"] == "concept"
        assert fields["name"] == colpali_node.name
        assert "tutorial.mp4" in json.loads(fields["mentions"])

        embedding_field = fields.get("embedding", {})
        embedding = embedding_field.get("values", []) if isinstance(embedding_field, dict) else embedding_field
        assert len(embedding) == 768, (
            f"Real Ollama embedding must be 768-dim, got {len(embedding)}"
        )
        non_zero_count = sum(1 for v in embedding if v != 0.0)
        assert non_zero_count > 700, (
            f"Real Ollama embedding should have ~all dims non-zero, got {non_zero_count}/768"
        )

    def test_vlm_descriptions_extract_real_entities(self, graph_manager, real_doc_extractor):
        """Realistic VLMDescriptor output round-trips through GLiNER and Vespa."""
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "descriptions": {
                "video_id": "architecture_overview",
                "descriptions": {
                    "frame_0": (
                        "A technical architecture diagram drawn on a whiteboard. "
                        "The diagram shows a Kubernetes cluster with Vespa deployed "
                        "as a stateful service."
                    ),
                    "frame_1": {
                        "description": (
                            "A Python code editor window showing imports from DSPy "
                            "and the ColPali library. The file is named search_agent.py."
                        )
                    },
                    "frame_2": {
                        "text": (
                            "A terminal window running Docker commands against the "
                            "Ollama service."
                        )
                    },
                },
            }
        }

        harvested = _extract_text_for_graph(pipeline_result)
        assert "Kubernetes" in harvested
        assert "Vespa" in harvested
        assert "DSPy" in harvested
        assert "ColPali" in harvested
        assert "Docker" in harvested
        assert "Ollama" in harvested

        result = real_doc_extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="architecture_overview.mp4",
        )

        names = {n.name.lower() for n in result.nodes}
        expected_concepts = {"kubernetes", "vespa", "dspy", "colpali", "docker", "ollama"}
        hits = sum(1 for c in expected_concepts if any(c in n for n in names))
        assert hits >= 4, (
            f"GLiNER should find at least 4 of {expected_concepts} in {names}"
        )

        counts = manager.upsert(result)
        unique_node_count = len({n.node_id for n in result.nodes})
        assert counts["nodes_upserted"] == unique_node_count

        persisted = []
        for concept in expected_concepts:
            matching = [n for n in result.nodes if concept in n.name.lower()]
            if matching:
                doc = _get_vespa_doc(port, f"kg_node_test_tenant_{matching[0].node_id}")
                if doc is not None:
                    persisted.append(concept)

        assert len(persisted) >= 4, (
            f"At least 4 expected concepts should persist, got {persisted}"
        )

    def test_combined_sources_produce_unified_graph(
        self, graph_manager, real_doc_extractor
    ):
        """Whisper + VLM + OCR outputs all merge into the same graph via one upsert.

        Uses well-known technology names (Kubernetes, PostgreSQL, React, TypeScript)
        that GLiNER reliably recognizes, so the test isn't fragile to GLiNER's
        entity recognition quality on obscure names.
        """
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "transcript": {
                "full_text": (
                    "This video covers deploying a React application to "
                    "Kubernetes with PostgreSQL as the database."
                ),
                "segments": [
                    {"text": "We will use TypeScript for type safety."},
                ],
            },
            "descriptions": {
                "descriptions": {
                    "frame_0": (
                        "A code editor showing a Redis client connecting to a "
                        "PostgreSQL database via the Prisma ORM."
                    ),
                }
            },
            "keyframes": {
                "keyframes": [
                    {"ocr_text": "Docker Compose running Nginx and Redis services"},
                ]
            },
        }

        harvested = _extract_text_for_graph(pipeline_result)
        for fragment in ("React", "Kubernetes", "PostgreSQL", "TypeScript", "Redis", "Prisma", "Docker", "Nginx"):
            assert fragment in harvested, f"{fragment} should be in harvested text"

        result = real_doc_extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="combined_overview.mp4",
        )

        names_lower = {n.name.lower() for n in result.nodes}
        well_known = {"react", "kubernetes", "postgresql", "typescript", "redis", "prisma", "docker", "nginx"}
        found = {w for w in well_known if any(w in n for n in names_lower)}
        assert len(found) >= 5, (
            f"GLiNER should find at least 5 of {well_known} in {names_lower}, found {found}"
        )

        counts = manager.upsert(result)
        unique_node_count = len({n.node_id for n in result.nodes})
        assert counts["nodes_upserted"] == unique_node_count
        assert counts["edges_upserted"] == len(result.edges)

        persisted_names: list = []
        for node in result.nodes:
            doc = _get_vespa_doc(port, node.doc_id)
            assert doc is not None, f"Node {node.doc_id} should persist in Vespa"
            fields = doc["fields"]
            assert fields["tenant_id"] == TENANT_ID
            assert fields["doc_type"] == "node"
            mentions = json.loads(fields["mentions"])
            assert "combined_overview.mp4" in mentions
            persisted_names.append(fields["name"])

        assert len(persisted_names) == unique_node_count, (
            "Every extracted node should persist to Vespa"
        )

    def test_empty_pipeline_result_produces_zero_graph(
        self, graph_manager, real_doc_extractor
    ):
        """An ingestion result with no text (e.g. audio with Whisper disabled) extracts nothing."""
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, _ = graph_manager

        harvested = _extract_text_for_graph({"chunks": [{"score": 1.0}]})
        assert harvested == ""

        result = real_doc_extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="silent.mp4",
        )
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

        counts = manager.upsert(result)
        assert counts == {"nodes_upserted": 0, "edges_upserted": 0}
