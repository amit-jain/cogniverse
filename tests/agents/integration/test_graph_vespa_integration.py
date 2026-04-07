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
    """DocExtractor with real GLiNER and tech/infra domain labels.

    The default DocExtractor labels (Technology, Concept, Organization,
    etc.) are generic and miss specific infrastructure entities. These
    tests use a tech-focused label set (Database, Platform, Tool,
    Library, Framework, Service, Organization, Model) which GLiNER
    confidently maps real infra names to.

    This is how the extractor is meant to be used — callers pass domain-
    specific labels. The test just declares the tech domain explicitly
    rather than relying on the permissive defaults.
    """
    pytest.importorskip("gliner")

    from cogniverse_agents.graph.doc_extractor import DocExtractor

    tech_labels = [
        "Database", "Platform", "Tool", "Library", "Framework",
        "Service", "Organization", "Model", "Algorithm",
    ]
    extractor = DocExtractor(labels=tech_labels)
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

    def _extracted_names(self, result) -> set[str]:
        """Normalized node names as a set for subset assertions."""
        return {n.name.lower() for n in result.nodes}

    def _find_node(self, result, substring: str):
        """Return the first node whose lowercase name contains substring, or None."""
        return next(
            (n for n in result.nodes if substring.lower() in n.name.lower()),
            None,
        )

    def _edges_between(self, result, src_substring: str, tgt_substring: str) -> list:
        """Return edges whose source_node_id and target_node_id contain the given substrings."""
        src = src_substring.lower()
        tgt = tgt_substring.lower()
        matches = []
        for edge in result.edges:
            if src in edge.source_node_id and tgt in edge.target_node_id:
                matches.append(edge)
            elif src in edge.target_node_id and tgt in edge.source_node_id:
                matches.append(edge)
        return matches

    def test_whisper_transcript_extracts_real_entities(self, graph_manager, real_doc_extractor):
        """Transcript → GLiNER → graph: specific entity + relationship assertions.

        Three known entities in the transcript (ColPali, LightOn, Vespa) must
        all be extracted, pairwise linked with mentioned_with edges (because
        they co-occur in the same chunk), and querying ColPali's neighbors
        via real Vespa must return both LightOn and Vespa.
        """
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "transcript": {
                "full_text": self._TRANSCRIPT_TEXT,
                "segments": [
                    {"text": "The ColPali model was published by LightOn."},
                    {"text": "ColPali is paired with Vespa for retrieval."},
                ],
                "language": "en",
                "duration": 42.0,
            }
        }

        harvested = _extract_text_for_graph(pipeline_result)
        assert self._TRANSCRIPT_TEXT in harvested
        assert "ColPali model was published by LightOn" in harvested
        assert "ColPali is paired with Vespa" in harvested

        result = real_doc_extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="tutorial.mp4",
        )

        names = self._extracted_names(result)
        colpali_node = self._find_node(result, "colpali")
        lighton_node = self._find_node(result, "lighton")
        vespa_node = self._find_node(result, "vespa")

        assert colpali_node is not None, f"ColPali not in extracted nodes: {names}"
        assert lighton_node is not None, f"LightOn not in extracted nodes: {names}"
        assert vespa_node is not None, f"Vespa not in extracted nodes: {names}"

        colpali_lighton_edges = self._edges_between(result, "colpali", "lighton")
        colpali_vespa_edges = self._edges_between(result, "colpali", "vespa")
        lighton_vespa_edges = self._edges_between(result, "lighton", "vespa")

        assert len(colpali_lighton_edges) >= 1, (
            "ColPali and LightOn are in the same chunk — must have a mentioned_with edge"
        )
        assert len(colpali_vespa_edges) >= 1, (
            "ColPali and Vespa are in the same chunk — must have a mentioned_with edge"
        )
        assert len(lighton_vespa_edges) >= 1, (
            "LightOn and Vespa are both in the transcript — must be linked"
        )

        for edge in result.edges:
            assert edge.provenance == "INFERRED"
            assert edge.relation == "mentioned_with"
            assert edge.source_doc_id == "tutorial.mp4"
            assert edge.confidence == 0.5

        for node in result.nodes:
            assert node.tenant_id == TENANT_ID
            assert node.mentions == ["tutorial.mp4"]
            assert node.kind == "concept"

        counts = manager.upsert(result)
        assert counts["nodes_upserted"] == len({n.node_id for n in result.nodes})
        assert counts["edges_upserted"] == len(result.edges)

        time.sleep(2)
        neighbors = manager.get_neighbors(colpali_node.name)
        out_targets = {e["target_node_id"] for e in neighbors["out_edges"]}
        in_sources = {e["source_node_id"] for e in neighbors["in_edges"]}
        colpali_connections = out_targets | in_sources

        assert lighton_node.node_id in colpali_connections, (
            f"LightOn must be a neighbor of ColPali in Vespa, got {colpali_connections}"
        )
        assert vespa_node.node_id in colpali_connections, (
            f"Vespa must be a neighbor of ColPali in Vespa, got {colpali_connections}"
        )

        path = manager.get_path(lighton_node.name, vespa_node.name, max_depth=3)
        assert path is not None, (
            f"A path must exist between LightOn and Vespa, got None"
        )
        assert lighton_node.node_id in path
        assert vespa_node.node_id in path

        doc = _get_vespa_doc(port, colpali_node.doc_id)
        assert doc is not None
        fields = doc["fields"]
        assert fields["tenant_id"] == TENANT_ID
        assert fields["doc_type"] == "node"
        assert fields["kind"] == "concept"
        assert fields["name"] == colpali_node.name
        assert "tutorial.mp4" in json.loads(fields["mentions"])

        embedding_field = fields.get("embedding", {})
        embedding = (
            embedding_field.get("values", [])
            if isinstance(embedding_field, dict)
            else embedding_field
        )
        assert len(embedding) == 768
        non_zero_count = sum(1 for v in embedding if v != 0.0)
        assert non_zero_count > 700, (
            f"Real Ollama embedding should be densely non-zero, got {non_zero_count}/768"
        )

    def test_vlm_descriptions_extract_real_entities(self, graph_manager, real_doc_extractor):
        """VLM descriptions → graph: each frame's entities are linked intra-frame.

        Three separate VLM frame descriptions each have distinct entities.
        The extractor chunks per paragraph so entities from different frames
        are NOT linked (they're in different chunks), but entities from the
        same frame must be co-mention linked.
        """
        from cogniverse_runtime.routers.ingestion import _extract_text_for_graph

        manager, port = graph_manager

        pipeline_result = {
            "descriptions": {
                "video_id": "architecture_overview",
                "descriptions": {
                    "frame_0": (
                        "A technical architecture diagram drawn on a whiteboard "
                        "showing a Kubernetes cluster with Vespa deployed as a "
                        "stateful service."
                    ),
                    "frame_1": {
                        "description": (
                            "A Python code editor window showing imports from "
                            "DSPy and the ColPali library."
                        )
                    },
                    "frame_2": {
                        "text": "A terminal window running Docker against Ollama."
                    },
                },
            }
        }

        harvested = _extract_text_for_graph(pipeline_result)
        for fragment in ("Kubernetes", "Vespa", "DSPy", "ColPali", "Docker", "Ollama"):
            assert fragment in harvested

        result = real_doc_extractor.extract_from_text(
            text=harvested,
            tenant_id=TENANT_ID,
            source_doc_id="architecture_overview.mp4",
        )

        kubernetes_node = self._find_node(result, "kubernetes")
        vespa_node = self._find_node(result, "vespa")
        dspy_node = self._find_node(result, "dspy")
        colpali_node = self._find_node(result, "colpali")
        docker_node = self._find_node(result, "docker")
        ollama_node = self._find_node(result, "ollama")

        required_found = [
            ("Kubernetes", kubernetes_node),
            ("Vespa", vespa_node),
            ("DSPy", dspy_node),
            ("ColPali", colpali_node),
            ("Docker", docker_node),
            ("Ollama", ollama_node),
        ]
        missing = [name for name, node in required_found if node is None]
        assert not missing, (
            f"GLiNER must find all frame entities, missing: {missing}"
        )

        kube_vespa = self._edges_between(result, "kubernetes", "vespa")
        assert len(kube_vespa) >= 1, (
            "Kubernetes and Vespa are in the same frame — must be linked"
        )

        dspy_colpali = self._edges_between(result, "dspy", "colpali")
        assert len(dspy_colpali) >= 1, (
            "DSPy and ColPali are in the same frame — must be linked"
        )

        docker_ollama = self._edges_between(result, "docker", "ollama")
        assert len(docker_ollama) >= 1, (
            "Docker and Ollama are in the same frame — must be linked"
        )

        counts = manager.upsert(result)
        assert counts["nodes_upserted"] == len({n.node_id for n in result.nodes})
        assert counts["edges_upserted"] == len(result.edges)

        time.sleep(2)
        neighbors = manager.get_neighbors(kubernetes_node.name)
        kube_connections = {
            e["target_node_id"] for e in neighbors["out_edges"]
        } | {e["source_node_id"] for e in neighbors["in_edges"]}
        assert vespa_node.node_id in kube_connections, (
            f"Vespa must be a neighbor of Kubernetes (same frame), got {kube_connections}"
        )

        for node in (kubernetes_node, vespa_node, dspy_node, colpali_node, docker_node, ollama_node):
            doc = _get_vespa_doc(port, node.doc_id)
            assert doc is not None, f"Expected {node.name} to persist as {node.doc_id}"
            fields = doc["fields"]
            assert fields["tenant_id"] == TENANT_ID
            assert fields["doc_type"] == "node"
            assert "architecture_overview.mp4" in json.loads(fields["mentions"])

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
