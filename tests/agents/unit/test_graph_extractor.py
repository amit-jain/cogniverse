"""Unit tests for the knowledge graph extractor — code + docs extraction."""

import tempfile
from pathlib import Path

import pytest

from cogniverse_agents.graph.code_extractor import CodeExtractor
from cogniverse_agents.graph.doc_extractor import DocExtractor
from cogniverse_agents.graph.graph_schema import (
    Edge,
    Node,
    normalize_name,
)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestNormalizeName:
    def test_lowercases(self):
        assert normalize_name("SearchAgent") == "searchagent"

    def test_replaces_spaces(self):
        assert normalize_name("Search Agent") == "search_agent"

    def test_strips_punctuation(self):
        assert normalize_name("What's New?") == "what_s_new"

    def test_handles_unicode(self):
        assert normalize_name("Café au lait") == "cafe_au_lait"

    def test_collapses_multiple_separators(self):
        assert normalize_name("Foo--Bar__Baz") == "foo_bar_baz"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestNodeDataclass:
    def test_node_id_is_normalized_name(self):
        node = Node(tenant_id="t1", name="SearchAgent")
        assert node.node_id == "searchagent"

    def test_doc_id_includes_tenant_and_node_id(self):
        node = Node(tenant_id="acme:prod", name="SearchAgent")
        assert node.doc_id == "kg_node_acme_prod_searchagent"

    def test_to_vespa_document_has_all_fields(self):
        node = Node(
            tenant_id="t1",
            name="Retry",
            description="A retry decorator",
            kind="entity",
            mentions=["utils/retry.py"],
        )
        doc = node.to_vespa_document()
        fields = doc["fields"]
        assert fields["doc_type"] == "node"
        assert fields["name"] == "Retry"
        assert fields["kind"] == "entity"
        assert "utils/retry.py" in fields["mentions"]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestEdgeDataclass:
    def test_edge_id_is_deterministic(self):
        e1 = Edge(tenant_id="t1", source="A", target="B", relation="calls")
        e2 = Edge(tenant_id="t1", source="A", target="B", relation="calls")
        assert e1.edge_id == e2.edge_id

    def test_edge_id_differs_by_relation(self):
        e1 = Edge(tenant_id="t1", source="A", target="B", relation="calls")
        e2 = Edge(tenant_id="t1", source="A", target="B", relation="imports")
        assert e1.edge_id != e2.edge_id

    def test_normalized_source_and_target(self):
        edge = Edge(
            tenant_id="t1",
            source="SearchAgent",
            target="Vespa Backend",
            relation="calls",
        )
        assert edge.source_node_id == "searchagent"
        assert edge.target_node_id == "vespa_backend"

    def test_to_vespa_document_has_all_fields(self):
        edge = Edge(
            tenant_id="t1",
            source="SearchAgent",
            target="VespaBackend",
            relation="calls",
            provenance="EXTRACTED",
            source_doc_id="search_agent.py",
        )
        doc = edge.to_vespa_document()
        fields = doc["fields"]
        assert fields["doc_type"] == "edge"
        assert fields["relation"] == "calls"
        assert fields["provenance"] == "EXTRACTED"
        assert fields["source_node_id"] == "searchagent"
        assert fields["target_node_id"] == "vespabackend"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCodeExtractor:
    def test_extracts_python_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text(
                "def greet(name):\n"
                "    print('hello', name)\n"
                "\n"
                "class Greeter:\n"
                "    def say_hi(self):\n"
                "        greet('world')\n"
            )
            result = CodeExtractor().extract(f, "t1", "sample.py")

        assert result is not None
        node_names = {n.name for n in result.nodes}
        assert "sample" in node_names
        assert "greet" in node_names
        assert "Greeter" in node_names
        assert "say_hi" in node_names

        edges_by_rel = {}
        for e in result.edges:
            edges_by_rel.setdefault(e.relation, []).append(e)
        assert "defines" in edges_by_rel
        assert any(e.target_node_id == "greet" for e in edges_by_rel["defines"])

    def test_returns_none_for_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.txt"
            f.write_text("not code")
            result = CodeExtractor().extract(f, "t1", "sample.txt")
        assert result is None

    def test_extracts_imports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text(
                "import os\n"
                "from pathlib import Path\n"
                "\n"
                "def main():\n"
                "    Path('/tmp').exists()\n"
            )
            result = CodeExtractor().extract(f, "t1", "sample.py")

        assert result is not None
        import_edges = [e for e in result.edges if e.relation == "imports"]
        assert len(import_edges) >= 1
        targets = {e.target_node_id for e in import_edges}
        assert any(t in ("os", "path") for t in targets)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDocExtractor:
    def test_fallback_extracts_capitalized_phrases(self):
        """When GLiNER is unavailable, the fallback grabs capitalized concepts."""
        extractor = DocExtractor()
        extractor._gliner_failed = True
        result = extractor.extract_from_text(
            "The ColPali model uses late interaction over patch embeddings. "
            "It works with Vespa for video retrieval.",
            tenant_id="t1",
            source_doc_id="doc1.md",
        )
        names = {n.name for n in result.nodes}
        assert "ColPali" in names
        assert "Vespa" in names

    def test_chunks_long_text(self):
        extractor = DocExtractor()
        chunks = extractor._chunk_text("A" * 5000)
        assert len(chunks) >= 2

    def test_extract_from_text_produces_edges_between_entities(self):
        extractor = DocExtractor()
        extractor._gliner_failed = True
        result = extractor.extract_from_text(
            "The ColPali model beats Vespa's default ranker on video queries.",
            tenant_id="t1",
            source_doc_id="doc1.md",
        )
        assert len(result.edges) >= 1
        for edge in result.edges:
            assert edge.provenance == "INFERRED"
            assert edge.relation == "mentioned_with"

    def test_returns_none_for_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("def foo(): pass")
            result = DocExtractor().extract(f, "t1", "sample.py")
        assert result is None
