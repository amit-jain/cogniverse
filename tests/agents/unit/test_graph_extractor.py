"""Unit tests for the knowledge graph extractor — code + docs extraction."""

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_agents.graph.code_extractor import CodeExtractor
from cogniverse_agents.graph.doc_extractor import DocExtractor, _is_blocked_entity
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


def _stub_mention(evidence_span: str = "stub"):
    """Tiny Mention factory for unit tests that don't care about anchor data."""
    from cogniverse_agents.graph.graph_schema import Mention

    return Mention(
        source_doc_id="doc",
        segment_id="seg",
        ts_start=0.0,
        ts_end=0.0,
        modality="code",
        evidence_span=evidence_span,
    )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestNodeDataclass:
    def test_node_id_is_normalized_name(self):
        node = Node(tenant_id="t1", name="SearchAgent", mentions=[_stub_mention()])
        assert node.node_id == "searchagent"

    def test_doc_id_includes_tenant_and_node_id(self):
        node = Node(
            tenant_id="acme:prod", name="SearchAgent", mentions=[_stub_mention()]
        )
        assert node.doc_id == "kg_node_acme_prod_searchagent"

    def test_to_vespa_document_has_all_fields(self):
        node = Node(
            tenant_id="t1",
            name="Retry",
            description="A retry decorator",
            kind="entity",
            mentions=[_stub_mention("utils/retry.py")],
        )
        doc = node.to_vespa_document()
        fields = doc["fields"]
        assert fields["doc_type"] == "node"
        assert fields["name"] == "Retry"
        assert fields["kind"] == "entity"
        assert fields["label"] == "Concept"  # default label propagates
        assert "utils/retry.py" in fields["mentions"]

    def test_label_round_trips_through_to_vespa_document(self):
        node = Node(
            tenant_id="t1",
            name="Marie Curie",
            kind="entity",
            label="Person",
            mentions=[_stub_mention()],
        )
        assert node.label == "Person"
        assert node.to_vespa_document()["fields"]["label"] == "Person"


def _edge_anchor_kwargs():
    """Shared anchor kwargs for Edge construction — keep the tests focused
    on edge identity / serialisation rather than mention plumbing."""
    return {
        "evidence_span": "stub",
        "segment_id": "seg",
        "ts_start": 0.0,
        "ts_end": 0.0,
        "modality": "code",
    }


@pytest.mark.unit
@pytest.mark.ci_fast
class TestEdgeDataclass:
    def test_edge_id_is_deterministic(self):
        e1 = Edge(
            tenant_id="t1",
            source="A",
            target="B",
            relation="calls",
            **_edge_anchor_kwargs(),
        )
        e2 = Edge(
            tenant_id="t1",
            source="A",
            target="B",
            relation="calls",
            **_edge_anchor_kwargs(),
        )
        assert e1.edge_id == e2.edge_id

    def test_edge_id_differs_by_relation(self):
        e1 = Edge(
            tenant_id="t1",
            source="A",
            target="B",
            relation="calls",
            **_edge_anchor_kwargs(),
        )
        e2 = Edge(
            tenant_id="t1",
            source="A",
            target="B",
            relation="imports",
            **_edge_anchor_kwargs(),
        )
        assert e1.edge_id != e2.edge_id

    def test_normalized_source_and_target(self):
        edge = Edge(
            tenant_id="t1",
            source="SearchAgent",
            target="Vespa Backend",
            relation="calls",
            **_edge_anchor_kwargs(),
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
            **_edge_anchor_kwargs(),
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


def _doc_anchor():
    """Per-segment Mention anchor for DocExtractor.extract_from_text calls."""
    from cogniverse_agents.graph.graph_schema import Mention

    return Mention(
        source_doc_id="doc1.md",
        segment_id="file",
        ts_start=0.0,
        ts_end=0.0,
        modality="document",
        evidence_span="stub",
    )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDocExtractor:
    def test_explicit_gliner_endpoint_is_used_without_global_discovery(
        self, monkeypatch
    ):
        import cogniverse_core.common.models as model_helpers

        model = SimpleNamespace(predict_entities=lambda *args, **kwargs: [])
        calls = []

        def load_model(model_name, *, logger, inference_url):
            calls.append((model_name, inference_url))
            return model

        extractor = DocExtractor(
            gliner_inference_url="http://worker-gliner.internal:8000"
        )
        monkeypatch.setattr(model_helpers, "get_or_load_gliner", load_model)
        monkeypatch.setattr(
            extractor,
            "_discover_gliner_url",
            lambda: pytest.fail("global configuration discovery was called"),
        )

        assert extractor._get_gliner() is model
        assert calls == [
            (
                "urchade/gliner_large-v2.1",
                "http://worker-gliner.internal:8000",
            )
        ]

    def test_gliner_url_uses_only_validated_system_configuration(self, monkeypatch):
        import cogniverse_agents.graph.doc_extractor as doc_extractor_module

        manager = SimpleNamespace(
            get_system_config=lambda: SimpleNamespace(
                inference_service_urls={"gliner": "https://models.example.test/gliner"}
            )
        )
        monkeypatch.setattr(
            doc_extractor_module, "get_config_manager_singleton", lambda: manager
        )
        monkeypatch.setenv("INFERENCE_SERVICE_URLS", "not-json")
        monkeypatch.setenv("GLINER_INFERENCE_URL", "http://obsolete.example.test")

        assert DocExtractor._discover_gliner_url() == (
            "https://models.example.test/gliner"
        )

    def test_absent_system_gliner_url_does_not_read_obsolete_environment(
        self, monkeypatch
    ):
        import cogniverse_agents.graph.doc_extractor as doc_extractor_module

        manager = SimpleNamespace(
            get_system_config=lambda: SimpleNamespace(inference_service_urls={})
        )
        monkeypatch.setattr(
            doc_extractor_module, "get_config_manager_singleton", lambda: manager
        )
        monkeypatch.setenv("GLINER_INFERENCE_URL", "http://obsolete.example.test")

        assert DocExtractor._discover_gliner_url() is None

    def test_system_configuration_failure_raises_with_context(self, monkeypatch):
        import cogniverse_agents.graph.doc_extractor as doc_extractor_module

        class FailingManager:
            @staticmethod
            def get_system_config():
                raise ConnectionError("configuration Vespa refused the request")

        monkeypatch.setattr(
            doc_extractor_module,
            "get_config_manager_singleton",
            lambda: FailingManager(),
        )

        with pytest.raises(
            RuntimeError, match="failed to resolve GLiNER inference service URL"
        ) as exc_info:
            DocExtractor._discover_gliner_url()

        assert isinstance(exc_info.value.__cause__, ConnectionError)
        assert str(exc_info.value.__cause__) == (
            "configuration Vespa refused the request"
        )

    def test_concurrent_gliner_url_discovery_never_crosses_config_sources(
        self, monkeypatch
    ):
        import cogniverse_agents.graph.doc_extractor as doc_extractor_module

        manager = SimpleNamespace(
            get_system_config=lambda: SimpleNamespace(
                inference_service_urls={"gliner": "http://gliner.internal:8000"}
            )
        )
        monkeypatch.setattr(
            doc_extractor_module, "get_config_manager_singleton", lambda: manager
        )
        monkeypatch.setenv("GLINER_INFERENCE_URL", "http://obsolete.example.test")

        with ThreadPoolExecutor(max_workers=8) as executor:
            urls = list(
                executor.map(lambda _: DocExtractor._discover_gliner_url(), range(32))
            )

        assert urls == ["http://gliner.internal:8000"] * 32

    def test_missing_gliner_raises_instead_of_fabricating_regex_entities(self):
        extractor = DocExtractor()
        extractor._get_gliner = lambda: None

        with pytest.raises(RuntimeError, match="GLiNER model is unavailable"):
            extractor.extract_from_text(
                "The ColPali model uses late interaction over patch embeddings.",
                tenant_id="t1",
                source_doc_id="doc1.md",
                segment_anchor=_doc_anchor(),
            )

    def test_chunks_long_text(self):
        extractor = DocExtractor()
        chunks = extractor._chunk_text("A" * 5000)
        assert len(chunks) >= 2

    def test_extract_from_text_emits_no_edges_without_claim_extractor(self):
        """Without a ClaimExtractor wired in, DocExtractor produces nodes only.

        The legacy "mentioned_with" co-occurrence edges were removed when SPO
        claim extraction took over (see ``doc_extractor.py`` module docstring).
        This test guards against accidental reintroduction.
        """
        extractor = DocExtractor()
        extractor._get_gliner = lambda: SimpleNamespace(
            predict_entities=lambda *args, **kwargs: []
        )
        result = extractor.extract_from_text(
            "The ColPali model beats Vespa's default ranker on video queries.",
            tenant_id="t1",
            source_doc_id="doc1.md",
            segment_anchor=_doc_anchor(),
        )
        assert result.edges == []
        assert result.nodes == []

    def test_empty_gliner_result_remains_empty(self):
        extractor = DocExtractor()
        extractor._get_gliner = lambda: SimpleNamespace(
            predict_entities=lambda *args, **kwargs: []
        )
        result = extractor.extract_from_text(
            "ColPali uses Vespa.",
            tenant_id="t1",
            source_doc_id="doc1.md",
            segment_anchor=_doc_anchor(),
        )
        assert result.nodes == []

    def test_chunk_prediction_failure_raises_with_chunk_context(self):
        extractor = DocExtractor()
        extractor._get_gliner = lambda: SimpleNamespace(
            predict_entities=lambda *args, **kwargs: (_ for _ in ()).throw(
                ConnectionError("sidecar disconnected")
            )
        )

        with pytest.raises(RuntimeError, match="GLiNER prediction failed for chunk 1"):
            extractor.extract_from_text(
                "Marie Curie discovered radium.",
                tenant_id="t1",
                source_doc_id="doc1.md",
                segment_anchor=_doc_anchor(),
            )

    def test_concurrent_cold_load_builds_gliner_once(self, monkeypatch):
        import cogniverse_core.common.models as model_helpers

        extractor = DocExtractor()
        model = SimpleNamespace(predict_entities=lambda *args, **kwargs: [])
        load_count = 0
        count_lock = threading.Lock()
        release = threading.Barrier(8)

        def load_model(*args, **kwargs):
            nonlocal load_count
            with count_lock:
                load_count += 1
            return model

        monkeypatch.setattr(model_helpers, "get_or_load_gliner", load_model)
        monkeypatch.setattr(extractor, "_discover_gliner_url", lambda: None)

        with ThreadPoolExecutor(max_workers=8) as executor:
            loaded = list(
                executor.map(
                    lambda _: (release.wait(), extractor._get_gliner())[1],
                    range(8),
                )
            )

        assert loaded == [model] * 8
        assert load_count == 1

    def test_bare_construction_routes_through_configured_inference_service(
        self, monkeypatch
    ):
        """A DocExtractor built without an explicit URL discovers the GLiNER
        inference service from system configuration and routes every
        prediction through it — never an in-process model load."""
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        import cogniverse_agents.graph.doc_extractor as doc_extractor_module

        seen = {}

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                seen["path"] = self.path
                seen["payload"] = json.loads(body)
                response = json.dumps(
                    {
                        "entities": [
                            {
                                "text": "Marie Curie",
                                "label": "Person",
                                "score": 0.93,
                                "start": 0,
                                "end": 11,
                            },
                            {
                                "text": "radium",
                                "label": "Substance",
                                "score": 0.88,
                                "start": 23,
                                "end": 29,
                            },
                        ],
                        "model": "urchade/gliner_large-v2.1",
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}"

        manager = SimpleNamespace(
            get_system_config=lambda: SimpleNamespace(
                inference_service_urls={"gliner": url}
            )
        )
        monkeypatch.setattr(
            doc_extractor_module, "get_config_manager_singleton", lambda: manager
        )

        extractor = DocExtractor()
        try:
            ents = extractor.extract_entities_from_text(
                "Marie Curie discovered radium",
                tenant_id="t1",
                source_doc_id="doc1.md",
                segment_anchor=_doc_anchor(),
            )
        finally:
            server.shutdown()

        assert seen["path"] == "/predict_entities"
        assert seen["payload"] == {
            "text": "Marie Curie discovered radium",
            "labels": [
                "Person",
                "Organization",
                "Location",
                "Date",
                "Substance",
                "Award",
                "Field",
                "Event",
                "Concept",
                "Technology",
                "Product",
                "Algorithm",
                "Model",
                "Framework",
                "Language",
            ],
            "threshold": 0.3,
            "model": "urchade/gliner_large-v2.1",
        }
        assert [(n.name, n.label) for n in ents.nodes] == [
            ("Marie Curie", "Person"),
            ("radium", "Substance"),
        ]
        assert ents.per_chunk_entity_names == [["Marie Curie", "radium"]]

    def test_returns_none_for_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("def foo(): pass")
            result = DocExtractor().extract(f, "t1", "sample.py")
        assert result is None


@pytest.mark.unit
@pytest.mark.ci_fast
class TestIsBlockedEntity:
    """Entity-candidate noise filter — pronouns, verbs, and verb phrases."""

    def test_blocks_bare_pronoun(self):
        assert _is_blocked_entity("She") is True

    def test_blocks_bare_verb(self):
        assert _is_blocked_entity("discovered") is True

    def test_blocks_adverb_plus_verb_phrase(self):
        # GLiNER emits "later won" as an Event span; it is verb-phrase noise.
        assert _is_blocked_entity("later won") is True

    def test_blocks_then_verb_phrase(self):
        assert _is_blocked_entity("then discovered") is True

    def test_keeps_person_entity(self):
        assert _is_blocked_entity("Marie Curie") is False

    def test_keeps_award_entity(self):
        assert _is_blocked_entity("Nobel Prize") is False

    def test_keeps_substance_entity(self):
        assert _is_blocked_entity("radium") is False

    def test_keeps_multiword_proper_noun_with_no_blocked_tokens(self):
        # Neither token is pronoun/verb/adverb noise — a real place name.
        assert _is_blocked_entity("New York") is False
