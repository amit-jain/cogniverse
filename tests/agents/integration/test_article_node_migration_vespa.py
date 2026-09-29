"""Article-node merge migration against real Vespa.

A graph ingested before claim extraction mapped "the Sorbonne" onto the
"Sorbonne" hint holds ``the_sorbonne`` where new ingests write ``sorbonne``.
``merge_article_nodes`` folds the first into the second: edges re-pointed,
identical edges collapsed with their provenance merged, the article node's
mentions unioned into its twin, the content docs' back-refs rewritten to the
surviving ids, and the article node deleted last.

Backed by the project-wide ``shared_memory_vespa`` container: two tenants'
``knowledge_graph`` schemas and tenant A's ``document_text`` schema are
deployed through SchemaRegistry, and every test starts from empty graphs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import requests

from cogniverse_agents.graph.article_node_migration import merge_article_nodes
from cogniverse_agents.graph.graph_schema import Edge, Mention, Node
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

pytestmark = [pytest.mark.integration]

TENANT_A = "kgmig_a:kgmig_a"
TENANT_B = "kgmig_b:kgmig_b"
GRAPH_A = schema_full_name("knowledge_graph", TENANT_A)
GRAPH_B = schema_full_name("knowledge_graph", TENANT_B)
CONTENT_A = schema_full_name("document_text", TENANT_A)
_GRAPH_NS = "graph_content"
_CONTENT_NS = "content"

_STUDIED = "Marie Curie studied at the Sorbonne."
_LOCATED = "The Sorbonne is in Paris."


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def migration_vespa(shared_memory_vespa):
    for tenant, base in (
        (TENANT_A, "knowledge_graph"),
        (TENANT_B, "knowledge_graph"),
        (TENANT_A, "document_text"),
    ):
        deploy_tenant_schema(
            shared_memory_vespa,
            tenant_id=tenant,
            base_schema_name=base,
            config_manager=shared_memory_vespa["config_manager"],
        )
    return shared_memory_vespa


@pytest.fixture(scope="module")
def resolve_backend(migration_vespa):
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    def resolve():
        return BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id=SYSTEM_TENANT_ID,
            config={
                "backend": {
                    "url": "http://localhost",
                    "port": migration_vespa["http_port"],
                    "config_port": migration_vespa["config_port"],
                }
            },
            config_manager=migration_vespa["config_manager"],
            schema_loader=schema_loader,
        )

    return resolve


@pytest.fixture
def vespa(migration_vespa, resolve_backend):
    """A seeding/inspection handle over empty test graphs and content."""
    handle = _Vespa(migration_vespa["http_port"], resolve_backend)
    handle.wipe()
    yield handle
    handle.wipe()


class _Vespa:
    def __init__(self, port: int, resolve) -> None:
        self.base = f"http://localhost:{port}"
        self.resolve = resolve

    # -- seeding ---------------------------------------------------------

    def put_node(self, tenant: str, name: str, mentions: List[Mention]) -> Node:
        node = Node(
            tenant_id=tenant,
            name=name,
            description=f"Entity mentioned in {mentions[0].source_doc_id}",
            label="Organization",
            mentions=mentions,
            created_at="2026-08-01T00:00:00+00:00",
            updated_at="2026-08-01T00:00:00+00:00",
        )
        self.resolve().put_document_fields(
            node.doc_id,
            node.to_vespa_document()["fields"],
            schema_name=_graph_schema(tenant),
            namespace=_GRAPH_NS,
        )
        return node

    def put_edge(self, edge: Edge) -> Edge:
        self.resolve().put_document_fields(
            edge.doc_id,
            edge.to_vespa_document()["fields"],
            schema_name=_graph_schema(edge.tenant_id),
            namespace=_GRAPH_NS,
        )
        return edge

    def put_content(
        self,
        doc_id: str,
        entity_ids: List[str],
        relation_ids: List[str],
        claim_ids: List[str],
    ) -> None:
        self.resolve().put_document_fields(
            doc_id,
            {
                "document_id": doc_id,
                "entity_ids": entity_ids,
                "relation_ids": relation_ids,
                "claim_ids": claim_ids,
            },
            schema_name=CONTENT_A,
            namespace=_CONTENT_NS,
        )

    # -- inspection ------------------------------------------------------

    def _visit(self, namespace: str, schema: str) -> Dict[str, Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        continuation: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"wantedDocumentCount": 500}
            if continuation:
                params["continuation"] = continuation
            resp = requests.get(
                f"{self.base}/document/v1/{namespace}/{schema}/docid",
                params=params,
                timeout=30,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            for doc in body.get("documents", []):
                fields = {
                    k: v
                    for k, v in doc.get("fields", {}).items()
                    if not k.startswith("embedding")
                }
                docs[doc["id"].split("::", 1)[1]] = fields
            continuation = body.get("continuation")
            if not continuation:
                return docs

    def graph(self, tenant: str) -> Dict[str, Dict[str, Any]]:
        return self._visit(_GRAPH_NS, _graph_schema(tenant))

    def content(self) -> Dict[str, Dict[str, Any]]:
        return self._visit(_CONTENT_NS, CONTENT_A)

    def snapshot(self) -> Tuple[Dict, Dict, Dict]:
        return self.graph(TENANT_A), self.graph(TENANT_B), self.content()

    def wipe(self) -> None:
        for namespace, schema in (
            (_GRAPH_NS, GRAPH_A),
            (_GRAPH_NS, GRAPH_B),
            (_CONTENT_NS, CONTENT_A),
        ):
            for doc_id in self._visit(namespace, schema):
                resp = requests.delete(
                    f"{self.base}/document/v1/{namespace}/{schema}/docid/{doc_id}",
                    timeout=30,
                )
                assert resp.status_code == 200, resp.text


def _graph_schema(tenant: str) -> str:
    return GRAPH_A if tenant == TENANT_A else GRAPH_B


def _mention(source_doc_id: str, segment_id: str, span: str) -> Mention:
    return Mention(
        source_doc_id=source_doc_id,
        segment_id=segment_id,
        ts_start=0.0,
        ts_end=30.0,
        modality="transcript",
        evidence_span=span,
    )


def _edge(
    tenant: str,
    source: str,
    relation: str,
    target: str,
    *,
    segment_id: str,
    source_doc_id: str,
    evidence: str,
    provenance: str = "EXTRACTED",
    confidence: float = 0.75,
    created_at: str = "2026-08-01T00:00:00+00:00",
) -> Edge:
    return Edge(
        tenant_id=tenant,
        source=source,
        target=target,
        relation=relation,
        evidence_span=evidence,
        segment_id=segment_id,
        ts_start=0.0,
        ts_end=30.0,
        modality="transcript",
        provenance=provenance,
        source_doc_id=source_doc_id,
        confidence=confidence,
        created_at=created_at,
    )


def _edges(graph: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in graph.items() if v["doc_type"] == "edge"}


def _nodes(graph: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in graph.items() if v["doc_type"] == "node"}


def _bare(edge: Edge) -> str:
    return edge.edge_id


# --------------------------------------------------------------------- #
# The Sorbonne scenario: an article node with edges and back-refs        #
# --------------------------------------------------------------------- #


def _seed_sorbonne(vespa: _Vespa) -> Dict[str, Edge]:
    vespa.put_node(TENANT_A, "Sorbonne", [_mention("docA", "seg_1", _STUDIED)])
    vespa.put_node(TENANT_A, "the Sorbonne", [_mention("docA", "seg_2", _LOCATED)])
    vespa.put_node(TENANT_A, "Marie Curie", [_mention("docA", "seg_1", _STUDIED)])
    vespa.put_node(TENANT_A, "Paris", [_mention("docA", "seg_2", _LOCATED)])
    studied = vespa.put_edge(
        _edge(
            TENANT_A,
            "Marie Curie",
            "studied_at",
            "the Sorbonne",
            segment_id="seg_1",
            source_doc_id="docA",
            evidence=_STUDIED,
        )
    )
    located = vespa.put_edge(
        _edge(
            TENANT_A,
            "the Sorbonne",
            "located_in",
            "Paris",
            segment_id="seg_2",
            source_doc_id="docA",
            evidence=_LOCATED,
        )
    )
    vespa.put_content(
        "docA_seg_1",
        ["marie_curie", "the_sorbonne"],
        [_bare(studied)],
        [_bare(studied)],
    )
    vespa.put_content(
        "docA_seg_2",
        ["the_sorbonne", "paris"],
        [_bare(located)],
        [_bare(located)],
    )
    return {"studied": studied, "located": located}


_SORBONNE_REPORT = {
    "tenant_id": TENANT_A,
    "merges": [
        {
            "from": "the_sorbonne",
            "into": "sorbonne",
            "node_doc": True,
            "edges_repointed": 2,
            "edges_deduped": 0,
        }
    ],
    "skipped": [],
    "edges_repointed": 2,
    "edges_deduped": 0,
    "mentions_added": 1,
    "content_docs_updated": 2,
}


def _assert_sorbonne_merged(vespa: _Vespa) -> None:
    graph = vespa.graph(TENANT_A)
    studied = _edge(
        TENANT_A,
        "Marie Curie",
        "studied_at",
        "Sorbonne",
        segment_id="seg_1",
        source_doc_id="docA",
        evidence=_STUDIED,
    )
    located = _edge(
        TENANT_A,
        "Sorbonne",
        "located_in",
        "Paris",
        segment_id="seg_2",
        source_doc_id="docA",
        evidence=_LOCATED,
    )

    assert sorted(_nodes(graph)) == [
        "kg_node_kgmig_a_kgmig_a_marie_curie",
        "kg_node_kgmig_a_kgmig_a_paris",
        "kg_node_kgmig_a_kgmig_a_sorbonne",
    ]
    sorbonne = graph["kg_node_kgmig_a_kgmig_a_sorbonne"]
    assert sorbonne["name"] == "Sorbonne"
    assert sorbonne["label"] == "Organization"
    assert [
        (m["source_doc_id"], m["segment_id"], m["evidence_span"])
        for m in json.loads(sorbonne["mentions"])
    ] == [("docA", "seg_1", _STUDIED), ("docA", "seg_2", _LOCATED)]

    edges = _edges(graph)
    assert sorted(edges) == sorted([studied.doc_id, located.doc_id])
    for expected in (studied, located):
        fields = edges[expected.doc_id]
        assert fields["doc_id"] == expected.doc_id
        assert (
            fields["source_node_id"],
            fields["relation"],
            fields["target_node_id"],
            fields["segment_id"],
            fields["source_doc_id"],
            fields["evidence_span"],
            fields["provenance"],
            fields["created_at"],
        ) == (
            expected.source_node_id,
            expected.relation,
            expected.target_node_id,
            expected.segment_id,
            "docA",
            expected.evidence_span,
            "EXTRACTED",
            "2026-08-01T00:00:00+00:00",
        )
        assert fields["confidence"] == pytest.approx(0.75)

    content = vespa.content()
    assert {
        doc_id: (f["entity_ids"], f["relation_ids"], f["claim_ids"])
        for doc_id, f in content.items()
    } == {
        "docA_seg_1": (
            ["marie_curie", "sorbonne"],
            [studied.edge_id],
            [studied.edge_id],
        ),
        "docA_seg_2": (
            ["sorbonne", "paris"],
            [located.edge_id],
            [located.edge_id],
        ),
    }


@pytest.mark.integration
class TestArticleNodeMigration:
    def test_merge_repoints_edges_and_back_refs_onto_the_twin(
        self, vespa, resolve_backend
    ):
        _seed_sorbonne(vespa)

        reports = merge_article_nodes(
            resolve_backend, tenant_ids=[TENANT_A], apply=True
        )

        assert [r.to_dict() for r in reports] == [{**_SORBONNE_REPORT, "applied": True}]
        _assert_sorbonne_merged(vespa)

    def test_identical_edges_dedupe_and_keep_every_provenance(
        self, vespa, resolve_backend
    ):
        vespa.put_node(TENANT_A, "Sorbonne", [_mention("docA", "seg_1", _STUDIED)])
        vespa.put_node(TENANT_A, "Marie Curie", [_mention("docA", "seg_1", _STUDIED)])
        survivor = vespa.put_edge(
            _edge(
                TENANT_A,
                "Marie Curie",
                "studied_at",
                "Sorbonne",
                segment_id="seg_1",
                source_doc_id="docA",
                evidence=_STUDIED,
                provenance="INFERRED",
                confidence=0.5,
                created_at="2026-09-02T00:00:00+00:00",
            )
        )
        duplicate = vespa.put_edge(
            _edge(
                TENANT_A,
                "Marie Curie",
                "studied_at",
                "the Sorbonne",
                segment_id="seg_1",
                source_doc_id="docA",
                evidence=_STUDIED,
                provenance="EXTRACTED",
                confidence=0.875,
                created_at="2026-08-01T00:00:00+00:00",
            )
        )
        # Segment ids repeat across documents, so docB's "a Sorbonne" edge
        # re-points onto the survivor's canonical id, yet its grounding is its
        # own and must survive the merge.
        other_source = vespa.put_edge(
            _edge(
                TENANT_A,
                "Marie Curie",
                "studied_at",
                "a Sorbonne",
                segment_id="seg_1",
                source_doc_id="docB",
                evidence="Curie enrolled at a Sorbonne college.",
                confidence=0.625,
            )
        )
        vespa.put_content(
            "docA_seg_1",
            ["marie_curie", "sorbonne"],
            [survivor.edge_id, duplicate.edge_id],
            [survivor.edge_id, duplicate.edge_id],
        )
        vespa.put_content(
            "docB_seg_1",
            ["marie_curie"],
            [other_source.edge_id],
            [other_source.edge_id],
        )

        reports = merge_article_nodes(
            resolve_backend, tenant_ids=[TENANT_A], apply=True
        )

        assert [r.to_dict() for r in reports] == [
            {
                "tenant_id": TENANT_A,
                "applied": True,
                "merges": [
                    {
                        "from": "a_sorbonne",
                        "into": "sorbonne",
                        "node_doc": False,
                        "edges_repointed": 1,
                        "edges_deduped": 0,
                    },
                    {
                        "from": "the_sorbonne",
                        "into": "sorbonne",
                        "node_doc": False,
                        "edges_repointed": 1,
                        "edges_deduped": 1,
                    },
                ],
                "skipped": [],
                "edges_repointed": 2,
                "edges_deduped": 1,
                "mentions_added": 0,
                "content_docs_updated": 1,
            }
        ]
        edges = _edges(vespa.graph(TENANT_A))
        assert sorted(edges) == sorted([survivor.doc_id, other_source.doc_id])
        merged = edges[survivor.doc_id]
        assert (
            merged["target_node_id"],
            merged["source_doc_id"],
            merged["evidence_span"],
            merged["provenance"],
            merged["created_at"],
        ) == ("sorbonne", "docA", _STUDIED, "EXTRACTED", "2026-08-01T00:00:00+00:00")
        assert merged["confidence"] == pytest.approx(0.875)
        kept = edges[other_source.doc_id]
        assert (
            kept["source_node_id"],
            kept["target_node_id"],
            kept["source_doc_id"],
            kept["evidence_span"],
            kept["provenance"],
        ) == (
            "marie_curie",
            "sorbonne",
            "docB",
            "Curie enrolled at a Sorbonne college.",
            "EXTRACTED",
        )
        assert kept["confidence"] == pytest.approx(0.625)
        content = vespa.content()
        assert (
            content["docA_seg_1"]["relation_ids"],
            content["docA_seg_1"]["claim_ids"],
            content["docB_seg_1"]["relation_ids"],
            content["docB_seg_1"]["claim_ids"],
        ) == (
            [survivor.edge_id],
            [survivor.edge_id],
            [other_source.edge_id],
            [other_source.edge_id],
        )

    def test_article_node_without_a_twin_is_left_alone(self, vespa, resolve_backend):
        vespa.put_node(TENANT_A, "the Louvre", [_mention("docL", "seg_0", "x")])
        vespa.put_node(TENANT_A, "Mona Lisa", [_mention("docL", "seg_0", "x")])
        vespa.put_edge(
            _edge(
                TENANT_A,
                "Mona Lisa",
                "located_in",
                "the Louvre",
                segment_id="seg_0",
                source_doc_id="docL",
                evidence="x",
            )
        )
        before = vespa.snapshot()

        reports = merge_article_nodes(
            resolve_backend, tenant_ids=[TENANT_A], apply=True
        )

        assert [r.to_dict() for r in reports] == [
            {
                "tenant_id": TENANT_A,
                "applied": True,
                "merges": [],
                "skipped": ["the_louvre"],
                "edges_repointed": 0,
                "edges_deduped": 0,
                "mentions_added": 0,
                "content_docs_updated": 0,
            }
        ]
        assert vespa.snapshot() == before

    def test_article_node_never_merges_into_another_tenants_twin(
        self, vespa, resolve_backend
    ):
        vespa.put_node(TENANT_A, "the Nile", [_mention("docN", "seg_0", "x")])
        vespa.put_node(TENANT_B, "Nile", [_mention("docN", "seg_0", "x")])
        vespa.put_node(TENANT_B, "the Seine", [_mention("docS", "seg_0", "y")])
        vespa.put_node(TENANT_B, "Seine", [_mention("docS", "seg_1", "y")])
        before = vespa.snapshot()

        both = merge_article_nodes(
            resolve_backend, tenant_ids=[TENANT_A], apply=True
        ) + merge_article_nodes(resolve_backend, tenant_ids=[TENANT_A], apply=False)

        assert [(r.tenant_id, r.merges, r.skipped) for r in both] == [
            (TENANT_A, [], ["the_nile"]),
            (TENANT_A, [], ["the_nile"]),
        ]
        assert vespa.snapshot() == before

        reports = merge_article_nodes(
            resolve_backend, tenant_ids=[TENANT_B], apply=True
        )

        assert [
            (r.tenant_id, [m.to_dict() for m in r.merges], r.skipped) for r in reports
        ] == [
            (
                TENANT_B,
                [
                    {
                        "from": "the_seine",
                        "into": "seine",
                        "node_doc": True,
                        "edges_repointed": 0,
                        "edges_deduped": 0,
                    }
                ],
                [],
            )
        ]
        graph_a, graph_b, _ = vespa.snapshot()
        assert graph_a == before[0]
        assert sorted(graph_b) == [
            "kg_node_kgmig_b_kgmig_b_nile",
            "kg_node_kgmig_b_kgmig_b_seine",
        ]

    def test_second_run_changes_nothing(self, vespa, resolve_backend):
        _seed_sorbonne(vespa)
        merge_article_nodes(resolve_backend, tenant_ids=[TENANT_A], apply=True)
        after_first = vespa.snapshot()

        reports = merge_article_nodes(
            resolve_backend, tenant_ids=[TENANT_A], apply=True
        )

        assert [r.to_dict() for r in reports] == [
            {
                "tenant_id": TENANT_A,
                "applied": True,
                "merges": [],
                "skipped": [],
                "edges_repointed": 0,
                "edges_deduped": 0,
                "mentions_added": 0,
                "content_docs_updated": 0,
            }
        ]
        assert vespa.snapshot() == after_first

    def test_dry_run_reports_the_plan_and_changes_nothing(self, vespa, resolve_backend):
        _seed_sorbonne(vespa)
        before = vespa.snapshot()

        scoped = merge_article_nodes(resolve_backend, tenant_ids=[TENANT_A])
        every_tenant = merge_article_nodes(resolve_backend, apply=False)

        assert [r.to_dict() for r in scoped] == [{**_SORBONNE_REPORT, "applied": False}]
        by_tenant = {r.tenant_id: r.to_dict() for r in every_tenant}
        assert by_tenant[TENANT_A] == {**_SORBONNE_REPORT, "applied": False}
        assert by_tenant[TENANT_B]["merges"] == []
        assert vespa.snapshot() == before

    @pytest.mark.parametrize(
        ("method", "succeed_first"),
        [
            ("put_document_fields", 1),
            ("update_document_fields", 0),
            ("update_document_fields", 1),
            ("delete_document_fields", 1),
            ("delete_document_fields", 2),
        ],
    )
    def test_interrupted_run_resumes_to_the_same_result(
        self, vespa, resolve_backend, method, succeed_first
    ):
        _seed_sorbonne(vespa)
        failing = _FailingBackend(resolve_backend(), method, succeed_first)

        with pytest.raises(RuntimeError, match="interrupted"):
            merge_article_nodes(lambda: failing, tenant_ids=[TENANT_A], apply=True)
        assert failing.calls == succeed_first + 1

        merge_article_nodes(resolve_backend, tenant_ids=[TENANT_A], apply=True)

        _assert_sorbonne_merged(vespa)
        settled = vespa.snapshot()
        reports = merge_article_nodes(
            resolve_backend, tenant_ids=[TENANT_A], apply=True
        )
        assert [(r.merges, r.skipped) for r in reports] == [([], [])]
        assert vespa.snapshot() == settled


class _FailingBackend:
    """The real backend, except that one write method fails after
    ``succeed_first`` successful calls — a process dying mid-migration."""

    def __init__(self, inner: Any, method: str, succeed_first: int) -> None:
        self._inner = inner
        self._method = method
        self._succeed_first = succeed_first
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != self._method:
            return attr

        def fail_after(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls > self._succeed_first:
                raise RuntimeError(f"interrupted at {name} call {self.calls}")
            return attr(*args, **kwargs)

        return fail_after
