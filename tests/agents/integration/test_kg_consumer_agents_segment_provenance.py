"""KG-consumer agent integration tests against real GraphManager+Vespa.

The nine KG-aware agents must surface segment-level provenance correctly
through their public ``.<verb>(...)`` methods, against a pre-ingested set
of four fixtures (``marie_curie_30s``, ``curie_sorbonne_60s``,
``curie_birth_v1``, ``curie_birth_v2``).

Every test pins its output against a hand-reviewed golden file via
byte-equal JSON / text comparison. Re-record with ``RECORD_GOLDEN=1``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List

import pytest

from cogniverse_agents.audit_explanation_agent import (
    AuditExplanationAgent,
    AuditExplanationDeps,
)
from cogniverse_agents.citation_tracing_agent import (
    CitationTracingAgent,
    CitationTracingDeps,
)
from cogniverse_agents.contradiction_reconciliation_agent import (
    ContradictionReconciliationAgent,
    ContradictionReconciliationDeps,
)
from cogniverse_agents.cross_tenant_comparison_agent import (
    CrossTenantComparisonAgent,
    CrossTenantComparisonDeps,
)
from cogniverse_agents.federated_query_agent import (
    FederatedQueryAgent,
    FederatedQueryDeps,
)
from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Mention,
    Node,
)
from cogniverse_agents.kg_traversal_agent import (
    KGTraversalDeps,
    KnowledgeGraphTraversalAgent,
)
from cogniverse_agents.knowledge_summarization_agent import (
    KnowledgeSummarizationAgent,
    KnowledgeSummarizationDeps,
)
from cogniverse_agents.multi_document_synthesis_agent import (
    MultiDocSynthesisDeps,
    MultiDocumentSynthesisAgent,
)
from cogniverse_agents.temporal_reasoning_agent import (
    TemporalReasoningAgent,
    TemporalReasoningDeps,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration


# Reuse the canonical ``graph_manager`` fixture from the existing graph
# Vespa integration suite (it owns the Vespa container + pylate sidecar
# bring-up). Pytest discovers it via the shared conftest hierarchy.
# Re-export fixtures from the canonical graph Vespa integration suite so
# pytest discovers them in this module's namespace. The ``_reexport`` prefix
# keeps the imports out of the way; tests reference the fixtures by their
# canonical names via the assignments below.
from tests.agents.integration.test_graph_vespa_integration import (  # noqa: E402
    graph_manager as _reexport_graph_manager,
)
from tests.agents.integration.test_graph_vespa_integration import (
    graph_vespa as _reexport_graph_vespa,
)
from tests.agents.integration.test_graph_vespa_integration import (
    pylate_server as _reexport_pylate_server,
)

graph_manager = _reexport_graph_manager
graph_vespa = _reexport_graph_vespa
pylate_server = _reexport_pylate_server

# ---------------------------------------------------------------------------
# Golden file helper
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "goldens"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def assert_golden_json(actual: Any, name: str) -> None:
    """Byte-equal assertion for JSON-serialisable structures."""
    path = GOLDEN_DIR / name
    actual_json = _canonical_json(actual)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    if not path.exists():
        pytest.fail(
            f"Golden file missing: {path}. Re-run with RECORD_GOLDEN=1 to record."
        )
    expected = path.read_text().rstrip("\n")
    assert actual_json == expected, (
        f"Golden mismatch for {name}\n--- expected ---\n{expected}\n"
        f"--- actual ---\n{actual_json}"
    )


def assert_golden_text(actual: str, name: str) -> None:
    """Byte-equal assertion for plain-text goldens."""
    path = GOLDEN_DIR / name
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual + ("\n" if not actual.endswith("\n") else ""))
        return
    if not path.exists():
        pytest.fail(
            f"Golden file missing: {path}. Re-run with RECORD_GOLDEN=1 to record."
        )
    expected = path.read_text().rstrip("\n")
    assert actual.rstrip("\n") == expected, (
        f"Golden mismatch for {name}\n--- expected ---\n{expected}\n"
        f"--- actual ---\n{actual}"
    )


# ---------------------------------------------------------------------------
# Pre-ingested fixture: build the four Curie clips once per module
# ---------------------------------------------------------------------------

_TENANT = "test_tenant"
_PINNED_TS = "2026-05-19T00:00:00+00:00"


def _mention(
    source_doc_id: str,
    segment_id: str,
    ts_start: float,
    ts_end: float,
    text: str,
    modality: str = "transcript",
) -> Mention:
    return Mention(
        source_doc_id=source_doc_id,
        segment_id=segment_id,
        ts_start=ts_start,
        ts_end=ts_end,
        modality=modality,
        evidence_span=text,
    )


def _edge(
    source: str,
    relation: str,
    target: str,
    source_doc_id: str,
    segment_id: str,
    ts_start: float,
    ts_end: float,
    evidence_span: str,
    confidence: float,
    modality: str = "transcript",
) -> Edge:
    return Edge(
        tenant_id=_TENANT,
        source=source,
        target=target,
        relation=relation,
        evidence_span=evidence_span,
        segment_id=segment_id,
        ts_start=ts_start,
        ts_end=ts_end,
        modality=modality,
        provenance="EXTRACTED",
        source_doc_id=source_doc_id,
        confidence=confidence,
        created_at=_PINNED_TS,
    )


def _build_curie_extraction() -> ExtractionResult:
    """Assemble the four-clip Curie KG fixture as a single ExtractionResult.

    Shared fixture layout:
      * ``marie_curie_30s`` seg_3 (12.0–18.5): discovered radium in 1898 at Sorbonne.
      * ``marie_curie_30s`` seg_4 (18.5–25.0): won Nobel Prize in Physics.
      * ``curie_sorbonne_60s`` seg_2 (10.0–20.0): professor at the Sorbonne in Paris.
      * ``curie_birth_v1`` seg_1 (0.0–10.0): born in Paris.
      * ``curie_birth_v2`` seg_1 (0.0–10.0): born in Warsaw.
    """
    span_30s_3 = "Marie Curie discovered radium in 1898 at the Sorbonne."
    span_30s_4 = "She later won the Nobel Prize in Physics."
    span_sorb = "Marie Curie was a professor at the Sorbonne in Paris."
    span_birth_paris = "Marie Curie was born in Paris."
    span_birth_warsaw = "Marie Curie was born in Warsaw."

    mentions_marie_curie = [
        _mention("marie_curie_30s", "seg_3", 12.0, 18.5, span_30s_3),
        _mention("marie_curie_30s", "seg_4", 18.5, 25.0, span_30s_4),
        _mention("curie_sorbonne_60s", "seg_2", 10.0, 20.0, span_sorb),
        _mention("curie_birth_v1", "seg_1", 0.0, 10.0, span_birth_paris),
        _mention("curie_birth_v2", "seg_1", 0.0, 10.0, span_birth_warsaw),
    ]

    nodes: List[Node] = [
        Node(
            tenant_id=_TENANT,
            name="Marie Curie",
            mentions=mentions_marie_curie,
            kind="entity",
            created_at=_PINNED_TS,
            updated_at=_PINNED_TS,
        ),
        Node(
            tenant_id=_TENANT,
            name="radium",
            mentions=[
                _mention("marie_curie_30s", "seg_3", 12.0, 18.5, span_30s_3),
            ],
            created_at=_PINNED_TS,
            updated_at=_PINNED_TS,
        ),
        Node(
            tenant_id=_TENANT,
            name="1898",
            mentions=[
                _mention("marie_curie_30s", "seg_3", 12.0, 18.5, span_30s_3),
            ],
            created_at=_PINNED_TS,
            updated_at=_PINNED_TS,
        ),
        Node(
            tenant_id=_TENANT,
            name="Sorbonne",
            mentions=[
                _mention("marie_curie_30s", "seg_3", 12.0, 18.5, span_30s_3),
                _mention("curie_sorbonne_60s", "seg_2", 10.0, 20.0, span_sorb),
            ],
            created_at=_PINNED_TS,
            updated_at=_PINNED_TS,
        ),
        Node(
            tenant_id=_TENANT,
            name="Nobel Prize",
            mentions=[
                _mention("marie_curie_30s", "seg_4", 18.5, 25.0, span_30s_4),
            ],
            created_at=_PINNED_TS,
            updated_at=_PINNED_TS,
        ),
        Node(
            tenant_id=_TENANT,
            name="Paris",
            mentions=[
                _mention("curie_sorbonne_60s", "seg_2", 10.0, 20.0, span_sorb),
                _mention("curie_birth_v1", "seg_1", 0.0, 10.0, span_birth_paris),
            ],
            created_at=_PINNED_TS,
            updated_at=_PINNED_TS,
        ),
        Node(
            tenant_id=_TENANT,
            name="Warsaw",
            mentions=[
                _mention("curie_birth_v2", "seg_1", 0.0, 10.0, span_birth_warsaw),
            ],
            created_at=_PINNED_TS,
            updated_at=_PINNED_TS,
        ),
    ]

    edges: List[Edge] = [
        _edge(
            "Marie Curie",
            "discovered",
            "radium",
            "marie_curie_30s",
            "seg_3",
            12.0,
            18.5,
            span_30s_3,
            0.92,
        ),
        _edge(
            "Marie Curie",
            "discovered_in",
            "1898",
            "marie_curie_30s",
            "seg_3",
            12.0,
            18.5,
            span_30s_3,
            0.87,
        ),
        _edge(
            "Marie Curie",
            "worked_at",
            "Sorbonne",
            "marie_curie_30s",
            "seg_3",
            12.0,
            18.5,
            span_30s_3,
            0.88,
        ),
        _edge(
            "Marie Curie",
            "won",
            "Nobel Prize",
            "marie_curie_30s",
            "seg_4",
            18.5,
            25.0,
            span_30s_4,
            0.85,
        ),
        _edge(
            "Marie Curie",
            "professor_at",
            "Sorbonne",
            "curie_sorbonne_60s",
            "seg_2",
            10.0,
            20.0,
            span_sorb,
            0.83,
        ),
        _edge(
            "Marie Curie",
            "born_in",
            "Paris",
            "curie_birth_v1",
            "seg_1",
            0.0,
            10.0,
            span_birth_paris,
            0.89,
        ),
        _edge(
            "Marie Curie",
            "born_in",
            "Warsaw",
            "curie_birth_v2",
            "seg_1",
            0.0,
            10.0,
            span_birth_warsaw,
            0.91,
        ),
    ]

    return ExtractionResult(
        source_doc_id="curie_fixture_bundle",
        nodes=nodes,
        edges=edges,
    )


@pytest.fixture(scope="module")
def ingested_curie_graph(graph_manager):
    """Idempotently ingest the four-clip Curie fixture into the test KG.

    Deterministic edge ids mean re-running tests against an already-populated
    Vespa is a no-op write; the fixture still re-upserts so the test is
    self-contained when run in isolation.
    """
    manager, _http_port = graph_manager
    extraction = _build_curie_extraction()
    counts = manager.upsert(extraction)
    logger.info(
        "ingested_curie_graph: nodes=%d edges=%d",
        counts.get("nodes_upserted", 0),
        counts.get("edges_upserted", 0),
    )
    return manager


# ---------------------------------------------------------------------------
# Per-agent assertions
# ---------------------------------------------------------------------------


def _edge_id_of(
    extraction: ExtractionResult, source: str, relation: str, target: str
) -> str:
    """Find the Edge with matching (source, relation, target) and return its edge_id."""
    from cogniverse_agents.graph.graph_schema import normalize_name

    src_id = normalize_name(source)
    tgt_id = normalize_name(target)
    for e in extraction.edges:
        if (
            e.source_node_id == src_id
            and e.relation == relation
            and e.target_node_id == tgt_id
        ):
            return e.edge_id
    raise KeyError(f"No edge for ({source}, {relation}, {target})")


class TestKGConsumerAgentsSegmentProvenance:
    """Every KG-consumer agent surfaces structured Mention/Edge data."""

    def test_kg_traversal_temporal_filter(self, ingested_curie_graph):
        """KGTraversalAgent.traverse with video_id + ts_range filter."""
        agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.traverse(
            "Marie Curie",
            filters={
                "video_id": "marie_curie_30s",
                "ts_range": (10.0, 20.0),
            },
        )
        assert_golden_json(result, "kg_traversal_curie_temporal.json")

    def test_kg_traversal_no_filter(self, ingested_curie_graph):
        """KGTraversalAgent.traverse without filters covers all four clips."""
        agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.traverse("Marie Curie")
        assert_golden_json(result, "kg_traversal_curie_all.json")

    def test_temporal_reasoning_compare_over_time(self, ingested_curie_graph):
        """TemporalReasoningAgent.compare_over_time yields ordered timeline."""
        agent = TemporalReasoningAgent(deps=TemporalReasoningDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.compare_over_time(
            node_name="Marie Curie",
            videos=["marie_curie_30s", "curie_sorbonne_60s", "curie_birth_v1"],
        )
        assert_golden_json(result, "temporal_reasoning_curie.json")

    def test_citation_tracing_trace(self, ingested_curie_graph):
        """CitationTracingAgent.trace(edge_id) returns one grounded step."""
        extraction = _build_curie_extraction()
        edge_id = _edge_id_of(extraction, "Marie Curie", "discovered", "radium")
        agent = CitationTracingAgent(deps=CitationTracingDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.trace(claim_id=edge_id)
        assert_golden_json(result, "citation_chain_discovered.json")

    def test_contradiction_reconciliation_detect(self, ingested_curie_graph):
        """ContradictionReconciliationAgent.detect groups conflicting Edges."""
        agent = ContradictionReconciliationAgent(deps=ContradictionReconciliationDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.detect(node_name="Marie Curie", predicate="born_in")
        assert_golden_json(result, "contradiction_curie_birth.json")

    def test_multi_document_synthesis_synthesize(self, ingested_curie_graph):
        """MultiDocumentSynthesisAgent.synthesize groups claims by video."""
        agent = MultiDocumentSynthesisAgent(deps=MultiDocSynthesisDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.synthesize(query="Marie Curie biography")
        assert_golden_json(result, "multidoc_synthesis_curie.json")

    def test_audit_explanation_explain(self, ingested_curie_graph):
        """AuditExplanationAgent.explain renders the canonical claim block."""
        extraction = _build_curie_extraction()
        edge_id = _edge_id_of(extraction, "Marie Curie", "discovered", "radium")
        agent = AuditExplanationAgent(deps=AuditExplanationDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.explain(answer_id=edge_id)
        assert_golden_text(result["text"], "audit_explanation_curie.txt")

    def test_knowledge_summarization_summarize(self, ingested_curie_graph):
        """KnowledgeSummarizationAgent.summarize emits per-segment lines."""
        agent = KnowledgeSummarizationAgent(deps=KnowledgeSummarizationDeps())
        agent.set_graph_manager(ingested_curie_graph)
        result = agent.summarize(video_id="marie_curie_30s")
        assert_golden_text(result["text"], "knowledge_summary_curie.txt")

    def test_federated_query_query(self, ingested_curie_graph):
        """FederatedQueryAgent.query merges nodes across overlays."""
        agent = FederatedQueryAgent(deps=FederatedQueryDeps())
        # Bind the same GraphManager under two logical names so the
        # federated path exercises the cross-source merge (the dedupe is
        # what's under test here, not the multi-Vespa fan-out).
        agent.set_graph_managers(
            {"acme": ingested_curie_graph, "acme_corp": ingested_curie_graph}
        )
        result = agent.query("Marie Curie", ["acme", "acme_corp"])
        assert_golden_json(result, "federated_curie.json")

    def test_cross_tenant_comparison_compare(self, ingested_curie_graph):
        """CrossTenantComparisonAgent.compare diffs tenant node sets."""
        agent = CrossTenantComparisonAgent(deps=CrossTenantComparisonDeps())
        # Same dedupe rationale as the federated query test — bind the
        # same manager under both tenant names. The diff will report all
        # nodes as shared which is captured in the locked golden (and
        # serves as the regression marker for any future symmetric-overlay
        # change).
        agent.set_graph_managers(
            {"acme": ingested_curie_graph, "globex": ingested_curie_graph}
        )
        result = agent.compare(tenant_a="acme", tenant_b="globex")
        assert_golden_json(result, "cross_tenant_curie.json")
