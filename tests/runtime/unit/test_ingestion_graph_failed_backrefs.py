"""Failed graph upserts must not leave dangling provenance back-refs.

GraphManager.upsert returns failed_ids (the node.doc_id / edge.doc_id it could
not persist). The per-segment ingestion path PATCHes back-refs onto content
docs; before this it wrote back-refs for every extracted id regardless of
whether the upsert landed it, so a partial KG loss left content docs pointing
at nodes/edges that never persisted. _prune_failed_backrefs maps the failed
doc_ids back to the bare node_id / edge_id the buckets key on and drops them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogniverse_runtime.routers.ingestion import _prune_failed_backrefs

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _node(node_id, doc_id):
    return SimpleNamespace(node_id=node_id, doc_id=doc_id)


def _edge(edge_id, doc_id):
    return SimpleNamespace(edge_id=edge_id, doc_id=doc_id)


def test_prunes_failed_node_and_edge_backrefs():
    linked = SimpleNamespace(
        nodes=[_node("n1", "kg_node_t_n1"), _node("n2", "kg_node_t_n2")],
        edges=[_edge("e1", "kg_edge_t_e1"), _edge("e2", "kg_edge_t_e2")],
    )
    backrefs = {
        "seg1": {
            "entity_ids": ["n1", "n2"],
            "relation_ids": ["e1", "e2"],
            "claim_ids": ["e1", "e2"],
        }
    }
    # n2 and e2 failed to persist (their doc_ids are in the upsert failure list).
    failed_doc_ids = {"kg_node_t_n2", "kg_edge_t_e2"}

    pruned = _prune_failed_backrefs(backrefs, linked, failed_doc_ids)

    assert pruned == {"n2", "e2"}
    assert backrefs["seg1"]["entity_ids"] == ["n1"]
    assert backrefs["seg1"]["relation_ids"] == ["e1"]
    assert backrefs["seg1"]["claim_ids"] == ["e1"]


def test_no_failures_leaves_backrefs_untouched():
    linked = SimpleNamespace(
        nodes=[_node("n1", "kg_node_t_n1")],
        edges=[_edge("e1", "kg_edge_t_e1")],
    )
    backrefs = {
        "seg1": {"entity_ids": ["n1"], "relation_ids": ["e1"], "claim_ids": ["e1"]}
    }

    pruned = _prune_failed_backrefs(backrefs, linked, set())

    assert pruned == set()
    assert backrefs["seg1"]["entity_ids"] == ["n1"]
    assert backrefs["seg1"]["relation_ids"] == ["e1"]


def test_failed_doc_id_absent_from_linked_prunes_nothing():
    linked = SimpleNamespace(nodes=[_node("n1", "kg_node_t_n1")], edges=[])
    backrefs = {"seg1": {"entity_ids": ["n1"], "relation_ids": [], "claim_ids": []}}

    pruned = _prune_failed_backrefs(backrefs, linked, {"kg_node_t_unknown"})

    assert pruned == set()
    assert backrefs["seg1"]["entity_ids"] == ["n1"]
