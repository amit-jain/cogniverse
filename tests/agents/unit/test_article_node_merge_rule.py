"""The rule deciding which article-prefixed KG ids merge, and into what.

Claim extraction stores an endpoint the model writes with a leading article
("the Sorbonne") under the entity hint it names ("Sorbonne"). A graph built
before that holds ``the_sorbonne`` beside ``sorbonne``; the migration merges
the first into the second only when the second is a node of the same tenant.
"""

import json

import pytest

from cogniverse_agents.graph.article_node_migration import (
    ARTICLE_ID_PREFIXES,
    article_twin_id,
    carries_same_grounding,
    merge_edge_provenance,
    plan_article_merges,
    validate_article_id,
)
from cogniverse_agents.graph.claim_extractor import _LEADING_ARTICLES
from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import Mention, Node, merge_mentions

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _mention(source_doc_id: str, segment_id: str, span: str = "e") -> Mention:
    return Mention(
        source_doc_id=source_doc_id,
        segment_id=segment_id,
        ts_start=0.0,
        ts_end=1.0,
        modality="transcript",
        evidence_span=span,
    )


class TestArticleTwinId:
    def test_prefixes_are_the_claim_extractors_articles_as_ids(self):
        assert ARTICLE_ID_PREFIXES == ("the_", "a_", "an_")
        assert ARTICLE_ID_PREFIXES == tuple(
            f"{article.strip()}_" for article in _LEADING_ARTICLES
        )

    @pytest.mark.parametrize(
        ("node_id", "twin"),
        [
            ("the_sorbonne", "sorbonne"),
            ("a_team", "team"),
            ("an_apple_pie", "apple_pie"),
            ("the_a_team", "a_team"),
        ],
    )
    def test_article_prefixed_id_names_its_bare_twin(self, node_id, twin):
        assert article_twin_id(node_id) == twin

    @pytest.mark.parametrize(
        "node_id",
        ["sorbonne", "theatre", "then_what", "anarchy", "the_", "a_", "an_", "the"],
    )
    def test_id_without_an_article_word_prefix_has_no_twin(self, node_id):
        assert article_twin_id(node_id) is None


class TestPlanArticleMerges:
    def test_node_merges_into_its_bare_twin_node(self):
        plan = plan_article_merges(
            node_ids={"sorbonne", "the_sorbonne", "marie_curie"},
            endpoint_ids={"the_sorbonne", "marie_curie"},
        )
        assert plan.merges == {"the_sorbonne": "sorbonne"}
        assert plan.skipped == []

    def test_article_node_without_a_twin_node_is_skipped(self):
        plan = plan_article_merges(
            node_ids={"the_louvre", "sorbonne"},
            endpoint_ids=set(),
        )
        assert plan.merges == {}
        assert plan.skipped == ["the_louvre"]

    def test_edge_endpoint_with_a_twin_node_merges_without_its_own_node(self):
        plan = plan_article_merges(
            node_ids={"sorbonne"},
            endpoint_ids={"the_sorbonne", "paris"},
        )
        assert plan.merges == {"the_sorbonne": "sorbonne"}
        assert plan.skipped == []

    def test_twin_that_is_only_an_edge_endpoint_is_not_a_merge_target(self):
        plan = plan_article_merges(
            node_ids=set(),
            endpoint_ids={"sorbonne", "the_sorbonne"},
        )
        assert plan.merges == {}
        assert plan.skipped == ["the_sorbonne"]

    def test_chain_resolves_to_the_final_bare_node(self):
        plan = plan_article_merges(
            node_ids={"team", "a_team", "the_a_team"},
            endpoint_ids=set(),
        )
        assert plan.merges == {"a_team": "team", "the_a_team": "team"}
        assert plan.skipped == []

    def test_skipped_ids_are_sorted_and_unique(self):
        plan = plan_article_merges(
            node_ids={"the_zoo", "an_owl"},
            endpoint_ids={"the_zoo", "a_bee"},
        )
        assert plan.skipped == ["a_bee", "an_owl", "the_zoo"]


class TestExclusions:
    def test_excluded_article_id_is_never_merged(self):
        plan = plan_article_merges(
            node_ids={"who", "the_who", "sorbonne", "the_sorbonne"},
            endpoint_ids={"the_who"},
            exclude={"the_who"},
        )
        assert plan.merges == {"the_sorbonne": "sorbonne"}
        assert plan.skipped == []
        assert plan.excluded == ["the_who"]

    def test_excluding_the_middle_link_stops_the_chain_there(self):
        plan = plan_article_merges(
            node_ids={"team", "a_team", "the_a_team"},
            endpoint_ids=set(),
            exclude={"a_team"},
        )
        assert plan.merges == {"the_a_team": "a_team"}
        assert plan.excluded == ["a_team"]

    def test_excluded_id_without_a_twin_is_listed_as_excluded_not_skipped(self):
        plan = plan_article_merges(
            node_ids={"the_louvre", "the_zoo"},
            endpoint_ids=set(),
            exclude={"the_louvre"},
        )
        assert plan.merges == {}
        assert plan.skipped == ["the_zoo"]
        assert plan.excluded == ["the_louvre"]

    def test_excluded_id_absent_from_the_graph_is_not_listed(self):
        plan = plan_article_merges(
            node_ids={"sorbonne", "the_sorbonne"},
            endpoint_ids=set(),
            exclude={"the_who"},
        )
        assert plan.merges == {"the_sorbonne": "sorbonne"}
        assert plan.excluded == []

    @pytest.mark.parametrize("article_id", ["the_who", "a_team", "an_apple_pie"])
    def test_article_node_id_is_a_valid_exclusion(self, article_id):
        assert validate_article_id(article_id) == article_id

    @pytest.mark.parametrize(
        "value",
        ["", "The Who", "the who", "the__who", "_the_who", "the_", "who", "theatre"],
    )
    def test_anything_but_an_article_node_id_is_rejected(self, value):
        with pytest.raises(ValueError, match="article node id"):
            validate_article_id(value)


class TestDuplicateEdgeMerge:
    def _edge(self, **overrides):
        fields = {
            "source_doc_id": "doc1",
            "evidence_span": "Marie Curie studied at the Sorbonne.",
            "modality": "transcript",
            "provenance": "INFERRED",
            "confidence": 0.6,
            "created_at": "2026-09-02T00:00:00+00:00",
        }
        fields.update(overrides)
        return fields

    def test_same_source_segment_text_is_the_same_grounding(self):
        assert carries_same_grounding(self._edge(), self._edge(provenance="EXTRACTED"))

    @pytest.mark.parametrize(
        "field, value",
        [
            ("source_doc_id", "doc2"),
            ("evidence_span", "She studied at the Sorbonne."),
            ("modality", "document"),
        ],
    )
    def test_a_different_source_or_span_is_different_grounding(self, field, value):
        assert not carries_same_grounding(self._edge(), self._edge(**{field: value}))

    def test_merge_keeps_extracted_highest_confidence_and_earliest_creation(self):
        survivor = self._edge()
        duplicate = self._edge(
            provenance="EXTRACTED",
            confidence=0.9,
            created_at="2026-08-01T00:00:00+00:00",
        )
        assert merge_edge_provenance(survivor, duplicate) == {
            "provenance": "EXTRACTED",
            "confidence": 0.9,
            "created_at": "2026-08-01T00:00:00+00:00",
        }

    def test_merge_never_downgrades_the_survivor(self):
        survivor = self._edge(
            provenance="EXTRACTED",
            confidence=0.8,
            created_at="2026-07-01T00:00:00+00:00",
        )
        duplicate = self._edge()
        assert merge_edge_provenance(survivor, duplicate) == {
            "provenance": "EXTRACTED",
            "confidence": 0.8,
            "created_at": "2026-07-01T00:00:00+00:00",
        }


class TestMergeMentions:
    def test_adds_only_mentions_the_target_lacks(self):
        target = [_mention("doc1", "seg_1")]
        added = merge_mentions(
            target, [_mention("doc1", "seg_1", span="other"), _mention("doc2", "seg_4")]
        )
        assert added == 1
        assert [(m.source_doc_id, m.segment_id) for m in target] == [
            ("doc1", "seg_1"),
            ("doc2", "seg_4"),
        ]

    def test_upsert_node_dedupe_unions_mentions_the_same_way(self):
        nodes = [
            Node(tenant_id="t:t", name="Sorbonne", mentions=[_mention("d", "s1")]),
            Node(
                tenant_id="t:t",
                name="sorbonne",
                mentions=[_mention("d", "s1"), _mention("d", "s2")],
            ),
        ]
        merged = GraphManager._merge_duplicate_nodes(None, nodes)
        assert [m.segment_id for m in merged[0].mentions] == ["s1", "s2"]
        assert (
            json.loads(merged[0].to_vespa_document()["fields"]["mentions"])[1][
                "segment_id"
            ]
            == "s2"
        )
