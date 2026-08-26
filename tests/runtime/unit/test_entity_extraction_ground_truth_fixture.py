"""Shape pins for the committed entity-extraction ground truth the e2e uploads.

The fixture is hand-labelled truth, not agent output. These pins fail when a
row stops being a verbatim span of its query, uses a type the agent cannot
emit, duplicates a pair or a query, or drifts from the evaluation corpus.
"""

from __future__ import annotations

import collections
import json
import pathlib

import pytest

from cogniverse_agents.entity_extraction_agent import (
    ENTITY_TYPES,
    EntityExtractionAgent,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "e2e" / "data" / "entity_extraction_ground_truth.json"
CORPUS = (
    REPO / "data" / "testset" / "evaluation" / "sample_videos_retrieval_queries.json"
)

EXPECTED_ROWS = 39
EXPECTED_ENTITIES = 84
EXPECTED_TYPE_COUNTS = {
    "PERSON": 33,
    "CONCEPT": 39,
    "PLACE": 8,
    "EVENT": 1,
    "TECHNOLOGY": 3,
}


def _rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_every_gliner_label_maps_into_the_declared_type_set():
    assert set(EntityExtractionAgent._GLINER_TYPE_MAP.values()) <= ENTITY_TYPES
    assert ENTITY_TYPES == {
        "PERSON",
        "ORGANIZATION",
        "CONCEPT",
        "PLACE",
        "EVENT",
        "TECHNOLOGY",
    }


def test_fixture_holds_the_exact_recorded_population():
    rows = _rows()
    entities = [entity for row in rows for entity in row["entities"]]
    assert len(rows) == EXPECTED_ROWS
    assert len(entities) == EXPECTED_ENTITIES
    assert (
        dict(collections.Counter(e["type"] for e in entities)) == EXPECTED_TYPE_COUNTS
    )


def test_every_entity_is_a_verbatim_span_of_its_query_with_a_declared_type():
    violations = [
        (index, entity["text"], entity["type"])
        for index, row in enumerate(_rows())
        for entity in row["entities"]
        if entity["text"] not in row["query"] or entity["type"] not in ENTITY_TYPES
    ]
    assert violations == []


def test_no_row_repeats_a_pair_and_no_query_repeats():
    rows = _rows()
    pair_repeats = [
        index
        for index, row in enumerate(rows)
        if len({(e["text"].casefold(), e["type"]) for e in row["entities"]})
        != len(row["entities"])
    ]
    query_counts = collections.Counter(row["query"] for row in rows)
    assert pair_repeats == []
    assert [q for q, n in query_counts.items() if n > 1] == []
    assert [row for row in rows if not row["entities"]] == []


def test_every_query_comes_from_the_evaluation_corpus():
    corpus = {
        (record.get("query") or "").strip() for record in json.loads(CORPUS.read_text())
    }
    assert [row["query"] for row in _rows() if row["query"] not in corpus] == []
