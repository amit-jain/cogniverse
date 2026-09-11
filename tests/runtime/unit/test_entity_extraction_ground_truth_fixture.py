"""Shape pins for the committed entity-extraction ground truth the e2e uploads.

The fixture is hand-labelled truth, not agent output. These pins fail when a
row stops being a verbatim span of its query, uses a type the agent cannot
emit, duplicates a pair or a query, leaves a type the agent emits without a
row, or drifts from the evaluation corpus.
"""

from __future__ import annotations

import collections
import json
import pathlib

import pytest

from cogniverse_agents.entity_extraction_agent import (
    ENTITY_TYPES,
    EntityExtractionAgent,
    EntityExtractionSignature,
)
from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
    canonicalize_entity_extraction_ground_truth_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "e2e" / "data" / "entity_extraction_ground_truth.json"
CORPUS = (
    REPO / "data" / "testset" / "evaluation" / "sample_videos_retrieval_queries.json"
)
PROCESSED = REPO / "data" / "testset" / "evaluation" / "processed"

# Rows whose query is a verbatim span of a committed transcript segment or
# frame description, in fixture order: query -> (file under PROCESSED, entry).
CAPTION_QUERIES = {
    "an emergency room doctor at Cleveland Clinic": (
        "transcripts/v_-IMXSEIabMM.json",
        3,
    ),
    "the New York City Fire Department": ("descriptions/v_-IMXSEIabMM.json", "184"),
    "a logo that appears to beassociated with Blender Institute": (
        "descriptions/big_buck_bunny_clip.json",
        "2142",
    ),
    "released by Coca-Cola": ("descriptions/elephant_dream_clip.json", "1913"),
}

EXPECTED_ROWS = 43
EXPECTED_ENTITIES = 90
EXPECTED_TYPE_COUNTS = {
    "PERSON": 34,
    "CONCEPT": 40,
    "PLACE": 8,
    "EVENT": 1,
    "TECHNOLOGY": 3,
    "ORGANIZATION": 4,
}


def _rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def _uncovered_types(rows: list[dict]) -> set[str]:
    return set(ENTITY_TYPES) - {
        entity["type"] for row in rows for entity in row["entities"]
    }


def _caption_text(relative: str, entry: int | str) -> str:
    value = json.loads((PROCESSED / relative).read_text())[entry]
    return value["text"] if isinstance(value, dict) else value


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


def test_entity_extraction_signature_outputs_entities_only():
    assert list(EntityExtractionSignature.output_fields.keys()) == ["entities"]


def test_fixture_holds_the_exact_recorded_population():
    rows = _rows()
    entities = [entity for row in rows for entity in row["entities"]]
    assert len(rows) == EXPECTED_ROWS
    assert len(entities) == EXPECTED_ENTITIES
    assert (
        dict(collections.Counter(e["type"] for e in entities)) == EXPECTED_TYPE_COUNTS
    )


def test_fixture_passes_the_upload_validator_unchanged():
    rows = _rows()
    assert len(rows) == EXPECTED_ROWS
    assert tuple(canonicalize_entity_extraction_ground_truth_rows(rows)) == tuple(rows)


def test_every_entity_is_a_verbatim_span_of_its_query_with_a_declared_type():
    violations = [
        (index, entity["text"], entity["type"])
        for index, row in enumerate(_rows())
        for entity in row["entities"]
        if entity["text"].casefold() not in row["query"].casefold()
        or entity["type"] not in ENTITY_TYPES
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


def test_every_type_the_agent_emits_has_a_truth_row():
    assert _uncovered_types(_rows()) == set()


def test_type_coverage_pin_fires_when_a_type_loses_its_rows():
    rows = [
        row
        for row in _rows()
        if all(entity["type"] != "ORGANIZATION" for entity in row["entities"])
    ]
    assert len(rows) == EXPECTED_ROWS - EXPECTED_TYPE_COUNTS["ORGANIZATION"]
    assert _uncovered_types(rows) == {"ORGANIZATION"}


def test_every_query_comes_from_the_evaluation_corpus():
    corpus = {
        (record.get("query") or "").strip() for record in json.loads(CORPUS.read_text())
    }
    assert [row["query"] for row in _rows() if row["query"] not in corpus] == list(
        CAPTION_QUERIES
    )
    assert [
        query
        for query, (relative, entry) in CAPTION_QUERIES.items()
        if query not in _caption_text(relative, entry)
    ] == []
