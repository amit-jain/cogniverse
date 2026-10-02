"""analyze_query_performance must read the FLATTENED Phoenix frame.

Phoenix get_spans returns input/output/attributes as dotted columns
(``attributes.input.value``), not bare ``input``. The generator read
``row.get("input")``, so every row was skipped and the golden dataset came out
silently empty against real traces. These feed the real flattened shape.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from create_golden_dataset_from_traces import (  # noqa: E402
    GoldenDatasetGenerator,
    _span_attributes,
    _span_output,
    _span_query,
)

pytestmark = pytest.mark.unit


def _generator(min_occurrences=1):
    gen = object.__new__(GoldenDatasetGenerator)
    gen.min_occurrences = min_occurrences
    gen.score_threshold = 1.0
    return gen


def _flattened_row(query, videos, profile, strategy, score):
    # Canonical span contract: input.value is the clean query, output.value is a
    # bare list of result rows.
    return {
        "attributes.input.value": query,
        "attributes.output.value": json.dumps([{"video_id": v} for v in videos]),
        "attributes.profile": profile,
        "attributes.ranking_strategy": strategy,
        "score": score,
        "start_time": pd.Timestamp("2026-01-01T00:00:00Z"),
    }


def test_analyze_extracts_from_flattened_columns():
    df = pd.DataFrame(
        [
            _flattened_row(
                "man lifting barbell", ["v_a", "v_b"], "colpali", "float_float", 0.3
            ),
        ]
    )
    stats = _generator().analyze_query_performance(df)

    assert "man lifting barbell" in stats, "dataset must not be empty on real traces"
    entry = stats["man lifting barbell"]
    assert entry["occurrences"] == 1
    assert entry["avg_score"] == pytest.approx(0.3)
    assert "colpali" in entry["profiles_tested"]
    assert "float_float" in entry["strategies_tested"]


def test_bare_input_column_still_skips_only_when_truly_absent():
    # A row with no input/query columns at all yields no query -> skipped, but
    # must not raise.
    df = pd.DataFrame(
        [{"score": 0.2, "start_time": pd.Timestamp("2026-01-01T00:00:00Z")}]
    )
    assert _generator().analyze_query_performance(df) == {}


def test_span_query_reads_clean_input_value_and_nested_input():
    # Canonical: input.value is the clean query text.
    assert (
        _span_query(pd.Series({"attributes.input.value": "raw query"})) == "raw query"
    )
    # A JSON-dict string is query TEXT under the clean contract — no unwrapping.
    assert (
        _span_query(pd.Series({"attributes.input.value": '{"query": "q1"}'}))
        == '{"query": "q1"}'
    )
    # A nested input dict still resolves via the input.query fallback.
    assert _span_query(pd.Series({"input": {"query": "q2"}})) == "q2"
    assert _span_query(pd.Series({"other": 1})) == ""


def test_span_output_parses_json_and_dict():
    assert _span_output(pd.Series({"attributes.output.value": '{"results": [1]}'})) == {
        "results": [1]
    }
    assert _span_output(pd.Series({"output": {"results": []}})) == {"results": []}
    assert _span_output(pd.Series({"x": 1})) == {}


def test_span_attributes_reconstructs_from_dotted_columns():
    row = pd.Series(
        {"attributes.profile": "p", "attributes.ranking_strategy": "s", "name": "x"}
    )
    attrs = _span_attributes(row)
    assert attrs == {"profile": "p", "ranking_strategy": "s"}


def _span_rows(titles):
    """Canonical ``output.value`` rows for content-hash sources, built by the
    production row writer from backend SearchResults."""
    from cogniverse_foundation.telemetry.span_contract import search_result_row
    from cogniverse_sdk.document import ContentType, Document, SearchResult

    rows = []
    for index, title in enumerate(titles):
        source_id = hashlib.sha256(str(title).encode()).hexdigest()
        document = Document(id=f"{source_id}_seg_0", content_type=ContentType.VIDEO)
        document.add_metadata("source_id", source_id)
        if title is not None:
            document.add_metadata("source_title", title)
        rows.append(search_result_row(SearchResult(document, 1.0 - index / 10)))
    return rows


def _trace(query, rows, score):
    return {
        "attributes.input.value": query,
        "attributes.output.value": json.dumps(rows),
        "score": score,
        "start_time": pd.Timestamp("2026-01-01T00:00:00Z"),
    }


def test_expected_videos_are_title_stems_not_hash_ids():
    rows = _span_rows(["v_-uJnucdW6DY.mp4", "v_-HpCLXdtcas.mkv"])
    df = pd.DataFrame(
        [
            _trace("man lifting barbell", rows, 0.3),
            _trace("man lifting barbell", rows, 0.4),
        ]
    )

    stats = _generator().analyze_query_performance(df)

    assert stats["man lifting barbell"]["expected_videos"] == [
        "v_-uJnucdW6DY",
        "v_-HpCLXdtcas",
    ]


def test_untitled_rows_are_left_out_and_reported(caplog):
    rows = _span_rows(["v_-uJnucdW6DY.mp4", None])
    df = pd.DataFrame([_trace("man lifting barbell", rows, 0.3)])

    with caplog.at_level("WARNING"):
        stats = _generator().analyze_query_performance(df)

    assert stats["man lifting barbell"]["expected_videos"] == ["v_-uJnucdW6DY"]
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "create_golden_dataset_from_traces"
    ] == ["1 result rows carry no source_title and are left out of expected_videos"]
