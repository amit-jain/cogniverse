"""analyze_query_performance must read the FLATTENED Phoenix frame.

Phoenix get_spans returns input/output/attributes as dotted columns
(``attributes.input.value``), not bare ``input``. The generator read
``row.get("input")``, so every row was skipped and the golden dataset came out
silently empty against real traces. These feed the real flattened shape.
"""

from __future__ import annotations

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
