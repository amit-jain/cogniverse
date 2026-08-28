"""Unit tests for source-granularity collapse helper behavior."""

import json
from pathlib import Path

from cogniverse_sdk.document import (
    ContentType,
    Document,
    ProcessingStatus,
    SearchResult,
)
from cogniverse_vespa.search_backend import (
    _collapse_results_by_source,
    _schema_temporal_field_names,
)


def _make_result(doc_id: str, source_id: str, score: float) -> SearchResult:
    doc = Document(
        id=doc_id,
        content_type=ContentType.VIDEO,
        status=ProcessingStatus.COMPLETED,
        metadata={"source_id": source_id},
    )
    return SearchResult(document=doc, score=score)


def test_schema_temporal_field_names_are_derived_per_schema():
    schema_dir = Path(__file__).resolve().parents[3] / "configs" / "schemas"
    video_schema = json.loads(
        (schema_dir / "video_colpali_smol500_mv_frame_schema.json").read_text()
    )
    audio_schema = json.loads((schema_dir / "audio_content_schema.json").read_text())

    assert _schema_temporal_field_names(video_schema) == (
        "start_time",
        "end_time",
    )
    assert _schema_temporal_field_names(audio_schema) == ()


def test_collapse_keeps_best_document_per_source():
    results = [
        _make_result("doc-a-0", "source-a", 0.99),
        _make_result("doc-a-1", "source-a", 0.95),
        _make_result("doc-b-0", "source-b", 0.90),
        _make_result("doc-c-0", "source-c", 0.85),
    ]

    collapsed = _collapse_results_by_source(
        results,
        top_k=3,
        fetch_limit=4,
        total_count=4,
    )

    assert [result.document.id for result in collapsed] == [
        "doc-a-0",
        "doc-b-0",
        "doc-c-0",
    ]
    assert [result.document.metadata["source_id"] for result in collapsed] == [
        "source-a",
        "source-b",
        "source-c",
    ]
    assert collapsed[0].matched_segments == [
        {"document_id": "doc-a-0", "score": 0.99},
        {"document_id": "doc-a-1", "score": 0.95},
    ]
    assert collapsed[0].segments_in_window == 2
    assert collapsed[1].matched_segments == [{"document_id": "doc-b-0", "score": 0.90}]
    assert collapsed[1].segments_in_window == 1
    assert collapsed[2].matched_segments == [{"document_id": "doc-c-0", "score": 0.85}]
    assert collapsed[2].segments_in_window == 1
    assert collapsed.result_granularity == "source"
    assert collapsed.num_collapsed_documents == 1
    assert collapsed.total_count == 4


def test_collapse_returns_available_sources_when_window_is_insufficient():
    results = [
        _make_result("doc-a-0", "source-a", 0.99),
        _make_result("doc-a-1", "source-a", 0.95),
        _make_result("doc-a-2", "source-a", 0.90),
        _make_result("doc-a-3", "source-a", 0.85),
    ]

    collapsed = _collapse_results_by_source(
        results,
        top_k=3,
        fetch_limit=4,
        total_count=10,
    )

    assert [result.document.id for result in collapsed] == ["doc-a-0"]
    assert collapsed[0].matched_segments == [
        {"document_id": "doc-a-0", "score": 0.99},
        {"document_id": "doc-a-1", "score": 0.95},
        {"document_id": "doc-a-2", "score": 0.90},
        {"document_id": "doc-a-3", "score": 0.85},
    ]
    assert collapsed[0].segments_in_window == 4
    assert collapsed.result_granularity == "source"
    assert collapsed.num_collapsed_documents == 9
    assert collapsed.total_count == 10
