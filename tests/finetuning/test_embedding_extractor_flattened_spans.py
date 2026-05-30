"""TripletExtractor must read Phoenix's FLATTENED span columns.

Phoenix's get_spans dataframe has no bare ``attributes`` column — span
attributes live in dotted ``attributes.*`` columns (verified by a live-Phoenix
probe: a search span surfaces ``attributes.input.value`` /
``attributes.output.value`` leaf columns, never a single ``attributes`` dict).
The extractor previously did ``spans_df["attributes"]`` and
``span.get("attributes", {})``, so the whole embedding-finetuning path crashed
with KeyError on real data and extracted zero triplets. These feed the
real flattened column shape and assert a triplet is produced.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor


class _FakeTraceStore:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    async def get_spans(self, project: str, **kw) -> pd.DataFrame:
        return self._df


class _FakeAnnotationStore:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    async def get_annotations(
        self, spans_df, project, annotation_names
    ) -> pd.DataFrame:
        return self._df


class _FakeProvider:
    def __init__(self, spans: pd.DataFrame, annotations: pd.DataFrame):
        self._trace_store = _FakeTraceStore(spans)
        self._annotation_store = _FakeAnnotationStore(annotations)


def _flattened_search_span_df() -> pd.DataFrame:
    """One search span in Phoenix's real flattened column layout."""
    results = [
        {"document_id": "doc_pos", "title": "a clicked cat video", "score": 0.9},
        {"document_id": "doc_neg", "title": "an unrelated dog video", "score": 0.85},
    ]
    return pd.DataFrame(
        [
            {
                "name": "search_service.search",
                "context.span_id": "span_1",
                "attributes.input.value": "cats playing",
                "attributes.input.modality": "video",
                "attributes.output.value": json.dumps(results),
                "start_time": pd.Timestamp.utcnow(),
            }
        ]
    )


def _annotations_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "span_id": "span_1",
                "name": "result_click",
                "result.result_id": "doc_pos",
            }
        ]
    )


@pytest.mark.asyncio
async def test_extract_reads_flattened_columns_and_builds_triplet():
    extractor = TripletExtractor(
        provider=_FakeProvider(_flattened_search_span_df(), _annotations_df())
    )

    triplets = await extractor.extract(
        project="cogniverse-acme", modality="video", strategy="top_k", min_triplets=1
    )

    # doc_pos was clicked (positive), doc_neg is the hard negative.
    assert len(triplets) == 1
    t = triplets[0]
    assert t.anchor == "cats playing"
    assert t.positive == "a clicked cat video"
    assert t.negative == "an unrelated dog video"
    assert t.modality == "video"
    assert t.metadata["span_id"] == "span_1"


@pytest.mark.asyncio
async def test_filter_search_spans_does_not_crash_on_flattened_df():
    extractor = TripletExtractor(
        provider=_FakeProvider(_flattened_search_span_df(), _annotations_df())
    )
    # The pre-fix code raised KeyError: 'attributes' here.
    filtered = extractor._filter_search_spans(_flattened_search_span_df(), "video")
    assert len(filtered) == 1
    # A non-matching modality is filtered out.
    assert extractor._filter_search_spans(_flattened_search_span_df(), "image").empty
