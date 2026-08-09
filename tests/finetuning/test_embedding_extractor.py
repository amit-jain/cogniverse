import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor


def _span(*, results, span_id=101):
    return pd.DataFrame(
        [
            {
                "name": "video_search",
                "context.span_id": span_id,
                "attributes.modality": "video",
                "attributes.input.value": "find the launch keynote",
                "attributes.output.value": json.dumps(results),
            }
        ]
    )


def _annotations(*, span_id="101", result_id="7"):
    return pd.DataFrame(
        [
            {
                "span_id": span_id,
                "annotation_name": "result_relevance",
                "result.score": 1.0,
                "metadata": {"result_id": result_id},
            }
        ]
    )


def _provider(spans, annotations):
    return SimpleNamespace(
        traces=SimpleNamespace(
            get_spans=AsyncMock(return_value=spans),
            get_all_spans=AsyncMock(return_value=spans),
        ),
        annotations=SimpleNamespace(
            get_annotations=AsyncMock(return_value=annotations)
        ),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_uses_public_stores_and_canonicalizes_result_ids():
    provider = _provider(
        _span(
            results=[
                {"document_id": 7, "content": "the exact launch keynote"},
                {"document_id": 8, "content": "an unrelated cooking lesson"},
            ]
        ),
        _annotations(result_id="7"),
    )

    triplets = await TripletExtractor(provider).extract(
        project="cogniverse-tenant-video",
        modality="video",
        min_triplets=1,
    )

    assert len(triplets) == 1
    assert triplets[0].anchor == "find the launch keynote"
    assert triplets[0].positive == "the exact launch keynote"
    assert triplets[0].negative == "an unrelated cooking lesson"
    provider.traces.get_all_spans.assert_awaited_once_with(
        project="cogniverse-tenant-video"
    )
    provider.traces.get_spans.assert_not_awaited()
    provider.annotations.get_annotations.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_oldest_qualifying_search_beyond_default_page_yields_exact_triplet():
    newest_page = pd.DataFrame(
        [
            {
                "name": "unrelated_operation",
                "context.span_id": f"noise-{index}",
                "attributes.modality": "text",
                "attributes.input.value": f"noise query {index}",
                "attributes.output.value": "[]",
            }
            for index in range(1_000)
        ]
    )
    oldest_search = _span(
        span_id="oldest-search-span",
        results=[
            {"document_id": "launch", "content": "the first launch keynote"},
            {"document_id": "recipe", "content": "a sourdough starter guide"},
        ],
    )
    provider = _provider(
        pd.concat([newest_page, oldest_search], ignore_index=True),
        _annotations(span_id="oldest-search-span", result_id="launch"),
    )
    provider.traces.get_spans.return_value = newest_page

    triplets = await TripletExtractor(provider).extract(
        project="cogniverse-tenant-history",
        modality="video",
        min_triplets=1,
    )

    assert [
        (
            triplet.anchor,
            triplet.positive,
            triplet.negative,
            triplet.modality,
            triplet.metadata["span_id"],
        )
        for triplet in triplets
    ] == [
        (
            "find the launch keynote",
            "the first launch keynote",
            "a sourdough starter guide",
            "video",
            "oldest-search-span",
        )
    ]
    provider.traces.get_all_spans.assert_awaited_once_with(
        project="cogniverse-tenant-history"
    )
    provider.traces.get_spans.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_projects_keep_triplets_isolated():
    entered = 0
    both_entered = asyncio.Event()

    async def get_all_spans(*, project):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        tenant = project.rsplit("-", maxsplit=1)[-1]
        return _span(
            span_id=f"{tenant}-span",
            results=[
                {
                    "document_id": f"{tenant}-positive",
                    "content": f"{tenant} exact positive",
                },
                {
                    "document_id": f"{tenant}-negative",
                    "content": f"{tenant} exact negative",
                },
            ],
        )

    async def get_annotations(*, spans_df, project, annotation_names):
        assert annotation_names == ["result_click", "result_relevance"]
        tenant = project.rsplit("-", maxsplit=1)[-1]
        assert spans_df["context.span_id"].tolist() == [f"{tenant}-span"]
        return _annotations(
            span_id=f"{tenant}-span",
            result_id=f"{tenant}-positive",
        )

    provider = SimpleNamespace(
        traces=SimpleNamespace(get_all_spans=AsyncMock(side_effect=get_all_spans)),
        annotations=SimpleNamespace(
            get_annotations=AsyncMock(side_effect=get_annotations)
        ),
    )

    alpha, beta = await asyncio.gather(
        TripletExtractor(provider).extract(
            project="cogniverse-alpha",
            modality="video",
            min_triplets=1,
        ),
        TripletExtractor(provider).extract(
            project="cogniverse-beta",
            modality="video",
            min_triplets=1,
        ),
    )

    assert [
        (triplet.positive, triplet.negative, triplet.metadata["span_id"])
        for triplet in alpha
    ] == [("alpha exact positive", "alpha exact negative", "alpha-span")]
    assert [
        (triplet.positive, triplet.negative, triplet.metadata["span_id"])
        for triplet in beta
    ] == [("beta exact positive", "beta exact negative", "beta-span")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_clicked_result_raises_with_span_context():
    provider = _provider(
        _span(
            results=[
                {"document_id": "7", "score": 0.95},
                {"document_id": "8", "content": "an unrelated cooking lesson"},
            ],
            span_id="span-bad",
        ),
        _annotations(span_id="span-bad", result_id="7"),
    )

    with pytest.raises(
        ValueError,
        match="Search span span-bad result 7 has no non-empty textual content",
    ):
        await TripletExtractor(provider).extract(
            project="cogniverse-tenant-video",
            modality="video",
            min_triplets=1,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trace_boundary_failure_raises_with_project_context():
    provider = _provider(pd.DataFrame(), pd.DataFrame())
    provider.traces.get_all_spans.side_effect = ConnectionError("Phoenix offline")

    with pytest.raises(
        RuntimeError,
        match=("Failed to query search spans from project cogniverse-tenant-video"),
    ) as error:
        await TripletExtractor(provider).extract(
            project="cogniverse-tenant-video",
            modality="video",
            min_triplets=1,
        )

    assert isinstance(error.value.__cause__, ConnectionError)
    assert str(error.value.__cause__) == "Phoenix offline"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_annotation_boundary_failure_raises_with_project_context():
    provider = _provider(
        _span(
            results=[
                {"document_id": "7", "content": "the exact launch keynote"},
                {"document_id": "8", "content": "an unrelated cooking lesson"},
            ]
        ),
        pd.DataFrame(),
    )
    provider.annotations.get_annotations.side_effect = TimeoutError(
        "Phoenix annotations timed out"
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to query result annotations from project cogniverse-tenant-video"
        ),
    ) as error:
        await TripletExtractor(provider).extract(
            project="cogniverse-tenant-video",
            modality="video",
            min_triplets=1,
        )

    assert isinstance(error.value.__cause__, TimeoutError)
    assert str(error.value.__cause__) == "Phoenix annotations timed out"
