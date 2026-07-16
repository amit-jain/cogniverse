"""query_annotated_spans must read annotations from the real provider seam.

It went through a nonexistent ``provider.evaluations.get_evaluations`` and raised
AttributeError on every project that had annotations — the orchestration
feedback loop crashed the moment any span was annotated. It now joins the
provider's get_annotations frame (indexed by span_id), like the routing sibling.
"""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotationStorage,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_storage(spans_df, annotations_df):
    storage = object.__new__(OrchestrationAnnotationStorage)
    storage.project_name = "acme"
    provider = MagicMock()
    provider.traces.get_spans = AsyncMock(return_value=spans_df)
    provider.annotations.get_annotations = AsyncMock(return_value=annotations_df)
    storage.provider = provider
    return storage


@pytest.mark.asyncio
async def test_query_annotated_spans_joins_annotations_without_crashing():
    from datetime import datetime, timezone

    spans_df = pd.DataFrame(
        [
            {"context.span_id": "span-1", "name": "orchestration"},
            {"context.span_id": "span-2", "name": "orchestration"},
        ]
    )
    annotations_df = pd.DataFrame(
        [
            {
                "result.label": "good",
                "result.score": 0.9,
                "annotator_kind": "human",
                "metadata": {"reviewer": "alice"},
            }
        ],
        index=["span-1"],  # only span-1 is annotated
    )
    storage = _bare_storage(spans_df, annotations_df)

    now = datetime.now(timezone.utc)
    result = await storage.query_annotated_spans(
        start_time=now, end_time=now, only_human_reviewed=True
    )

    # Only the annotated span comes back, with its annotation joined.
    assert len(result) == 1
    assert result[0]["span_id"] == "span-1"
    ann = result[0]["annotations"][0]
    assert ann["annotator_kind"] == "human"
    assert ann["result"]["label"] == "good"
    assert ann["result"]["score"] == 0.9


@pytest.mark.asyncio
async def test_query_annotated_spans_empty_when_no_annotations():
    spans_df = pd.DataFrame([{"context.span_id": "span-1", "name": "orchestration"}])
    storage = _bare_storage(spans_df, pd.DataFrame())

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    result = await storage.query_annotated_spans(start_time=now, end_time=now)
    assert result == []
