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
                # Phoenix stamps annotator_kind="HUMAN" (uppercase) on every
                # add_annotation write — the real row shape, not "human".
                "annotator_kind": "HUMAN",
                "metadata": {"annotation_source": "human", "reviewer": "alice"},
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
    assert ann["annotator_kind"] == "HUMAN"
    assert ann["result"]["label"] == "good"
    assert ann["result"]["score"] == 0.9


@pytest.mark.asyncio
async def test_query_annotated_spans_pushes_orchestration_name_filter_server_side():
    """The span pull carries a server-side ``name`` predicate for the
    orchestration span so Phoenix returns just those spans, not the whole
    project window. The joined rows are unchanged versus the unfiltered pull."""
    from datetime import datetime, timezone

    from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION

    spans_df = pd.DataFrame(
        [
            {"context.span_id": "span-1", "name": SPAN_NAME_ORCHESTRATION},
        ]
    )
    annotations_df = pd.DataFrame(
        [
            {
                "result.label": "good",
                "result.score": 0.9,
                "annotator_kind": "HUMAN",
                "metadata": {"annotation_source": "human", "reviewer": "alice"},
            }
        ],
        index=["span-1"],
    )
    storage = _bare_storage(spans_df, annotations_df)

    now = datetime.now(timezone.utc)
    result = await storage.query_annotated_spans(
        start_time=now, end_time=now, only_human_reviewed=True
    )

    storage.provider.traces.get_spans.assert_awaited_once()
    _, kwargs = storage.provider.traces.get_spans.await_args
    assert kwargs["filters"] == {"name": SPAN_NAME_ORCHESTRATION}
    assert kwargs["project"] == "acme"
    # Same joined row as the unfiltered pull would have produced.
    assert len(result) == 1
    assert result[0]["span_id"] == "span-1"
    assert result[0]["annotations"][0]["result"]["score"] == 0.9


@pytest.mark.asyncio
async def test_query_annotated_spans_empty_when_no_annotations():
    spans_df = pd.DataFrame([{"context.span_id": "span-1", "name": "orchestration"}])
    storage = _bare_storage(spans_df, pd.DataFrame())

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    result = await storage.query_annotated_spans(start_time=now, end_time=now)
    assert result == []


@pytest.mark.asyncio
async def test_only_human_reviewed_excludes_llm_auto_by_annotation_source():
    """The human-vs-auto split rides annotation_source in metadata, NOT
    annotator_kind (which add_annotation hardcodes "HUMAN" for both). A human
    annotation must return under only_human_reviewed=True; an llm_auto one must
    not — and both return when the flag is off. Under the old
    ``annotator_kind == "human"`` filter the real-shape human row (uppercase
    "HUMAN") matched nothing, so the read silently returned []."""
    from datetime import datetime, timezone

    spans_df = pd.DataFrame(
        [
            {"context.span_id": "span-h", "name": "orchestration"},
            {"context.span_id": "span-a", "name": "orchestration"},
        ]
    )
    annotations_df = pd.DataFrame(
        [
            {
                "result.label": "good",
                "result.score": 0.9,
                "annotator_kind": "HUMAN",
                "metadata": {"annotation_source": "human"},
            },
            {
                "result.label": "ok",
                "result.score": 0.7,
                # add_annotation stamps "HUMAN" even for an auto annotation —
                # only annotation_source separates them.
                "annotator_kind": "HUMAN",
                "metadata": {"annotation_source": "llm_auto"},
            },
        ],
        index=["span-h", "span-a"],
    )
    storage = _bare_storage(spans_df, annotations_df)
    now = datetime.now(timezone.utc)

    human_only = await storage.query_annotated_spans(
        start_time=now, end_time=now, only_human_reviewed=True
    )
    assert [r["span_id"] for r in human_only] == ["span-h"]

    everything = await storage.query_annotated_spans(
        start_time=now, end_time=now, only_human_reviewed=False
    )
    assert {r["span_id"] for r in everything} == {"span-h", "span-a"}


def test_ctor_canonicalizes_tenant_and_resolves_canonical_project():
    """The runtime persists orchestration annotations under the canonical
    tenant provider AND emits spans to the canonical per-tenant project; a
    storage built with a raw id (e.g. a dashboard tab's current_tenant) must
    resolve the SAME provider scope and query the canonical project, never a
    literal "cogniverse"."""
    from unittest.mock import patch

    provider = MagicMock()
    mgr = MagicMock()
    mgr.get_provider.return_value = provider
    mgr.config.get_project_name.return_value = "cogniverse-acme:acme"

    with patch(
        "cogniverse_agents.routing.orchestration_annotation_storage.get_telemetry_manager",
        return_value=mgr,
    ):
        from cogniverse_agents.routing.orchestration_annotation_storage import (
            OrchestrationAnnotationStorage,
        )

        storage = OrchestrationAnnotationStorage(tenant_id="acme")

    assert storage.tenant_id == "acme:acme"
    mgr.get_provider.assert_called_once_with(tenant_id="acme:acme")
    assert storage.provider is provider
    mgr.config.get_project_name.assert_called_once_with("acme:acme")
    assert storage.project_name == "cogniverse-acme:acme"
