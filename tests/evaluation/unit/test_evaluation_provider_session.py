"""Unit test for PhoenixEvaluationProvider.log_session_evaluation wiring.

The annotation store's ``add_annotation`` requires a ``project`` argument.
The prior code omitted it, so the call raised TypeError inside a fire-and-forget
task whose exception was swallowed — the dashboard reported "Evaluation saved"
while nothing persisted. This pins that ``project`` (resolved from the
provider's configured project name) is passed through.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import (
    PhoenixEvaluationProvider,
)


@pytest.mark.unit
def test_log_session_evaluation_passes_project_to_annotation_store():
    provider = PhoenixEvaluationProvider()
    provider._initialized = True
    provider._project_name = "cogniverse-search"
    annotations = MagicMock()
    annotations.add_annotation = AsyncMock(return_value="ann-1")
    provider._telemetry_provider = MagicMock(annotations=annotations)

    # Called from a sync context (no running loop), so log_session_evaluation
    # awaits the annotation write before returning.
    provider.log_session_evaluation(
        session_id="span-123",
        evaluation_name="dashboard_annotation",
        session_score=0.8,
        session_outcome="good",
    )

    annotations.add_annotation.assert_awaited_once()
    kwargs = annotations.add_annotation.await_args.kwargs
    assert kwargs["project"] == "cogniverse-search"
    assert kwargs["span_id"] == "span-123"
    assert kwargs["score"] == 0.8
    assert kwargs["label"] == "good"


class TestPerLoopClientMemoization:
    """Phoenix store clients must be reused within one event loop (the
    runtime / quality monitor keep a long-lived loop, so per-call clients
    threw away the TCP pool every request) while fresh loops — Streamlit's
    asyncio.run per interaction — still get their own client."""

    def test_same_loop_reuses_client_fresh_loop_does_not(self):
        import asyncio

        from cogniverse_telemetry_phoenix.provider import (
            PhoenixAnnotationStore,
            PhoenixTraceStore,
        )

        traces = PhoenixTraceStore(
            http_endpoint="http://unused:1",
            tenant_id="t",
            project_template="p-{tenant_id}",
        )
        annotations = PhoenixAnnotationStore(
            http_endpoint="http://unused:1", tenant_id="t"
        )

        async def within_one_loop():
            a = traces._get_client()
            b = traces._get_client()
            c = annotations._get_client()
            return a, b, c

        a, b, c = asyncio.run(within_one_loop())
        assert a is b, "same loop + endpoint must reuse one client"
        assert a is c, (
            "stores sharing an endpoint on the same loop must share the client"
        )

        async def second_loop():
            return traces._get_client()

        d = asyncio.run(second_loop())
        assert d is not a, "a fresh event loop must get its own client"


class TestGetSpansServerSideNameFilter:
    """A ``{"name": ...}`` filter must become a SpanQuery predicate sent to
    Phoenix — client-side name filtering pulled the project's whole span
    frame per call and burned the limit on unrelated span types."""

    @pytest.mark.asyncio
    async def test_name_filter_builds_spanquery_predicate(self, monkeypatch):
        import pandas as pd

        from cogniverse_telemetry_phoenix import provider as provider_mod
        from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

        store = PhoenixTraceStore(
            http_endpoint="http://unused:1",
            tenant_id="t",
            project_template="p-{tenant_id}",
        )
        client = MagicMock()
        client.spans.get_spans_dataframe = AsyncMock(return_value=pd.DataFrame())
        monkeypatch.setattr(
            provider_mod, "_client_for_current_loop", lambda endpoint: client
        )

        await store.get_spans(project="proj", filters={"name": "workflow_checkpoint"})

        kwargs = client.spans.get_spans_dataframe.await_args.kwargs
        query = kwargs["query"]
        assert query is not None, "name filter must produce a SpanQuery"
        # The predicate rides inside the serialized query payload.
        assert "workflow_checkpoint" in str(query.to_dict())

    @pytest.mark.asyncio
    async def test_name_filter_escapes_quotes_and_backslashes(self, monkeypatch):
        import pandas as pd

        from cogniverse_telemetry_phoenix import provider as provider_mod
        from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

        store = PhoenixTraceStore(
            http_endpoint="http://unused:1",
            tenant_id="t",
            project_template="p-{tenant_id}",
        )
        client = MagicMock()
        client.spans.get_spans_dataframe = AsyncMock(return_value=pd.DataFrame())
        monkeypatch.setattr(
            provider_mod, "_client_for_current_loop", lambda endpoint: client
        )

        raw_name = "a'b\\"
        await store.get_spans(project="proj", filters={"name": raw_name})

        query_dict = client.spans.get_spans_dataframe.await_args.kwargs[
            "query"
        ].to_dict()
        condition = query_dict["filter"]["condition"]
        # Backslash escaped first, then the quote — quoting first would let
        # a trailing backslash re-escape the closing quote.
        escaped = raw_name.replace("\\", "\\\\").replace("'", "\\'")
        assert condition == f"name == '{escaped}'"

    @pytest.mark.asyncio
    async def test_no_filters_sends_no_query(self, monkeypatch):
        import pandas as pd

        from cogniverse_telemetry_phoenix import provider as provider_mod
        from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

        store = PhoenixTraceStore(
            http_endpoint="http://unused:1",
            tenant_id="t",
            project_template="p-{tenant_id}",
        )
        client = MagicMock()
        client.spans.get_spans_dataframe = AsyncMock(return_value=pd.DataFrame())
        monkeypatch.setattr(
            provider_mod, "_client_for_current_loop", lambda endpoint: client
        )

        await store.get_spans(project="proj")

        assert client.spans.get_spans_dataframe.await_args.kwargs["query"] is None
