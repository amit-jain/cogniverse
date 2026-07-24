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

pytestmark = pytest.mark.unit


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
        )
        annotations = PhoenixAnnotationStore(http_endpoint="http://unused:1")

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
    async def test_name_list_filter_builds_in_predicate(self, monkeypatch):
        # A list of names must build ``name in ['a', 'b']`` so a caller that
        # reconstructs an object from more than one span type (approval batch
        # + item children) still receives every span type.
        import pandas as pd

        from cogniverse_telemetry_phoenix import provider as provider_mod
        from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

        store = PhoenixTraceStore(
            http_endpoint="http://unused:1",
        )
        client = MagicMock()
        client.spans.get_spans_dataframe = AsyncMock(return_value=pd.DataFrame())
        monkeypatch.setattr(
            provider_mod, "_client_for_current_loop", lambda endpoint: client
        )

        await store.get_spans(
            project="proj",
            filters={"name": ["approval_batch", "approval_item"]},
        )

        condition = client.spans.get_spans_dataframe.await_args.kwargs[
            "query"
        ].to_dict()["filter"]["condition"]
        assert condition == "name in ['approval_batch', 'approval_item']"

    @pytest.mark.asyncio
    async def test_name_list_filter_escapes_each_element(self, monkeypatch):
        import pandas as pd

        from cogniverse_telemetry_phoenix import provider as provider_mod
        from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

        store = PhoenixTraceStore(
            http_endpoint="http://unused:1",
        )
        client = MagicMock()
        client.spans.get_spans_dataframe = AsyncMock(return_value=pd.DataFrame())
        monkeypatch.setattr(
            provider_mod, "_client_for_current_loop", lambda endpoint: client
        )

        await store.get_spans(project="proj", filters={"name": ["a'b", "c\\"]})

        condition = client.spans.get_spans_dataframe.await_args.kwargs[
            "query"
        ].to_dict()["filter"]["condition"]
        # Each element escaped independently, backslash-then-quote.
        assert condition == "name in ['a\\'b', 'c\\\\']"

    @pytest.mark.asyncio
    async def test_no_filters_sends_no_query(self, monkeypatch):
        import pandas as pd

        from cogniverse_telemetry_phoenix import provider as provider_mod
        from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

        store = PhoenixTraceStore(
            http_endpoint="http://unused:1",
        )
        client = MagicMock()
        client.spans.get_spans_dataframe = AsyncMock(return_value=pd.DataFrame())
        monkeypatch.setattr(
            provider_mod, "_client_for_current_loop", lambda endpoint: client
        )

        await store.get_spans(project="proj")

        assert client.spans.get_spans_dataframe.await_args.kwargs["query"] is None


@pytest.mark.unit
def test_initialize_raises_when_telemetry_registry_fails():
    """An init-time telemetry failure must surface, not leave a provider
    that looks constructed but has telemetry=None (every later call would
    AttributeError far from the root cause)."""
    from unittest.mock import patch

    provider = PhoenixEvaluationProvider()
    with patch(
        "cogniverse_foundation.telemetry.registry.get_telemetry_registry",
        side_effect=ConnectionError("telemetry registry down"),
    ):
        with pytest.raises(ConnectionError, match="telemetry registry down"):
            provider.initialize({"tenant_id": "acme:init"})

    assert provider._initialized is False


@pytest.mark.unit
def test_sync_log_session_evaluation_propagates_annotation_failure():
    """On the sync path (no running loop) an annotation-store failure must
    raise — the dashboard catches it and shows the failure instead of
    reporting 'Evaluation saved'."""
    provider = PhoenixEvaluationProvider()
    provider._initialized = True
    provider._project_name = "cogniverse-search"
    annotations = MagicMock()
    annotations.add_annotation = AsyncMock(
        side_effect=ConnectionError("annotation store down")
    )
    provider._telemetry_provider = MagicMock(annotations=annotations)

    with pytest.raises(ConnectionError, match="annotation store down"):
        provider.log_session_evaluation(
            session_id="sess-1",
            evaluation_name="dashboard_annotation",
            session_score=0.5,
            session_outcome="success",
        )


@pytest.mark.asyncio
async def test_async_log_session_evaluation_logs_background_failure(caplog):
    """On the async path the write is fire-and-forget; a failure must be
    logged at ERROR (not silently discarded with the task reference)."""
    import asyncio
    import logging

    provider = PhoenixEvaluationProvider()
    provider._initialized = True
    provider._project_name = "cogniverse-search"
    annotations = MagicMock()
    annotations.add_annotation = AsyncMock(
        side_effect=ConnectionError("annotation store down")
    )
    provider._telemetry_provider = MagicMock(annotations=annotations)

    with caplog.at_level(logging.ERROR):
        provider.log_session_evaluation(
            session_id="sess-2",
            evaluation_name="dashboard_annotation",
            session_score=0.5,
            session_outcome="success",
        )
        # Let the background task run to completion
        for _ in range(10):
            await asyncio.sleep(0)

    assert any(
        "session evaluation" in rec.message and "sess-2" in rec.message
        for rec in caplog.records
    ), f"background failure never logged: {[r.message for r in caplog.records]}"


class _FakeTelemetry:
    def __init__(self, datasets):
        self.datasets = datasets


class TestExperimentSurface:
    """create_experiment / log_evaluation persist durable records — the prior
    bodies returned a fabricated dict and logged a debug line, so nothing an
    operator could ever read back existed."""

    def _provider(self):
        from tests.evaluation.fakes import InMemoryDatasetStore

        provider = PhoenixEvaluationProvider()
        provider._initialized = True
        provider._project_name = "cogniverse-exp"
        store = InMemoryDatasetStore()
        provider._telemetry_provider = _FakeTelemetry(store)
        return provider, store

    @pytest.mark.unit
    def test_create_experiment_persists_registry_record(self):
        provider, store = self._provider()

        result = provider.create_experiment(
            "exp1", description="first experiment", metadata={"k": "v"}
        )

        assert result["id"] == "experiment-exp1"
        assert result["name"] == "exp1"
        df = store._frames["experiment-exp1"]
        assert len(df) == 1
        row = df.iloc[0]
        assert row["event"] == "experiment_created"
        assert row["experiment"] == "exp1"
        assert row["description"] == "first experiment"

    @pytest.mark.unit
    def test_log_evaluation_appends_row(self):
        provider, store = self._provider()
        result = provider.create_experiment("exp2")

        provider.log_evaluation(
            experiment_id=result["id"],
            evaluation_name="relevance",
            score=0.85,
            label="good",
            explanation="matched",
        )

        df = store._frames["experiment-exp2"]
        assert len(df) == 2
        row = df.iloc[-1]
        assert row["event"] == "evaluation"
        assert row["evaluation_name"] == "relevance"
        assert row["score"] == 0.85
        assert row["label"] == "good"
        assert row["explanation"] == "matched"

    @pytest.mark.unit
    def test_log_evaluation_unknown_experiment_raises(self):
        provider, _ = self._provider()

        with pytest.raises(ValueError):
            provider.log_evaluation(
                experiment_id="experiment-ghost",
                evaluation_name="relevance",
                score=0.5,
            )

    @pytest.mark.unit
    def test_create_experiment_uninitialized_raises(self):
        provider = PhoenixEvaluationProvider()

        with pytest.raises(RuntimeError, match="not initialized"):
            provider.create_experiment("nope")


@pytest.mark.unit
def test_log_session_evaluation_uninitialized_raises():
    """Consistent with the sibling experiment methods: an uninitialized
    provider raises instead of silently dropping the evaluation."""
    provider = PhoenixEvaluationProvider()

    with pytest.raises(RuntimeError, match="not initialized"):
        provider.log_session_evaluation(
            session_id="s",
            evaluation_name="e",
            session_score=0.5,
            session_outcome="success",
        )


class TestLogExperimentEvent:
    """log_experiment_event records the experiment lifecycle event as an
    OpenTelemetry span named after the event, carrying the event data as span
    attributes. The prior code built a throwaway monitor and emitted a span
    named "retrieval" with retrieval-shaped (all-zero) attributes, dropping the
    event identity entirely."""

    @pytest.fixture
    def captured_spans(self, monkeypatch):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # The provider calls trace.get_tracer(__name__), which reads the global
        # provider; redirect it to a real in-memory-backed provider. The span
        # still flows through the real OTel SDK — only the provider is swapped.
        monkeypatch.setattr(
            trace, "get_tracer", lambda *_a, **_kw: provider.get_tracer("test")
        )
        return exporter

    @pytest.mark.unit
    def test_emits_span_named_after_event_with_data_attributes(self, captured_spans):
        provider = PhoenixEvaluationProvider()
        provider._initialized = True

        provider.log_experiment_event(
            event_type="experiment_start",
            data={
                "profile": "frame_based_colpali",
                "strategy": "binary_binary",
                "description": "golden run",
                "dataset": "golden_eval_v1",
            },
        )

        spans = captured_spans.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "experiment_start"
        assert span.attributes["profile"] == "frame_based_colpali"
        assert span.attributes["strategy"] == "binary_binary"
        assert span.attributes["description"] == "golden run"
        assert span.attributes["dataset"] == "golden_eval_v1"

    @pytest.mark.unit
    def test_complete_event_carries_numeric_and_bool_attributes(self, captured_spans):
        provider = PhoenixEvaluationProvider()
        provider._initialized = True

        provider.log_experiment_event(
            event_type="experiment_complete",
            data={"profile": "p", "strategy": "s", "mrr": 0.42, "error": False},
        )

        spans = captured_spans.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "experiment_complete"
        assert span.attributes["mrr"] == 0.42
        assert span.attributes["error"] is False

    @pytest.mark.unit
    def test_non_primitive_values_coerced_and_none_skipped(self, captured_spans):
        # A nested list/dict value would make span.set_attribute raise; it must
        # be stringified. A None value must be dropped rather than passed to
        # set_attribute (which rejects None).
        provider = PhoenixEvaluationProvider()
        provider._initialized = True

        provider.log_experiment_event(
            event_type="experiment_start",
            data={"profile": "p", "tags": ["a", "b"], "notes": None},
        )

        span = captured_spans.get_finished_spans()[0]
        assert span.name == "experiment_start"
        assert span.attributes["profile"] == "p"
        assert span.attributes["tags"] == "['a', 'b']"
        assert "notes" not in span.attributes

    @pytest.mark.unit
    def test_telemetry_unconfigured_is_noop_not_error(self, monkeypatch):
        # Fault contract: with no exporting tracer configured, get_tracer
        # returns a no-op tracer; the call must not raise. Telemetry being down
        # never breaks the experiment run.
        from opentelemetry import trace

        monkeypatch.setattr(trace, "get_tracer", lambda *_a, **_kw: trace.NoOpTracer())
        provider = PhoenixEvaluationProvider()
        provider._initialized = True

        provider.log_experiment_event(
            event_type="experiment_start", data={"profile": "p"}
        )

    @pytest.mark.unit
    def test_uninitialized_provider_emits_no_span(self, captured_spans):
        provider = PhoenixEvaluationProvider()

        provider.log_experiment_event(
            event_type="experiment_start", data={"profile": "p"}
        )

        assert captured_spans.get_finished_spans() == ()
