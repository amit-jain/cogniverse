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
