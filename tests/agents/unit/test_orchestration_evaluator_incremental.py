"""OrchestrationEvaluator must evaluate spans incrementally.

The first run scans the lookback_hours window; each subsequent run resumes
from where the previous run finished (``_last_evaluation_time``) so spans are
not re-scanned every call. A failed query must NOT advance the resume point.
"""

from datetime import timezone

import pandas as pd
import pytest

from cogniverse_agents.routing.orchestration_evaluator import OrchestrationEvaluator

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _evaluator(get_spans):
    ev = object.__new__(OrchestrationEvaluator)
    ev.project_name = "proj"
    ev._processed_span_ids = set()
    ev._last_evaluation_time = None

    class _Traces:
        pass

    traces = _Traces()
    traces.get_spans = get_spans
    ev.provider = type("P", (), {"traces": traces})()
    return ev


@pytest.mark.asyncio
async def test_window_resumes_from_last_evaluation_time():
    calls = []

    async def get_spans(project, start_time, end_time, filters, limit):
        assert filters == {"name": "cogniverse.orchestration"}
        calls.append((start_time, end_time))
        return pd.DataFrame()

    ev = _evaluator(get_spans)

    await ev.evaluate_orchestration_spans(lookback_hours=1)
    first_start, first_end = calls[0]
    # First run: ~1h lookback window, UTC-aware.
    assert first_start.tzinfo == timezone.utc
    assert 3500 <= (first_end - first_start).total_seconds() <= 3700

    await ev.evaluate_orchestration_spans(lookback_hours=1)
    second_start, _ = calls[1]
    # Second run resumes from the first run's end, NOT end2 - 1h.
    assert second_start == first_end


@pytest.mark.asyncio
async def test_failed_query_does_not_advance_resume_point():
    state = {"first": True}

    async def get_spans(project, start_time, end_time, filters, limit):
        if state["first"]:
            state["first"] = False
            raise RuntimeError("telemetry down")
        return pd.DataFrame()

    ev = _evaluator(get_spans)

    await ev.evaluate_orchestration_spans(lookback_hours=2)
    # Failure must leave the resume point unset so the window is re-scanned.
    assert ev._last_evaluation_time is None


def test_ctor_canonicalizes_tenant_for_provider_and_project_scoping():
    """The runtime writes orchestration spans under the canonical tenant
    project; an evaluator built with a raw id must resolve the SAME provider
    scope and project name, not query an empty ``cogniverse-acme`` project."""
    from unittest.mock import MagicMock, patch

    mgr = MagicMock()
    mgr.get_provider.return_value = MagicMock()
    mgr.config.get_project_name.side_effect = lambda tid: f"cogniverse-{tid}"

    with patch(
        "cogniverse_agents.routing.orchestration_evaluator.get_telemetry_manager",
        return_value=mgr,
    ):
        ev = OrchestrationEvaluator(MagicMock(), tenant_id="acme")

    assert ev.tenant_id == "acme:acme"
    mgr.get_provider.assert_called_once_with(tenant_id="acme:acme")
    assert ev.project_name == "cogniverse-acme:acme"
