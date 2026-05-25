"""Regression test for the dashboard 'Save Evaluation' provider-resolution
path, exercised against a real Phoenix instance.

The dashboard resolved the evaluation provider via
``EvaluationRegistry.get_evaluation_provider(...)`` — a classmethod that does
not exist (the helper is a module-level function) — so clicking 'Save
Evaluation' raised ``AttributeError`` before any evaluation was logged. This
test pins the real resolution + log path so the broken call shape can't return.
"""

import pytest

from cogniverse_evaluation.providers.registry import (
    EvaluationRegistry,
    get_evaluation_provider,
)


@pytest.mark.integration
@pytest.mark.ci_fast
def test_dashboard_save_evaluation_resolves_phoenix_provider(
    search_evaluator_provider, phoenix_container
):
    """``get_evaluation_provider(name="phoenix", tenant_id=..., config=...)``
    resolves and initializes a real PhoenixEvaluationProvider, and the
    dashboard's follow-up ``log_session_evaluation`` call completes against
    real Phoenix.
    """
    from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import (
        PhoenixEvaluationProvider,
    )

    # The exact bug: the helper is module-level, never a classmethod on the
    # registry. Lock that so the broken ``EvaluationRegistry.get_evaluation_provider``
    # call shape can't be reintroduced and silently resolve.
    assert not hasattr(EvaluationRegistry, "get_evaluation_provider")

    provider = get_evaluation_provider(
        name="phoenix",
        tenant_id="test:unit",
        config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
            "project_name": "cogniverse-search",
        },
    )
    assert isinstance(provider, PhoenixEvaluationProvider)
    assert provider._initialized is True

    # The dashboard's next call after resolution. Returns None and must not
    # raise against real Phoenix.
    result = provider.log_session_evaluation(
        session_id="dashboard-regression-session",
        evaluation_name="dashboard_annotation",
        session_score=0.8,
        session_outcome="good",
        turn_scores=None,
        explanation="regression: dashboard save-evaluation path",
        metadata={"num_turns": 2, "queries": ["q1", "q2"]},
    )
    assert result is None
