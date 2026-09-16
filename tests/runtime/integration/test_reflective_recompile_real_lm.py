"""The reflective recompile against a real LM.

The metric's arity is decided by GEPA's own adapter, and the reflective
mutation only runs when a candidate scores below 1.0. Recorded failing rows are
exactly the served module's own outputs, so this test produces them from the
real LM and feeds them back as the outputs the candidate must not reproduce —
the production shape of an all-failure agent, driven end to end.
"""

from __future__ import annotations

import json

import pytest

from cogniverse_runtime import optimization_cli
from cogniverse_runtime.optimization_cli import _reflective_compile, _served_module
from tests.utils.hermetic_llm import MODEL as LM_MODEL
from tests.utils.hermetic_llm import ensure_llm

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

_CONTENT = [
    "Saturn's rings are made of ice and rock and span 280,000 km.",
    "The red kite is a raptor that hunts over open farmland.",
    "Kite surfing needs steady onshore wind and a wide beach.",
]


@pytest.fixture(scope="module")
def real_lm():
    """The test LM this repo resolves for every LM-backed integration test,
    built through the constructor the runtime builds its own LM with so the
    bearer resolves the same way."""
    from cogniverse_foundation.config.llm_factory import create_budgeted_dspy_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    base_url = ensure_llm()
    if base_url is None:
        pytest.fail("the test LM could not be provisioned")
    return create_budgeted_dspy_lm(
        LLMEndpointConfig(
            model=f"openai/{LM_MODEL}",
            api_base=base_url,
            temperature=0.1,
            max_tokens=512,
        )
    )


def test_reflective_recompile_runs_against_a_real_lm(real_lm, monkeypatch):
    import dspy
    from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

    from cogniverse_agents.summarizer_agent import SummarizationModule

    calls: list[tuple[int, object]] = []
    real_factory = optimization_cli._reflective_metric

    def recording_factory(agent_name):
        metric = real_factory(agent_name)

        def recorded(*args, **kwargs):
            result = metric(*args, **kwargs)
            calls.append((len(args) + len(kwargs), result))
            return result

        return recorded

    monkeypatch.setattr(optimization_cli, "_reflective_metric", recording_factory)

    # The recorded failures are what the served module itself produces, so the
    # candidate reproducing them scores below 1.0 and GEPA must reflect.
    stock = _served_module("summary")
    rows = []
    with dspy.context(lm=real_lm):
        for content in _CONTENT:
            prediction = stock(
                content=content,
                query="Summarize this",
                summary_type="brief",
                keyframes=[],
            )
            rows.append(
                {
                    "query": "Summarize this",
                    "output": json.dumps({"summary": str(prediction.summary)}),
                }
            )

    assert [json.loads(row["output"])["summary"] for row in rows] != ["", "", ""]

    with dspy.context(lm=real_lm):
        compiled = _reflective_compile("summary", rows, real_lm, max_metric_calls=9)

    assert isinstance(compiled, SummarizationModule)
    # Every call GEPA made reached the metric and returned the feedback record;
    # none raised on arity, which is the regression this covers.
    assert calls, "GEPA never called the reflective metric"
    assert all(isinstance(result, ScoreWithFeedback) for _, result in calls)
    assert {arity for arity, _ in calls} <= {2, 3, 5}
    # The five-argument feedback shape is GEPA's reflective path: it fired, so a
    # candidate was evaluated against real feedback from real LM output.
    assert 5 in {arity for arity, _ in calls}
    assert all(0.0 <= result.score <= 1.0 for _, result in calls)
    assert {
        result.feedback.startswith("The recorded failing summary was ")
        for _, result in calls
    } == {True}
