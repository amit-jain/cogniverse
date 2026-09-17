"""A generation that cannot produce its required outputs ends the turn.

Pins that a completion which names no value for a required output field is a
failed turn carrying the fields the LM never produced — not a success carrying
a defaulted object. The input is sized from the shipped completion budget, so
the generation runs out of room before it can fill the signature's outputs.

Precondition: the cluster's primary LM runs with the ``max_tokens`` the shipped
config declares. The test reads that budget from the config the chart rendered
into the cluster rather than restating it, and sizes
its own content from it, so the turn's failure is the test's choice and not the
model's.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.e2e.conftest import (
    RUNTIME,
    _kubectl_e2e,
    _kubectl_e2e_command,
    _require_kubectl_success,
    register_tenant_and_wait,
    unique_id,
)

# ``charts/cogniverse/files/config.json`` is a Helm template; the runtime reads
# the rendered copy this ConfigMap carries.
RENDERED_CONFIG_ARGS = (
    "-n",
    "cogniverse",
    "get",
    "configmap/cogniverse-config",
    "-o",
    "jsonpath={.data.config\\.json}",
)
# One recorded passage per multiple of the completion budget: a faithful
# detailed summary of this many distinct passages cannot fit in the budget, so
# the generation stops before it names every required output.
PASSAGES_PER_BUDGET_TOKEN = 2
PASSAGE = (
    "The recorded segment describes a distinct sequence of events with its own "
    "participants, location, timing and outcome, none of which appear in any "
    "other segment of this collection. "
)
PROCESS_TIMEOUT_S = 600.0


def _shipped_primary_max_tokens() -> int:
    result = _kubectl_e2e(*RENDERED_CONFIG_ARGS)
    _require_kubectl_success(result, _kubectl_e2e_command(*RENDERED_CONFIG_ARGS))
    config = json.loads(result.stdout)
    return int(config["llm_config"]["primary"]["max_tokens"])


def _served_output_fields() -> set[str]:
    """Output fields the served summarizer module's signature requires."""
    from cogniverse_agents.summarizer_agent import SummarizationModule

    predictor = next(iter(SummarizationModule().predictors()))
    return set(predictor.signature.output_fields)


@pytest.fixture(scope="module")
def truncation_tenant() -> str:
    org_id = unique_id("opt_trunc")
    suffix = org_id.rsplit("_", 1)[1]
    tenant_id = f"{org_id}:t1"
    resp = httpx.post(
        f"{RUNTIME}/admin/organizations",
        json={
            "org_id": org_id,
            "org_name": f"opt-trunc-{suffix}",
            "created_by": "e2e",
        },
        timeout=60.0,
    )
    assert resp.status_code in (200, 201, 409), resp.text
    register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)
    return tenant_id


@pytest.mark.e2e
class TestIncompleteGenerationEndsTheTurn:
    """The route reports the generation failure instead of answering."""

    def test_a_generation_that_cannot_fit_the_budget_is_a_failed_turn(
        self, truncation_tenant
    ):
        budget = _shipped_primary_max_tokens()
        request_id = "opt-truncation-turn"
        passages = [
            {
                "document_id": f"passage-{index}",
                "content": f"Passage {index}. {PASSAGE}",
                "score": 1.0,
            }
            for index in range(budget * PASSAGES_PER_BUDGET_TOKEN)
        ]

        response = httpx.post(
            f"{RUNTIME}/agents/summarizer_agent/process",
            json={
                "agent_name": "summarizer_agent",
                "query": "Summarise every recorded segment in full detail.",
                "top_k": len(passages),
                "context": {
                    "tenant_id": truncation_tenant,
                    "request_id": request_id,
                    "summary_type": "detailed",
                    "search_results": passages,
                },
            },
            timeout=PROCESS_TIMEOUT_S,
        )
        assert response.status_code == 200, response.text[:500]
        body = response.json()

        assert body["status"] == "error", body
        assert body["agent"] == "summarizer_agent", body
        prefix = (
            f"Agent 'summarizer_agent' generation for request '{request_id}' "
            "produced no "
        )
        assert body["error"].startswith(prefix), body
        produced = body["error"][len(prefix) :]

        # The turn names the signature's own fields, or says the completion
        # carried no parsable output at all. A renamed output field breaks this.
        if produced != "parsable output":
            named = produced.split(", ")
            assert named == sorted(named), body
            assert set(named) <= _served_output_fields(), body

        # Nothing was answered and nothing was synthesised in place of the
        # fields the LM never produced.
        assert "answer" not in body, body
        assert "summary" not in body, body
        assert "result" not in body, body
