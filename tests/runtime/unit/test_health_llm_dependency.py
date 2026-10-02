"""/health names the chat LLM's state; readiness never depends on it.

The LLM is reached the way a request reaches it, through a real LM call to an
HTTP endpoint that answers what Modal answers when its app is undeployed and
when it is serving again.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import openai
import pytest

import cogniverse_vespa.backend  # noqa: F401 - registers the vespa backend class
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.lm_endpoint_availability import (
    NOT_SERVING_RECHECK_S,
    LMEndpointNotServing,
    lm_endpoint_availability,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_runtime.routers import health
from tests.runtime.unit.test_health_backend_reachability import (
    _client,
    _dead_url,
    _NoAgents,
    _stub_backend,
)
from tests.utils.modal_app import ModalApp

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_injected_agent_registry")]

MODEL = "openai/google/gemma-4-e4b-it"
NOT_SERVING_REASON = (
    "answered HTTP 404: nothing is deployed for this model; "
    "calls fail fast until the next recheck"
)


@pytest.fixture
def no_agents(monkeypatch):
    monkeypatch.setattr(
        health,
        "_get_agent_registry",
        lambda: _NoAgents(),
    )


@pytest.fixture
def modal_app():
    app = ModalApp()
    yield app
    app.close()


def _lm(app: ModalApp):
    return create_dspy_lm(
        LLMEndpointConfig(model=MODEL, api_base=app.api_base, api_key="bearer")
    )


def _undeployed_call(app: ModalApp) -> LMEndpointNotServing:
    with pytest.raises(LMEndpointNotServing) as raised:
        _lm(app).forward(f"rewrite {uuid.uuid4()}")
    return raised.value


def _assert_recent(observed_at: str) -> None:
    age_s = (
        datetime.now(timezone.utc) - datetime.fromisoformat(observed_at)
    ).total_seconds()
    assert 0.0 <= age_s < 60.0, observed_at


def _not_serving_entry(app: ModalApp, entry: dict) -> None:
    _assert_recent(entry.pop("observed_at"))
    recheck_in_s = entry.pop("recheck_in_s")
    assert NOT_SERVING_RECHECK_S - 60.0 < recheck_in_s <= NOT_SERVING_RECHECK_S
    assert entry == {
        "endpoint": app.api_base,
        "model": MODEL,
        "route": None,
        "state": "not_serving",
        "upstream_status": 404,
        "failure": None,
        "reason": NOT_SERVING_REASON,
    }


class TestHealthNamesTheLLM:
    def test_before_any_lm_call_the_llm_is_not_called(self, no_agents):
        with _stub_backend() as base:
            response = _client(base).get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["dependencies"] == {
            "llm": {"status": "not_called", "endpoints": []}
        }

    def test_an_undeployed_llm_is_degraded_with_its_reason(self, no_agents, modal_app):
        _undeployed_call(modal_app)

        with _stub_backend() as base:
            response = _client(base).get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["dependencies"]["llm"]["status"] == "not_serving"
        (entry,) = body["dependencies"]["llm"]["endpoints"]
        _not_serving_entry(modal_app, entry)

    def test_a_redeployed_llm_is_healthy_after_its_recheck(
        self, no_agents, modal_app, monkeypatch
    ):
        monkeypatch.setattr(lm_endpoint_availability(), "recheck_after_s", 0.2)
        _undeployed_call(modal_app)
        modal_app.deploy()
        time.sleep(0.2)
        prompt = f"rewrite {uuid.uuid4()}"

        answer = _lm(modal_app).forward(prompt)
        with _stub_backend() as base:
            response = _client(base).get("/health")

        assert answer.choices[0].message.content == f"served:{prompt}"
        body = response.json()
        assert (response.status_code, body["status"]) == (200, "healthy")
        (entry,) = body["dependencies"]["llm"]["endpoints"]
        _assert_recent(entry.pop("observed_at"))
        assert body["dependencies"]["llm"]["status"] == "serving"
        assert entry == {
            "endpoint": modal_app.api_base,
            "model": MODEL,
            "route": None,
            "state": "serving",
            "upstream_status": None,
            "failure": None,
            "reason": "answered its last call",
            "recheck_in_s": None,
        }

    def test_an_unreachable_llm_is_degraded_as_failing(self, no_agents):
        """litellm reports a refused connection as a 500 InternalServerError;
        that typed status is what the endpoint is recorded with."""
        dead = f"{_dead_url()}/v1"
        lm = create_dspy_lm(
            LLMEndpointConfig(
                model=MODEL,
                api_base=dead,
                api_key="bearer",
                num_retries=0,
                request_timeout=2.0,
            )
        )
        with pytest.raises(openai.InternalServerError):
            lm.forward("rewrite")

        with _stub_backend() as base:
            body = _client(base).get("/health").json()

        assert body["status"] == "degraded"
        assert body["dependencies"]["llm"]["status"] == "failing"
        (entry,) = body["dependencies"]["llm"]["endpoints"]
        assert (
            entry["endpoint"],
            entry["state"],
            entry["upstream_status"],
            entry["failure"],
            entry["reason"],
            entry["recheck_in_s"],
        ) == (
            dead,
            "failing",
            500,
            "InternalServerError",
            "its last call failed: InternalServerError (HTTP 500)",
            None,
        )
        assert lm_endpoint_availability().admit(lm.availability_endpoint()) is None, (
            "a failing endpoint is still called"
        )

    def test_a_backend_outage_stays_unhealthy_and_still_names_the_llm(
        self, no_agents, modal_app
    ):
        _undeployed_call(modal_app)

        response = _client(_dead_url()).get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"
        assert body["dependencies"]["llm"]["status"] == "not_serving"


class TestReadinessIgnoresTheLLM:
    """A down LLM never takes the pod out of the Service: search serves
    without it."""

    def test_ready_with_the_llm_not_serving(self, modal_app):
        _undeployed_call(modal_app)

        with _stub_backend() as base:
            response = _client(base).get("/health/ready")

        assert (response.status_code, response.json()) == (
            200,
            {
                "status": "ready",
                "backends": len(BackendRegistry.get_instance().list_backends()),
            },
        )

    def test_liveness_with_the_llm_not_serving(self, modal_app):
        _undeployed_call(modal_app)

        response = _client(None).get("/health/live")

        assert (response.status_code, response.json()) == (200, {"status": "alive"})
