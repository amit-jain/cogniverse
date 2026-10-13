"""/health names the chat LLM's state; readiness never depends on it.

The LLM is reached the way a request reaches it, through a real LM call to an
HTTP endpoint that answers what Modal answers when its app is undeployed and
when it is serving again.
"""

from __future__ import annotations

import os
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
    SERVING_FRESHNESS_S,
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


def _assert_small_age(age_s: float) -> None:
    assert 0.0 <= age_s < 60.0, age_s


def _not_serving_entry(app: ModalApp, entry: dict) -> None:
    _assert_recent(entry.pop("observed_at"))
    _assert_small_age(entry.pop("age_s"))
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
        "last_answered_at": None,
        "last_answered_age_s": None,
    }


class TestHealthNamesTheLLM:
    def test_before_any_lm_call_the_llm_is_not_called(self, no_agents):
        with _stub_backend() as base:
            response = _client(base).get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["dependencies"] == {
            "llm": {
                "status": "not_called",
                "observed_by": {"scope": "worker", "pid": os.getpid()},
                "freshness_window_s": SERVING_FRESHNESS_S,
                "last_answered_at": None,
                "last_answered_age_s": None,
                "endpoints": [],
            }
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
        llm = body["dependencies"]["llm"]
        (entry,) = llm["endpoints"]
        observed_at = entry.pop("observed_at")
        _assert_recent(observed_at)
        _assert_small_age(entry.pop("age_s"))
        _assert_small_age(entry.pop("last_answered_age_s"))
        _assert_small_age(llm.pop("last_answered_age_s"))
        assert llm == {
            "status": "serving",
            "observed_by": {"scope": "worker", "pid": os.getpid()},
            "freshness_window_s": SERVING_FRESHNESS_S,
            "last_answered_at": observed_at,
            "endpoints": [entry],
        }
        assert entry == {
            "endpoint": modal_app.api_base,
            "model": MODEL,
            "route": None,
            "state": "serving",
            "upstream_status": None,
            "failure": None,
            "reason": "answered its last call",
            "last_answered_at": observed_at,
            "recheck_in_s": None,
        }

    def test_an_llm_that_last_answered_before_the_window_is_idle_not_serving(
        self, no_agents, modal_app, monkeypatch
    ):
        """An answer says nothing once the endpoint has had time to scale to
        zero: /health reports it idle with its age, and makes no LM call."""
        modal_app.deploy()
        _lm(modal_app).forward(f"rewrite {uuid.uuid4()}")
        calls_before = list(modal_app.requests)
        monkeypatch.setattr(lm_endpoint_availability(), "serving_freshness_s", 0.2)
        time.sleep(0.3)

        with _stub_backend() as base:
            response = _client(base).get("/health")

        body = response.json()
        assert (response.status_code, body["status"]) == (200, "healthy")
        llm = body["dependencies"]["llm"]
        (entry,) = llm["endpoints"]
        age_s = entry["age_s"]
        assert 0.3 <= age_s < 60.0
        assert (
            llm["status"],
            llm["freshness_window_s"],
            llm["last_answered_at"],
            entry["state"],
            entry["reason"],
            entry["last_answered_at"],
        ) == (
            "idle",
            0.2,
            entry["observed_at"],
            "idle",
            f"answered its last call {age_s:.0f}s ago and has not been called "
            "since; whether it still serves is unknown until the next call",
            entry["observed_at"],
        )
        assert 0.3 <= llm["last_answered_age_s"] < 60.0
        assert modal_app.requests == calls_before

    def test_one_fresh_answer_among_idle_endpoints_is_serving(
        self, no_agents, modal_app, monkeypatch
    ):
        modal_app.deploy()
        other = ModalApp()
        try:
            other.deploy()
            _lm(other).forward(f"rewrite {uuid.uuid4()}")
            monkeypatch.setattr(lm_endpoint_availability(), "serving_freshness_s", 0.2)
            time.sleep(0.3)
            _lm(modal_app).forward(f"rewrite {uuid.uuid4()}")

            with _stub_backend() as base:
                llm = _client(base).get("/health").json()["dependencies"]["llm"]
        finally:
            other.close()

        assert llm["status"] == "serving"
        assert [(e["endpoint"], e["state"]) for e in llm["endpoints"]] == [
            (other.api_base, "idle"),
            (modal_app.api_base, "serving"),
        ]
        assert llm["last_answered_at"] == llm["endpoints"][1]["observed_at"]

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
