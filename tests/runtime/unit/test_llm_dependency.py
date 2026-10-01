"""Which HTTP answer a request that failed on the chat LLM gets."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import litellm
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_foundation.config.lm_endpoint_availability import (
    LMEndpoint,
    LMEndpointNotServing,
)
from cogniverse_foundation.config.routed_lm import (
    RouterDecodeFailed,
    UpstreamAuthRejected,
    UpstreamNotServing,
    UpstreamRateLimited,
    UpstreamUnavailable,
)
from cogniverse_runtime.llm_dependency import llm_dependency_failure
from cogniverse_runtime.routers import agents as agents_router

pytestmark = pytest.mark.unit

ROUTED = "openai/cogniverse-classification"
FIELDS = dict(
    router_code=None,
    tenant_id="acme:prod",
    tier="pro",
    routed_model=ROUTED,
    endpoint="http://admin:sekrit@router:8801/v1",
)


def _routed(kind, status, **extra):
    return kind(f"{kind.__name__} summary", status=status, **FIELDS, **extra)


def _raised_from(outer: BaseException, cause: BaseException) -> BaseException:
    try:
        try:
            raise cause
        except BaseException as inner:
            raise outer from inner
    except BaseException as raised:
        return raised


def _answer(exc):
    failure = llm_dependency_failure(exc)
    return (
        failure.http_status,
        failure.error,
        failure.failure,
        failure.upstream_status,
        failure.model,
        failure.retry_after_s,
    )


class TestTheAnswerFollowsTheFailureType:
    def test_a_not_serving_routed_endpoint_is_503_with_its_recheck(self):
        failure = _routed(UpstreamNotServing, 404, failed_fast=True, recheck_in_s=12.2)
        assert _answer(failure) == (
            503,
            "llm_unavailable",
            "UpstreamNotServing",
            404,
            ROUTED,
            13,
        )

    def test_a_recheck_in_flight_still_tells_the_caller_to_wait_a_second(self):
        failure = _routed(UpstreamNotServing, 404, failed_fast=True, recheck_in_s=0.0)
        assert llm_dependency_failure(failure).retry_after_s == 1

    @pytest.mark.parametrize(
        ("kind", "status"),
        [
            (UpstreamUnavailable, 503),
            (UpstreamUnavailable, None),
            (UpstreamRateLimited, 429),
        ],
    )
    def test_an_unavailable_routed_endpoint_is_503(self, kind, status):
        assert _answer(_routed(kind, status)) == (
            503,
            "llm_unavailable",
            kind.__name__,
            status,
            ROUTED,
            None,
        )

    @pytest.mark.parametrize(
        ("kind", "status"), [(UpstreamAuthRejected, 403), (RouterDecodeFailed, 400)]
    )
    def test_a_rejected_routed_request_is_502(self, kind, status):
        assert _answer(_routed(kind, status)) == (
            502,
            "llm_request_rejected",
            kind.__name__,
            status,
            ROUTED,
            None,
        )

    def test_a_not_serving_direct_endpoint_is_503_with_its_recheck(self):
        failure = LMEndpointNotServing(
            endpoint=LMEndpoint(api_base="https://llm.example/v1", model="openai/m"),
            status=404,
            failed_fast=False,
            recheck_in_s=30.0,
        )
        assert _answer(failure) == (
            503,
            "llm_unavailable",
            "LMEndpointNotServing",
            404,
            "openai/m",
            30,
        )

    def test_direct_provider_failures_split_on_their_status(self):
        request = httpx.Request("POST", "http://llm.example/v1/chat/completions")
        unavailable = litellm.ServiceUnavailableError(
            message="down", llm_provider="openai", model="m"
        )
        timed_out = litellm.Timeout(message="slow", model="m", llm_provider="openai")
        rejected = litellm.BadRequestError(
            message="bad", model="m", llm_provider="openai"
        )
        connection = litellm.APIConnectionError(
            message="refused", llm_provider="openai", model="m", request=request
        )

        assert [_answer(e) for e in (unavailable, timed_out, rejected, connection)] == [
            (503, "llm_unavailable", "ServiceUnavailableError", 503, "m", None),
            (503, "llm_unavailable", "Timeout", None, "m", None),
            (502, "llm_request_rejected", "BadRequestError", 400, "m", None),
            (503, "llm_unavailable", "APIConnectionError", None, "m", None),
        ]


class TestOnlyACauseIsAttributedToTheLLM:
    def test_a_wrapper_raised_from_the_llm_failure_is_answered_as_it(self):
        wrapped = _raised_from(
            RuntimeError("entity extraction dispatch failed"),
            _routed(UpstreamUnavailable, 503),
        )
        assert _answer(wrapped)[:3] == (503, "llm_unavailable", "UpstreamUnavailable")

    def test_a_handled_llm_failure_is_not_the_cause_of_a_later_one(self):
        try:
            try:
                raise _routed(UpstreamUnavailable, 503)
            except UpstreamUnavailable:
                raise KeyError("unrelated")
        except KeyError as raised:
            assert llm_dependency_failure(raised) is None

    def test_an_exception_group_is_answered_by_its_llm_leaf(self):
        group = ExceptionGroup(
            "fan-out", [ValueError("leg"), _routed(UpstreamAuthRejected, 401)]
        )
        assert _answer(group)[:3] == (
            502,
            "llm_request_rejected",
            "UpstreamAuthRejected",
        )

    def test_an_unrelated_failure_is_none(self):
        assert llm_dependency_failure(RuntimeError("vespa refused")) is None


def _post_with_dispatch_raising(monkeypatch, exc):
    stub = MagicMock()
    stub.dispatch = AsyncMock(side_effect=exc)
    monkeypatch.setattr(agents_router, "_ensure_dispatcher", lambda: stub)
    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    with TestClient(app, raise_server_exceptions=False) as client:
        return client.post(
            "/agents/gateway_agent/process",
            json={
                "agent_name": "gateway_agent",
                "query": "find machine learning videos and summarize them",
                "context": {"tenant_id": "acme:prod"},
                "session_id": "sess-7",
            },
        )


class TestTheAgentRouteAnswersTheLLMFailure:
    def test_a_rejected_request_is_502_and_never_leaks_the_endpoint(self, monkeypatch):
        response = _post_with_dispatch_raising(
            monkeypatch, _routed(UpstreamAuthRejected, 401)
        )

        assert response.status_code == 502
        assert "retry-after" not in response.headers
        assert response.json() == {
            "detail": {
                "error": "llm_request_rejected",
                "dependency": "llm",
                "agent": "gateway_agent",
                "failure": "UpstreamAuthRejected",
                "upstream_status": 401,
                "model": ROUTED,
                "retry_after_s": None,
                "request_id": "sess-7",
                "message": (
                    "Agent 'gateway_agent' could not complete: the chat LLM "
                    "rejected the request (UpstreamAuthRejected, upstream HTTP 401)."
                ),
            }
        }
        assert "sekrit" not in response.text

    def test_a_not_serving_llm_is_503_with_retry_after(self, monkeypatch):
        response = _post_with_dispatch_raising(
            monkeypatch,
            _routed(UpstreamNotServing, 404, failed_fast=True, recheck_in_s=4.5),
        )

        assert (response.status_code, response.headers["retry-after"]) == (503, "5")
        assert response.json()["detail"]["message"] == (
            "Agent 'gateway_agent' could not complete: the chat LLM is not serving: "
            "nothing is deployed for the model (UpstreamNotServing, upstream HTTP "
            "404). Retry after 5s."
        )

    def test_any_other_failure_keeps_the_opaque_500(self, monkeypatch):
        response = _post_with_dispatch_raising(monkeypatch, RuntimeError("boom"))

        assert response.status_code == 500
        assert response.json() == {
            "detail": (
                "Agent 'gateway_agent' failed with RuntimeError "
                "(request_id=sess-7). See runtime logs for detail."
            )
        }
