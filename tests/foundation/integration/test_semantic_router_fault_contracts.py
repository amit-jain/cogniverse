"""What a caller gets when a routed completion fails, against the live stack.

The stack is the shipped one (Envoy -> vLLM Semantic Router -> stub upstream);
the stub is made to fail in a named way by a ``FAULT:`` sentinel in the prompt,
so every case here is a real HTTP failure travelling the real router hop.

The router preserves the upstream's status and error body, which is what makes
the status a usable contract -- and what makes litellm's own classes unusable
as one: a 403 from the upstream and a 400 from the router both arrive as
``BadRequestError``. ``RoutedLM`` is the layer that separates them.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from cogniverse_foundation.config.routed_lm import (
    RouterDecodeFailed,
    UpstreamAuthRejected,
    UpstreamRateLimited,
    UpstreamUnavailable,
)
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from tests.foundation.integration._sr_stack.stub_upstream import refusal_body
from tests.utils.semantic_router_stack import ext_proc_message_timeouts

pytestmark = pytest.mark.integration

TENANT = "fault-tenant"
TIER = "free"

# Every budget below is derived from what the shipped config declares, so a
# changed deadline or retry count reaches these assertions.
MESSAGE_TIMEOUT_S, MAX_MESSAGE_TIMEOUT_S = ext_proc_message_timeouts()
ATTEMPT_TIMEOUT_S = LLMEndpointConfig(model="m").request_timeout
ATTEMPTS = LLMEndpointConfig(model="m").num_retries + 1
RETRY_BUDGET_S = ATTEMPT_TIMEOUT_S * ATTEMPTS
ROUTED_MODEL = SemanticRouterConfig().routed_model
# A per-attempt deadline short enough to be exceeded on purpose; the shipped
# one is what the retry-budget assertions above are derived from.
SHORT_TIMEOUT_S = 3.0


@pytest.fixture(scope="module")
def sr_base_url(semantic_router_stack) -> str:
    return semantic_router_stack["base_url"]


def _routed_lm(
    base_url: str,
    tenant_id: str = TENANT,
    request_timeout: float = ATTEMPT_TIMEOUT_S,
):
    """The LM an agent gets for ``tenant_id`` with routing enabled."""
    lm = create_routed_lm(
        endpoint=LLMEndpointConfig(
            model="unused",
            api_base="http://unused:1/v1",
            request_timeout=request_timeout,
        ),
        config=SemanticRouterConfig(enabled=True, semantic_router_url=base_url),
        tenant_id=tenant_id,
        tier=TIER,
        call_site="summarizer_agent",
    )
    lm.cache = False
    return lm


def _outcome(
    base_url: str,
    prompt: str,
    tenant_id: str = TENANT,
    request_timeout: float = ATTEMPT_TIMEOUT_S,
):
    """``("raised", exc)`` or ``("returned", value)`` for one routed call.

    A completion is never allowed to come back empty in place of a failure, so
    the two are told apart by which one happened at all, not by inspecting a
    value that may be falsy.
    """
    try:
        return "returned", _routed_lm(base_url, tenant_id, request_timeout)(prompt)
    except BaseException as exc:  # noqa: BLE001 - the contract under test
        return "raised", exc


def _upstream_calls(base_url: str) -> int:
    """How many requests have reached the stub upstream so far.

    The stub counts every request it serves, faults included, and reports the
    running count in its reflection. A unique prompt keeps the router's
    response cache out of the measurement.
    """
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": f"count {uuid.uuid4()}"}],
        },
        headers={"x-authz-user-id": TENANT, "x-authz-user-groups": TIER},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return json.loads(response.json()["choices"][0]["message"]["content"])["call_index"]


class TestEachUpstreamRefusalGetsItsOwnType:
    """One status in, exactly one typed error out, carrying the routing inputs."""

    def test_a_rejected_key_is_an_auth_rejection_naming_the_tenant(self, sr_base_url):
        how, exc = _outcome(sr_base_url, "FAULT:status:401")

        assert how == "raised"
        assert type(exc) is UpstreamAuthRejected
        assert exc.status == 401
        assert exc.router_code is None
        assert exc.tenant_id == TENANT
        assert exc.tier == TIER
        assert exc.routed_model == ROUTED_MODEL
        assert str(exc) == (
            "the model endpoint rejected the credentials or the tenant: "
            f"tenant={TENANT} tier={TIER} routed_model={ROUTED_MODEL} "
            "status=401 router_code=None"
        )
        assert type(exc.__cause__) is AuthenticationError
        assert exc.__cause__.status_code == 401

    def test_a_forbidden_tenant_carries_the_upstreams_own_error_code(self, sr_base_url):
        how, exc = _outcome(sr_base_url, "FAULT:status:403")

        assert how == "raised"
        assert type(exc) is UpstreamAuthRejected
        assert exc.status == 403
        assert exc.router_code == refusal_body(403)["error"]["code"]
        assert str(exc) == (
            "the model endpoint rejected the credentials or the tenant: "
            f"tenant={TENANT} tier={TIER} routed_model={ROUTED_MODEL} "
            "status=403 router_code=model_not_permitted"
        )
        # litellm hands 403 the same class it hands a request the router
        # refused, so its class alone cannot separate "not allowed" from
        # "malformed"; the typed error does.
        assert type(exc.__cause__) is BadRequestError

    def test_a_quota_refusal_is_rate_limited_not_an_outage(self, sr_base_url):
        how, exc = _outcome(sr_base_url, "FAULT:status:429")

        assert how == "raised"
        assert type(exc) is UpstreamRateLimited
        assert exc.status == 429
        # litellm fills ``code`` with the stringified status when the provider
        # sent none; that is not a router code.
        assert exc.router_code is None
        assert str(exc) == (
            "the model endpoint refused the request for quota: "
            f"tenant={TENANT} tier={TIER} routed_model={ROUTED_MODEL} "
            "status=429 router_code=None"
        )
        assert type(exc.__cause__) is RateLimitError

    def test_an_overloaded_engine_is_an_outage(self, sr_base_url):
        how, exc = _outcome(sr_base_url, "FAULT:status:503")

        assert how == "raised"
        assert type(exc) is UpstreamUnavailable
        assert exc.status == 503
        assert str(exc) == (
            "the model endpoint did not answer: "
            f"tenant={TENANT} tier={TIER} routed_model={ROUTED_MODEL} "
            "status=503 router_code=None"
        )
        assert type(exc.__cause__) is ServiceUnavailableError

    def test_a_connection_reset_is_an_outage_not_a_refusal(self, sr_base_url):
        how, exc = _outcome(sr_base_url, "FAULT:reset")

        assert how == "raised"
        assert type(exc) is UpstreamUnavailable
        # Envoy answers a reset upstream with its own 503, so the caller sees
        # an outage rather than a transport error with no status.
        assert exc.status == 503
        assert type(exc.__cause__) is ServiceUnavailableError

    def test_a_request_the_router_refuses_is_not_an_upstream_fault(self, sr_base_url):
        """A raw provider model id: the router rejects it before any upstream."""
        lm = create_routed_lm(
            endpoint=LLMEndpointConfig(
                model="some-provider/not-a-catalog-model",
                api_base="http://unused:1/v1",
            ),
            config=SemanticRouterConfig(
                enabled=True,
                semantic_router_url=sr_base_url,
                routed_model="some-provider/not-a-catalog-model",
            ),
            tenant_id=TENANT,
            tier=TIER,
            call_site="summarizer_agent",
        )
        lm.cache = False
        with pytest.raises(RouterDecodeFailed) as excinfo:
            lm("hello")

        assert excinfo.value.status == 400
        assert excinfo.value.tenant_id == TENANT
        assert excinfo.value.routed_model == "some-provider/not-a-catalog-model"
        assert type(excinfo.value.__cause__) is BadRequestError

    def test_the_four_failure_modes_are_four_distinct_types(self, sr_base_url):
        """The separation litellm does not make, made once, end to end."""
        raised = {
            prompt: type(_outcome(sr_base_url, prompt)[1]).__name__
            for prompt in (
                "FAULT:status:401",
                "FAULT:status:403",
                "FAULT:status:429",
                "FAULT:status:503",
            )
        }

        assert raised == {
            "FAULT:status:401": "UpstreamAuthRejected",
            "FAULT:status:403": "UpstreamAuthRejected",
            "FAULT:status:429": "UpstreamRateLimited",
            "FAULT:status:503": "UpstreamUnavailable",
        }


class TestTheRetryBudgetIsBounded:
    """Attempts are spent only where a second one can answer.

    ``_upstream_calls`` is itself one request to the stub, so a measured delta
    is ``attempts + 1``.
    """

    def _attempts(self, base_url: str, prompt: str):
        before = _upstream_calls(base_url)
        started = time.monotonic()
        outcome = _outcome(base_url, prompt)
        elapsed = time.monotonic() - started
        return _upstream_calls(base_url) - before - 1, elapsed, outcome

    def test_a_credential_refusal_is_never_retried(self, sr_base_url):
        attempts, elapsed, (how, exc) = self._attempts(sr_base_url, "FAULT:status:401")

        assert how == "raised"
        assert type(exc) is UpstreamAuthRejected
        # A rejected key answers the same way every time, so it costs exactly
        # one upstream attempt however large the endpoint's retry allowance.
        assert attempts == 1
        assert ATTEMPTS == 2
        assert elapsed < ATTEMPT_TIMEOUT_S

    def test_an_outage_spends_the_configured_attempts_and_no_more(self, sr_base_url):
        attempts, elapsed, (how, exc) = self._attempts(sr_base_url, "FAULT:status:503")

        assert how == "raised"
        assert type(exc) is UpstreamUnavailable
        assert attempts == ATTEMPTS
        assert elapsed < RETRY_BUDGET_S

    def test_a_quota_refusal_is_retried_like_an_outage(self, sr_base_url):
        attempts, _, (how, exc) = self._attempts(sr_base_url, "FAULT:status:429")

        assert how == "raised"
        assert type(exc) is UpstreamRateLimited
        assert attempts == ATTEMPTS

    def test_a_request_the_router_refuses_is_never_retried(self, sr_base_url):
        """A 400 is cogniverse's own request; re-sending it cannot help."""
        before = _upstream_calls(sr_base_url)
        lm = create_routed_lm(
            endpoint=LLMEndpointConfig(
                model="some-provider/not-a-catalog-model",
                api_base="http://unused:1/v1",
            ),
            config=SemanticRouterConfig(
                enabled=True,
                semantic_router_url=sr_base_url,
                routed_model="some-provider/not-a-catalog-model",
            ),
            tenant_id=TENANT,
            tier=TIER,
            call_site="summarizer_agent",
        )
        lm.cache = False
        with pytest.raises(RouterDecodeFailed):
            lm("hello")

        # The router refuses before the request reaches any upstream, so the
        # stub sees only this reading's own call.
        assert _upstream_calls(sr_base_url) - before == 1

    def test_the_ext_proc_deadline_does_not_bound_the_upstream_answer(
        self, sr_base_url
    ):
        """An upstream slower than the ext_proc per-message deadline still answers.

        ``message_timeout`` bounds one Envoy<->router exchange, not the
        upstream's reply, so a slow generation is served rather than cancelled.
        """
        trickle_s = MESSAGE_TIMEOUT_S + 5
        started = time.monotonic()
        how, value = _outcome(sr_base_url, f"FAULT:trickle:{trickle_s}")
        elapsed = time.monotonic() - started

        assert how == "returned"
        assert value == ["late"]
        assert elapsed >= MESSAGE_TIMEOUT_S
        assert elapsed < ATTEMPT_TIMEOUT_S
        assert MESSAGE_TIMEOUT_S < MAX_MESSAGE_TIMEOUT_S


class TestConcurrentFailuresEachNameTheirOwnTenant:
    TENANTS = tuple(f"fault-tenant-{index}" for index in range(8))

    def test_eight_concurrent_rejections_raise_eight_times(self, sr_base_url):
        with ThreadPoolExecutor(max_workers=len(self.TENANTS)) as pool:
            outcomes = list(
                pool.map(
                    lambda tenant: _outcome(sr_base_url, "FAULT:status:401", tenant),
                    self.TENANTS,
                )
            )

        assert Counter(how for how, _ in outcomes) == {"raised": len(self.TENANTS)}
        assert Counter(type(exc).__name__ for _, exc in outcomes) == {
            "UpstreamAuthRejected": len(self.TENANTS)
        }
        assert sorted(exc.tenant_id for _, exc in outcomes) == sorted(self.TENANTS)
        assert {exc.status for _, exc in outcomes} == {401}


class TestAnUpstreamThatDoesNotAnswerIsAnOutage:
    """No status arrives at all: the client gives up, and that is an outage.

    Both cases ride the real stack; the upstream is made to hold the request
    open past the client's deadline, once by sleeping and once by being frozen.
    """

    def test_an_upstream_slower_than_the_attempt_deadline_is_an_outage(
        self, sr_base_url
    ):
        before = _upstream_calls(sr_base_url)
        started = time.monotonic()
        how, exc = _outcome(
            sr_base_url,
            f"FAULT:trickle:{SHORT_TIMEOUT_S * 3}",
            request_timeout=SHORT_TIMEOUT_S,
        )
        elapsed = time.monotonic() - started
        # ``_upstream_calls`` is itself one request to the stub.
        attempts = _upstream_calls(sr_base_url) - before - 1

        assert how == "raised"
        assert type(exc) is UpstreamUnavailable
        assert exc.status == 408
        assert exc.tenant_id == TENANT
        assert type(exc.__cause__) is Timeout
        # A timeout is retried like any outage: every allowed attempt is spent.
        assert attempts == ATTEMPTS
        assert elapsed >= SHORT_TIMEOUT_S * ATTEMPTS
        assert elapsed < SHORT_TIMEOUT_S * ATTEMPTS + SHORT_TIMEOUT_S

    def test_a_frozen_upstream_is_an_outage(self, semantic_router_stack):
        base_url = semantic_router_stack["base_url"]
        stub = semantic_router_stack["stub_container"]
        subprocess.run(["docker", "pause", stub], check=True, timeout=30)
        try:
            started = time.monotonic()
            how, exc = _outcome(
                base_url, f"hello {uuid.uuid4()}", request_timeout=SHORT_TIMEOUT_S
            )
            elapsed = time.monotonic() - started
        finally:
            subprocess.run(["docker", "unpause", stub], check=True, timeout=30)

        assert how == "raised"
        assert type(exc) is UpstreamUnavailable
        assert exc.status == 408
        assert type(exc.__cause__) is Timeout
        assert elapsed >= SHORT_TIMEOUT_S * ATTEMPTS

        # The stack answers again once the upstream is thawed.
        how, value = _outcome(base_url, f"hello {uuid.uuid4()}")
        assert how == "returned"
        assert json.loads(value[0])["echo"].startswith("hello ")
