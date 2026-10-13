"""What an LM endpoint that answered 404 does to the calls that follow it."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from cogniverse_foundation.config.lm_endpoint_availability import (
    MAX_TRACKED_ENDPOINTS,
    NOT_SERVING_RECHECK_S,
    PROBE_VERDICT_S,
    SERVING_FRESHNESS_S,
    LMEndpoint,
    LMEndpointAvailability,
    LMEndpointNotServing,
    not_serving_cause,
)
from cogniverse_foundation.config.request_body import http_status_of
from cogniverse_foundation.inference_specs import INFERENCE_SERVICE_SPECS
from cogniverse_foundation.telemetry.span_contract import (
    LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE,
    LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE,
    LLM_ENDPOINT_STATE_ATTRIBUTE,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

ROUTER = "http://cogniverse-semantic-router-envoy:8801/v1"
CLASSIFICATION = LMEndpoint(
    api_base=ROUTER, model="openai/cogniverse-classification", route="pro"
)
# 2026-10-01T00:00:00Z
EPOCH = 1790812800.0


class Clocks:
    """A monotonic clock and a wall clock that advance together."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> float:
        return EPOCH + (self.now - 1000.0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _iso(clocks: Clocks) -> str:
    return datetime.fromtimestamp(clocks.wall(), tz=timezone.utc).isoformat()


@pytest.fixture
def clocks() -> Clocks:
    return Clocks()


@pytest.fixture
def availability(clocks) -> LMEndpointAvailability:
    return LMEndpointAvailability(clock=clocks.monotonic, wall_clock=clocks.wall)


class TestNotServingFailsFast:
    def test_a_404_names_the_endpoint_and_is_not_itself_a_fast_failure(
        self, availability
    ):
        error = availability.not_serving(CLASSIFICATION, 404)

        assert type(error) is LMEndpointNotServing
        assert (error.endpoint, error.status, error.failed_fast) == (
            CLASSIFICATION,
            404,
            False,
        )
        assert http_status_of(error) == 404
        assert error.recheck_in_s == NOT_SERVING_RECHECK_S
        assert str(error) == (
            f"the LM endpoint is not serving: endpoint={ROUTER} "
            "model=openai/cogniverse-classification route=pro answered HTTP 404; "
            f"next recheck in {NOT_SERVING_RECHECK_S:.1f}s"
        )

    def test_the_next_call_inside_the_window_fails_fast(self, availability, clocks):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(10.0)

        with pytest.raises(LMEndpointNotServing) as raised:
            availability.admit(CLASSIFICATION)

        assert (raised.value.failed_fast, raised.value.recheck_in_s) == (
            True,
            NOT_SERVING_RECHECK_S - 10.0,
        )
        assert str(raised.value).endswith(
            "answered HTTP 404; failing fast, "
            f"next recheck in {NOT_SERVING_RECHECK_S - 10.0:.1f}s"
        )

    def test_other_endpoints_routes_and_models_are_untouched(self, availability):
        availability.not_serving(CLASSIFICATION, 404)

        for other in (
            LMEndpoint(api_base=ROUTER, model=CLASSIFICATION.model, route="free"),
            LMEndpoint(api_base=ROUTER, model="openai/auto", route="pro"),
            LMEndpoint(
                api_base="http://other:8801/v1", model=CLASSIFICATION.model, route="pro"
            ),
        ):
            assert availability.admit(other) is None

    def test_an_unseen_endpoint_is_admitted_without_a_probe(self, availability):
        assert availability.admit(CLASSIFICATION) is None


class TestRecheck:
    def test_one_call_rechecks_once_the_window_passes(self, availability, clocks):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)

        probe = availability.admit(CLASSIFICATION)

        assert probe is not None
        with pytest.raises(LMEndpointNotServing) as raised:
            availability.admit(CLASSIFICATION)
        assert (raised.value.failed_fast, raised.value.recheck_in_s) == (True, 0.0)

    def test_an_answered_recheck_reopens_the_endpoint(self, availability, clocks):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)
        probe = availability.admit(CLASSIFICATION)

        availability.answered(CLASSIFICATION)
        availability.release(CLASSIFICATION, probe)

        assert [availability.admit(CLASSIFICATION) for _ in range(3)] == [None] * 3
        assert availability.snapshot() == [
            {
                "endpoint": ROUTER,
                "model": CLASSIFICATION.model,
                "route": "pro",
                "state": "serving",
                "upstream_status": None,
                "failure": None,
                "reason": "answered its last call",
                "observed_at": _iso(clocks),
                "age_s": 0.0,
                "last_answered_at": _iso(clocks),
                "last_answered_age_s": 0.0,
                "recheck_in_s": None,
            }
        ]

    def test_a_recheck_answered_404_starts_a_fresh_window(self, availability, clocks):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S + 5.0)
        probe = availability.admit(CLASSIFICATION)

        availability.not_serving(CLASSIFICATION, 404)
        availability.release(CLASSIFICATION, probe)
        clocks.advance(1.0)

        with pytest.raises(LMEndpointNotServing) as raised:
            availability.admit(CLASSIFICATION)
        assert raised.value.recheck_in_s == NOT_SERVING_RECHECK_S - 1.0

    def test_a_recheck_that_failed_otherwise_stops_the_fast_failure(
        self, availability, clocks
    ):
        """A timeout is a cold start, not an undeployed endpoint: calls go
        through again."""
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)
        probe = availability.admit(CLASSIFICATION)

        availability.failed(CLASSIFICATION, status=None, failure="APITimeoutError")
        availability.release(CLASSIFICATION, probe)

        assert availability.admit(CLASSIFICATION) is None
        assert availability.snapshot()[0]["state"] == "failing"
        assert availability.snapshot()[0]["reason"] == (
            "its last call failed: APITimeoutError"
        )

    def test_a_recheck_abandoned_without_a_verdict_hands_over_to_the_next_call(
        self, availability, clocks
    ):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)
        abandoned = availability.admit(CLASSIFICATION)

        availability.release(CLASSIFICATION, abandoned)

        assert availability.admit(CLASSIFICATION) is not None

    def test_a_stale_release_does_not_end_a_newer_recheck(self, availability, clocks):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)
        first = availability.admit(CLASSIFICATION)
        availability.release(CLASSIFICATION, first)
        second = availability.admit(CLASSIFICATION)

        availability.release(CLASSIFICATION, first)

        assert second is not None
        with pytest.raises(LMEndpointNotServing):
            availability.admit(CLASSIFICATION)

    def test_a_recheck_still_pending_after_the_verdict_window_admits_everyone(
        self, availability, clocks
    ):
        """An undeployed endpoint answers 404 at once; one still pending is
        booting, and its callers wait for it like any cold start."""
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)
        assert availability.admit(CLASSIFICATION) is not None

        clocks.advance(PROBE_VERDICT_S - 0.5)
        with pytest.raises(LMEndpointNotServing):
            availability.admit(CLASSIFICATION)
        clocks.advance(0.5)
        assert availability.admit(CLASSIFICATION) is None


class TestRefusalWithoutAdmitting:
    """What a caller asks before preparing a call it would not send."""

    def test_inside_the_window_it_names_the_refusal(self, availability, clocks):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(4.0)

        refusal = availability.refusal(CLASSIFICATION)

        assert (type(refusal), refusal.failed_fast, refusal.recheck_in_s) == (
            LMEndpointNotServing,
            True,
            NOT_SERVING_RECHECK_S - 4.0,
        )
        assert refusal.span_attributes() == {
            LLM_ENDPOINT_STATE_ATTRIBUTE: "not_serving",
            LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE: True,
            LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE: NOT_SERVING_RECHECK_S - 4.0,
        }

    def test_once_the_window_passes_it_leaves_the_recheck_to_the_call(
        self, availability, clocks
    ):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)

        assert availability.refusal(CLASSIFICATION) is None
        assert availability.admit(CLASSIFICATION) is not None

    def test_an_endpoint_that_answered_is_never_refused(self, availability):
        availability.answered(CLASSIFICATION)
        assert availability.refusal(CLASSIFICATION) is None


class TestSnapshot:
    def test_a_not_serving_endpoint_reports_its_status_reason_and_recheck(
        self, availability, clocks
    ):
        availability.not_serving(CLASSIFICATION, 404)
        observed = _iso(clocks)
        clocks.advance(12.5)

        assert availability.snapshot() == [
            {
                "endpoint": ROUTER,
                "model": CLASSIFICATION.model,
                "route": "pro",
                "state": "not_serving",
                "upstream_status": 404,
                "failure": None,
                "reason": (
                    "answered HTTP 404: nothing is deployed for this model; "
                    "calls fail fast until the next recheck"
                ),
                "observed_at": observed,
                "age_s": 12.5,
                "last_answered_at": None,
                "last_answered_age_s": None,
                "recheck_in_s": NOT_SERVING_RECHECK_S - 12.5,
            }
        ]

    def test_the_freshness_window_is_the_shortest_chat_llm_scale_down(self):
        assert SERVING_FRESHNESS_S == float(
            min(
                INFERENCE_SERVICE_SPECS[name].scaledown_window
                for name in ("vllm_llm_student", "vllm_llm_teacher")
            )
        )

    def test_an_answer_is_serving_up_to_the_freshness_window(
        self, availability, clocks
    ):
        availability.answered(CLASSIFICATION)
        answered = _iso(clocks)
        clocks.advance(SERVING_FRESHNESS_S)

        assert availability.snapshot() == [
            {
                "endpoint": ROUTER,
                "model": CLASSIFICATION.model,
                "route": "pro",
                "state": "serving",
                "upstream_status": None,
                "failure": None,
                "reason": "answered its last call",
                "observed_at": answered,
                "age_s": SERVING_FRESHNESS_S,
                "last_answered_at": answered,
                "last_answered_age_s": SERVING_FRESHNESS_S,
                "recheck_in_s": None,
            }
        ]

    def test_an_answer_older_than_the_window_is_idle_with_its_age(
        self, availability, clocks
    ):
        availability.answered(CLASSIFICATION)
        answered = _iso(clocks)
        sixteen_hours = 16 * 3600.0
        clocks.advance(sixteen_hours)

        assert availability.snapshot() == [
            {
                "endpoint": ROUTER,
                "model": CLASSIFICATION.model,
                "route": "pro",
                "state": "idle",
                "upstream_status": None,
                "failure": None,
                "reason": (
                    "answered its last call 57600s ago and has not been called "
                    "since; whether it still serves is unknown until the next call"
                ),
                "observed_at": answered,
                "age_s": sixteen_hours,
                "last_answered_at": answered,
                "last_answered_age_s": sixteen_hours,
                "recheck_in_s": None,
            }
        ]

    def test_an_idle_endpoint_is_serving_again_once_it_answers(
        self, availability, clocks
    ):
        availability.answered(CLASSIFICATION)
        clocks.advance(SERVING_FRESHNESS_S + 1.0)
        assert availability.snapshot()[0]["state"] == "idle"

        availability.answered(CLASSIFICATION)

        entry = availability.snapshot()[0]
        assert (entry["state"], entry["age_s"], entry["last_answered_at"]) == (
            "serving",
            0.0,
            _iso(clocks),
        )

    def test_an_idle_endpoint_is_still_called(self, availability, clocks):
        availability.answered(CLASSIFICATION)
        clocks.advance(SERVING_FRESHNESS_S + 1.0)

        assert availability.admit(CLASSIFICATION) is None
        assert availability.refusal(CLASSIFICATION) is None

    def test_a_failure_keeps_when_the_endpoint_last_answered(
        self, availability, clocks
    ):
        availability.answered(CLASSIFICATION)
        answered = _iso(clocks)
        clocks.advance(40.0)
        availability.failed(CLASSIFICATION, status=503, failure="Timeout")
        failed = _iso(clocks)
        clocks.advance(SERVING_FRESHNESS_S + 60.0)

        entry = availability.snapshot()[0]
        assert (
            entry["state"],
            entry["observed_at"],
            entry["age_s"],
            entry["last_answered_at"],
            entry["last_answered_age_s"],
        ) == (
            "failing",
            failed,
            SERVING_FRESHNESS_S + 60.0,
            answered,
            SERVING_FRESHNESS_S + 100.0,
        )

    def test_concurrent_outcomes_leave_one_consistent_entry(self, clocks):
        """Answers and failures racing on one endpoint never tear the entry:
        the last-answered time is always the answer's, whichever lands last."""
        availability = LMEndpointAvailability(
            clock=clocks.monotonic, wall_clock=clocks.wall
        )
        workers = 16
        barrier = threading.Barrier(workers)

        def outcome(index: int) -> None:
            barrier.wait()
            for _ in range(200):
                if index % 2:
                    availability.answered(CLASSIFICATION)
                else:
                    availability.failed(CLASSIFICATION, status=503, failure="X")
                availability.snapshot()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(outcome, i) for i in range(workers)]:
                future.result()

        (entry,) = availability.snapshot()
        assert entry["state"] in ("serving", "failing")
        assert (entry["last_answered_at"], entry["last_answered_age_s"]) == (
            _iso(clocks),
            0.0,
        )

    def test_credentials_in_the_address_never_reach_the_snapshot(self, availability):
        availability.not_serving(
            LMEndpoint(api_base="https://user:secret@llm.example:8443/v1", model="m"),
            404,
        )

        assert availability.snapshot()[0]["endpoint"] == "https://llm.example:8443/v1"
        with pytest.raises(LMEndpointNotServing) as raised:
            availability.admit(
                LMEndpoint(
                    api_base="https://user:secret@llm.example:8443/v1", model="m"
                )
            )
        assert "secret" not in str(raised.value)

    def test_a_failing_endpoint_names_its_status(self, availability):
        availability.failed(
            CLASSIFICATION, status=503, failure="ServiceUnavailableError"
        )

        entry = availability.snapshot()[0]
        assert (entry["state"], entry["upstream_status"], entry["reason"]) == (
            "failing",
            503,
            "its last call failed: ServiceUnavailableError (HTTP 503)",
        )

    def test_tracking_is_bounded_least_recent_first_out(self, availability):
        endpoints = [
            LMEndpoint(api_base=f"http://llm-{i}:8000/v1", model="m")
            for i in range(MAX_TRACKED_ENDPOINTS + 1)
        ]
        for endpoint in endpoints:
            availability.answered(endpoint)

        assert [e["endpoint"] for e in availability.snapshot()] == [
            endpoint.api_base for endpoint in endpoints[1:]
        ]

    def test_clear_forgets_every_endpoint(self, availability):
        availability.not_serving(CLASSIFICATION, 404)

        availability.clear()

        assert availability.snapshot() == []
        assert availability.admit(CLASSIFICATION) is None


class TestNotServingCause:
    def test_found_on_the_error_itself_and_through_its_causes(self, availability):
        error = availability.not_serving(CLASSIFICATION, 404)
        try:
            try:
                raise error
            except LMEndpointNotServing as inner:
                raise RuntimeError("routed call failed") from inner
        except RuntimeError as outer:
            wrapped = outer

        assert not_serving_cause(error) is error
        assert not_serving_cause(wrapped) is error

    def test_an_unrelated_failure_has_none(self):
        try:
            try:
                raise ValueError("bad body")
            except ValueError as inner:
                raise RuntimeError("call failed") from inner
        except RuntimeError as outer:
            assert not_serving_cause(outer) is None

    def test_a_context_that_was_handled_is_not_a_cause(self, availability):
        """Only an explicit ``raise ... from`` attributes a failure to the LM."""
        try:
            try:
                raise availability.not_serving(CLASSIFICATION, 404)
            except LMEndpointNotServing:
                raise KeyError("unrelated")
        except KeyError as outer:
            assert not_serving_cause(outer) is None


class TestConcurrentRecheck:
    """Many callers reaching an endpoint whose window has passed at once."""

    THREADS = 16

    def _race(self, availability):
        barrier = threading.Barrier(self.THREADS)

        def call(_):
            barrier.wait(timeout=10)
            try:
                return ("probe", availability.admit(CLASSIFICATION))
            except LMEndpointNotServing as error:
                return ("failed_fast", error.failed_fast)

        with ThreadPoolExecutor(self.THREADS) as pool:
            return list(pool.map(call, range(self.THREADS)))

    def test_exactly_one_caller_rechecks_and_the_rest_fail_fast(
        self, availability, clocks
    ):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)

        outcomes = self._race(availability)

        probes = [token for kind, token in outcomes if kind == "probe"]
        assert len(probes) == 1 and probes[0] is not None
        assert sorted(o for o in outcomes if o[0] == "failed_fast") == [
            ("failed_fast", True)
        ] * (self.THREADS - 1)

    def test_once_answered_every_concurrent_caller_is_admitted(
        self, availability, clocks
    ):
        availability.not_serving(CLASSIFICATION, 404)
        clocks.advance(NOT_SERVING_RECHECK_S)
        availability.release(CLASSIFICATION, availability.admit(CLASSIFICATION))
        availability.answered(CLASSIFICATION)

        assert self._race(availability) == [("probe", None)] * self.THREADS

    def test_snapshots_taken_while_outcomes_land_are_whole(self, availability):
        """/health reads while requests record: every snapshot lists whole
        entries and never trips over a mapping changing size under it."""
        endpoints = [
            LMEndpoint(api_base=f"http://llm-{i}:8000/v1", model="m")
            for i in range(MAX_TRACKED_ENDPOINTS * 2)
        ]
        writers, readers = 8, 4
        barrier = threading.Barrier(writers + readers)

        def write(offset):
            barrier.wait(timeout=10)
            for index, endpoint in enumerate(endpoints[offset::writers]):
                if index % 2:
                    availability.not_serving(endpoint, 404)
                else:
                    availability.answered(endpoint)
            return set()

        def read():
            barrier.wait(timeout=10)
            seen = set()
            for _ in range(200):
                snapshot = availability.snapshot()
                assert len(snapshot) <= MAX_TRACKED_ENDPOINTS
                seen.update((e["state"], e["upstream_status"]) for e in snapshot)
            return seen

        with ThreadPoolExecutor(writers + readers) as pool:
            futures = [pool.submit(write, i) for i in range(writers)] + [
                pool.submit(read) for _ in range(readers)
            ]
            observed = set().union(*(future.result() for future in futures))

        assert observed <= {("serving", None), ("not_serving", 404)}
        assert len(availability.snapshot()) == MAX_TRACKED_ENDPOINTS
