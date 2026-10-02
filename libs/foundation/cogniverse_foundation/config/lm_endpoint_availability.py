"""What this process last observed of each LM endpoint it called.

An endpoint that answers HTTP 404 has nothing deployed for the model it was
asked for: an undeployed Modal app answers ``modal-http: invalid function
call`` at Modal's edge, and the semantic router passes that status through
(``x-vsr-response-path: upstream``). That answer does not change from one
request to the next, so after it every call to the same endpoint, model and
route fails fast with ``LMEndpointNotServing`` instead of paying the round
trip again. Once ``NOT_SERVING_RECHECK_S`` has passed, one call is let through
as the recheck; its outcome decides for the rest.

Only a 404 fails fast. A scaled-to-zero endpoint is still deployed: its first
request waits out the cold start and answers, so a timeout, a 5xx or a refused
connection is recorded as ``failing`` and never stops the calls that follow.

The state is per process and never probed in the background: any request to a
deployed Modal endpoint, a probe included, boots a GPU container.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from cogniverse_foundation.telemetry.span_contract import (
    LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE,
    LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE,
    LLM_ENDPOINT_STATE_ATTRIBUTE,
)

# The status an endpoint with nothing deployed for the requested model answers.
NOT_SERVING_STATUS = 404

# How long calls fail fast before one call rechecks the endpoint: a redeployed
# endpoint is used again within this window, and an undeployed one costs one
# 404 round trip per window per process instead of one per request.
NOT_SERVING_RECHECK_S = 30.0

# The slowest 404 measured from an undeployed Modal app: 0.98s through the
# semantic-router Envoy, 0.73s direct (2026-10-01, k3d host). A recheck still
# unanswered at five times that is a cold start, not an undeployed endpoint,
# and the calls queued behind it are let through instead of failing fast.
_MEASURED_NOT_SERVING_ANSWER_MAX_S = 0.98
PROBE_VERDICT_S = float(math.ceil(5 * _MEASURED_NOT_SERVING_ANSWER_MAX_S))

# Endpoints are configuration (an address, a model alias, a tier), so a
# process tracks a handful; the bound only stops an unbounded key space.
MAX_TRACKED_ENDPOINTS = 64

SERVING = "serving"
NOT_SERVING = "not_serving"
FAILING = "failing"


def display_endpoint(api_base: Optional[str]) -> Optional[str]:
    """``api_base`` without any credentials embedded in its authority."""
    if not api_base:
        return api_base
    parts = urlsplit(api_base)
    if parts.username is None and parts.password is None:
        return api_base
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit(parts._replace(netloc=netloc))


@dataclass(frozen=True)
class LMEndpoint:
    """Where an LM call goes: the address, the model it names there, and the
    route the address resolves the model on (the router tier), if any."""

    api_base: Optional[str]
    model: str
    route: Optional[str] = None


class LMEndpointNotServing(RuntimeError):
    """The LM endpoint answered 404: nothing is deployed there for the model.

    ``failed_fast`` is True when the call was refused without reaching the
    endpoint, because an earlier call already got the 404.
    ``recheck_in_s`` is how long until one call rechecks it. ``status_code``
    is the status the endpoint answered, read like a provider error's.
    """

    def __init__(
        self,
        *,
        endpoint: LMEndpoint,
        status: int,
        failed_fast: bool,
        recheck_in_s: float,
    ) -> None:
        fast = "; failing fast," if failed_fast else ";"
        super().__init__(
            f"the LM endpoint is not serving: "
            f"endpoint={display_endpoint(endpoint.api_base)} "
            f"model={endpoint.model} route={endpoint.route} "
            f"answered HTTP {status}{fast} next recheck in {recheck_in_s:.1f}s"
        )
        self.endpoint = endpoint
        self.status = status
        self.status_code = status
        self.failed_fast = failed_fast
        self.recheck_in_s = recheck_in_s

    def span_attributes(self) -> dict[str, Any]:
        """What the span of the call this refused or failed records."""
        return {
            LLM_ENDPOINT_STATE_ATTRIBUTE: NOT_SERVING,
            LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE: self.failed_fast,
            LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE: self.recheck_in_s,
        }


def not_serving_cause(exc: BaseException) -> Optional[LMEndpointNotServing]:
    """The ``LMEndpointNotServing`` ``exc`` is or was explicitly raised from.

    Follows ``__cause__`` only: a 404 that was handled before an unrelated
    failure is that failure's context, not its cause.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, LMEndpointNotServing):
            return current
        seen.add(id(current))
        current = current.__cause__
    return None


@dataclass
class _Observation:
    state: str
    upstream_status: Optional[int]
    failure: Optional[str]
    observed_at: float
    recheck_at: Optional[float] = None
    probe: Optional[object] = None
    probe_started_at: Optional[float] = None


class LMEndpointAvailability:
    """The last outcome of each endpoint, and the fast failure a 404 starts.

    Thread-safe: synchronous LM calls run on executor threads and
    asynchronous ones on the event loop, and both consult one instance.
    """

    def __init__(
        self,
        *,
        recheck_after_s: float = NOT_SERVING_RECHECK_S,
        probe_verdict_s: float = PROBE_VERDICT_S,
        max_endpoints: int = MAX_TRACKED_ENDPOINTS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.recheck_after_s = recheck_after_s
        self.probe_verdict_s = probe_verdict_s
        self.max_endpoints = max_endpoints
        self._clock = clock
        self._wall_clock = wall_clock
        self._observations: OrderedDict[LMEndpoint, _Observation] = OrderedDict()
        self._lock = threading.Lock()

    def _record(self, endpoint: LMEndpoint, observation: _Observation) -> None:
        self._observations[endpoint] = observation
        self._observations.move_to_end(endpoint)
        while len(self._observations) > self.max_endpoints:
            self._observations.popitem(last=False)

    def _refusal(
        self, endpoint: LMEndpoint, seen: Optional[_Observation], now: float
    ) -> Optional[LMEndpointNotServing]:
        """The fast failure a call to ``endpoint`` gets now, if any. Caller
        holds the lock."""
        if seen is None or seen.state != NOT_SERVING:
            return None
        if seen.probe is not None:
            if now - seen.probe_started_at >= self.probe_verdict_s:
                return None
        elif now >= seen.recheck_at:
            return None
        return LMEndpointNotServing(
            endpoint=endpoint,
            status=seen.upstream_status,
            failed_fast=True,
            recheck_in_s=max(seen.recheck_at - now, 0.0),
        )

    def refusal(self, endpoint: LMEndpoint) -> Optional[LMEndpointNotServing]:
        """The fast failure a call to ``endpoint`` would get now, without
        admitting one: for a caller deciding whether to prepare the call at
        all. ``None`` once a recheck is due, so the call itself rechecks."""
        with self._lock:
            return self._refusal(
                endpoint, self._observations.get(endpoint), self._clock()
            )

    def admit(self, endpoint: LMEndpoint) -> Optional[object]:
        """Let a call to ``endpoint`` go out, or raise ``LMEndpointNotServing``.

        Returns a recheck token when this call is the one rechecking a
        not-serving endpoint, ``None`` otherwise. A caller holding a token
        passes it to ``release`` when the call ends, whatever its outcome.
        """
        with self._lock:
            seen = self._observations.get(endpoint)
            now = self._clock()
            refused = self._refusal(endpoint, seen, now)
            if refused is not None:
                raise refused
            if seen is None or seen.state != NOT_SERVING or seen.probe is not None:
                return None
            seen.probe = object()
            seen.probe_started_at = now
            return seen.probe

    def release(self, endpoint: LMEndpoint, probe: Optional[object]) -> None:
        """End the recheck ``probe`` started; the next call rechecks when the
        call ended without recording an outcome."""
        if probe is None:
            return
        with self._lock:
            seen = self._observations.get(endpoint)
            if seen is not None and seen.probe is probe:
                seen.probe = None
                seen.probe_started_at = None

    def answered(self, endpoint: LMEndpoint) -> None:
        with self._lock:
            self._record(
                endpoint,
                _Observation(
                    state=SERVING,
                    upstream_status=None,
                    failure=None,
                    observed_at=self._wall_clock(),
                ),
            )

    def not_serving(self, endpoint: LMEndpoint, status: int) -> LMEndpointNotServing:
        """Record the 404 and return the error the call that got it raises."""
        with self._lock:
            self._record(
                endpoint,
                _Observation(
                    state=NOT_SERVING,
                    upstream_status=status,
                    failure=None,
                    observed_at=self._wall_clock(),
                    recheck_at=self._clock() + self.recheck_after_s,
                ),
            )
        return LMEndpointNotServing(
            endpoint=endpoint,
            status=status,
            failed_fast=False,
            recheck_in_s=self.recheck_after_s,
        )

    def failed(
        self, endpoint: LMEndpoint, *, status: Optional[int], failure: str
    ) -> None:
        """Record a call the endpoint did not answer (a timeout, a refused
        connection, a 5xx). It never makes the calls that follow fail fast."""
        with self._lock:
            self._record(
                endpoint,
                _Observation(
                    state=FAILING,
                    upstream_status=status,
                    failure=failure,
                    observed_at=self._wall_clock(),
                ),
            )

    def snapshot(self) -> list[dict[str, Any]]:
        """Every tracked endpoint's last outcome, least recently observed first."""
        with self._lock:
            now = self._clock()
            items = list(self._observations.items())
            return [_describe(endpoint, seen, now) for endpoint, seen in items]

    def clear(self) -> None:
        with self._lock:
            self._observations.clear()


def _reason(seen: _Observation) -> str:
    if seen.state == NOT_SERVING:
        return (
            f"answered HTTP {seen.upstream_status}: nothing is deployed for this "
            "model; calls fail fast until the next recheck"
        )
    if seen.state == FAILING:
        status = f" (HTTP {seen.upstream_status})" if seen.upstream_status else ""
        return f"its last call failed: {seen.failure}{status}"
    return "answered its last call"


def _describe(endpoint: LMEndpoint, seen: _Observation, now: float) -> dict[str, Any]:
    return {
        "endpoint": display_endpoint(endpoint.api_base),
        "model": endpoint.model,
        "route": endpoint.route,
        "state": seen.state,
        "upstream_status": seen.upstream_status,
        "failure": seen.failure,
        "reason": _reason(seen),
        "observed_at": datetime.fromtimestamp(
            seen.observed_at, tz=timezone.utc
        ).isoformat(),
        "recheck_in_s": (
            max(seen.recheck_at - now, 0.0) if seen.state == NOT_SERVING else None
        ),
    }


_PROCESS_AVAILABILITY = LMEndpointAvailability()


def lm_endpoint_availability() -> LMEndpointAvailability:
    """The availability every LM in this process consults."""
    return _PROCESS_AVAILABILITY


__all__ = [
    "FAILING",
    "MAX_TRACKED_ENDPOINTS",
    "NOT_SERVING",
    "NOT_SERVING_RECHECK_S",
    "NOT_SERVING_STATUS",
    "PROBE_VERDICT_S",
    "SERVING",
    "LMEndpoint",
    "LMEndpointAvailability",
    "LMEndpointNotServing",
    "display_endpoint",
    "lm_endpoint_availability",
    "not_serving_cause",
]
