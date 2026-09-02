"""Telemetry-provider gate decision for the metrics dashboard.

Kept out of the tab module so it is importable (and testable) without
executing Streamlit's UI body.

The metrics tab renders nothing when the telemetry store does not answer, so
the gate distinguishes a store that is misconfigured from one that was merely
slow. Only the former is a verdict; the latter is a statement about the probe,
and caching it blanks the tab for the cache lifetime after the store has
already recovered.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

# Conditions that describe the moment rather than the configuration: the same
# call can succeed on the next rerun, so their outcome must not be cached.
# Classified by exception TYPE -- a message substring match misroutes a
# permanent error whose text happens to mention a timeout.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.TransportError,
    OSError,
)


@dataclass(frozen=True)
class TelemetryProbe:
    """Outcome of asking the telemetry store for one span.

    ``reachable`` is whether it answered at all. ``transient`` is only
    meaningful when it did not, and marks a failure that may resolve without
    any change to the deployment. ``timed_out`` narrows that further: the
    store was reached but was too slow, which is the one failure where
    rendering the body anyway is better than blanking it.
    """

    reachable: bool
    transient: bool = False
    timed_out: bool = False
    detail: str = ""


@dataclass(frozen=True)
class TelemetryDecision:
    render: bool
    error: str = ""
    # Shown above a body that IS rendered: the store did not confirm itself
    # in time, so the figures below may be stale or partial.
    caveat: str = ""
    cacheable: bool = True

    @property
    def message(self) -> str:
        """The single user-facing reason, whichever branch produced it.

        Call sites interpolate one string into a warning. Reading ``error``
        directly renders a bare marker with no cause whenever the verdict
        populated ``caveat`` instead, so sites take this.
        """
        return self.error or self.caveat


# Bounds the ``__cause__`` walk below; chains are short and may be cyclic.
_MAX_CAUSE_DEPTH = 10


def classify_telemetry_probe(
    exc: BaseException | None, *, timeout_s: float
) -> TelemetryProbe:
    """Turn a probe outcome into a reachability verdict.

    ``exc`` is ``None`` when the store answered.
    """
    if exc is None:
        return TelemetryProbe(reachable=True)

    # Wrappers re-raise a busy store as their own error type (the routing
    # evaluator turns a timeout into RuntimeError) while chaining the cause.
    # Classifying only the outermost type made every transient read as a
    # configuration fault, so the whole verdict below was inert wherever a
    # wrapper sat in the path. The chain is the structural signal; the
    # message text is not.
    causes: list[BaseException] = []
    cursor: BaseException | None = exc
    while cursor is not None and len(causes) < _MAX_CAUSE_DEPTH:
        causes.append(cursor)
        cursor = cursor.__cause__
    exc = next((c for c in causes if isinstance(c, _TRANSIENT_ERRORS)), exc)

    name = type(exc).__name__
    if isinstance(exc, _TRANSIENT_ERRORS):
        timed_out = isinstance(
            exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)
        )
        detail = (
            f"{name}: did not answer within {timeout_s}s"
            if timed_out
            else f"{name}: {exc}"
        )
        return TelemetryProbe(
            reachable=False, transient=True, timed_out=timed_out, detail=detail
        )

    return TelemetryProbe(reachable=False, transient=False, detail=f"{name}: {exc}")


def decide_telemetry_gate(probe: TelemetryProbe) -> TelemetryDecision:
    """Whether to render the metrics body, and whether the answer may be cached."""
    if probe.reachable:
        return TelemetryDecision(render=True)

    if probe.timed_out:
        # Fail OPEN on slowness ONLY. The probe budget has to fit inside a render
        # to keep the page responsive, and the store is measurably slower
        # than that under load, so treating a timeout as a verdict blanks the
        # tab in exactly the conditions an operator wants to inspect it. A
        # timeout says the store did not answer in time, not that it is
        # unusable; the body's own calls report their own failures.
        return TelemetryDecision(
            render=True,
            caveat=(
                f"The telemetry store {probe.detail}. Some figures may be "
                "missing or stale."
            ),
            cacheable=False,
        )

    if probe.transient:
        # Reached-and-refused, e.g. connection refused: the store is down
        # rather than slow, so a rendered body would be a page of failures.
        return TelemetryDecision(
            render=False,
            error=(
                f"The telemetry store {probe.detail}. It may be starting "
                "rather than misconfigured -- rerun to try again."
            ),
            cacheable=False,
        )

    return TelemetryDecision(
        render=False,
        error=(
            f"The telemetry store rejected the query ({probe.detail}). Check "
            "the telemetry configuration for this tenant."
        ),
    )
