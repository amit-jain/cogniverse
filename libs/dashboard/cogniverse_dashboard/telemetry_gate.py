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
    any change to the deployment.
    """

    reachable: bool
    transient: bool = False
    detail: str = ""


@dataclass(frozen=True)
class TelemetryDecision:
    render: bool
    error: str = ""
    cacheable: bool = True


def classify_telemetry_probe(
    exc: BaseException | None, *, timeout_s: float
) -> TelemetryProbe:
    """Turn a probe outcome into a reachability verdict.

    ``exc`` is ``None`` when the store answered.
    """
    if exc is None:
        return TelemetryProbe(reachable=True)

    name = type(exc).__name__
    if isinstance(exc, _TRANSIENT_ERRORS):
        detail = (
            f"{name}: did not answer within {timeout_s}s"
            if isinstance(
                exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)
            )
            else f"{name}: {exc}"
        )
        return TelemetryProbe(reachable=False, transient=True, detail=detail)

    return TelemetryProbe(reachable=False, transient=False, detail=f"{name}: {exc}")


def decide_telemetry_gate(probe: TelemetryProbe) -> TelemetryDecision:
    """Whether to render the metrics body, and whether the answer may be cached."""
    if probe.reachable:
        return TelemetryDecision(render=True)

    if probe.transient:
        return TelemetryDecision(
            render=False,
            error=(
                f"The telemetry store {probe.detail}. It may be busy rather "
                "than down -- rerun to try again."
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
