"""The metrics dashboard hides its whole body behind a telemetry probe, so the
probe must say WHICH condition it hit and must not cache a transient one.

Collapsing timeout, outage and misconfiguration onto a single "provider is not
available" tells the user a definite thing the probe has no basis for, and
caching that negative blanks the tab for the cache TTL after the pressure that
caused it has passed. Phoenix is measurably slow under load here -- a dataset
read timed at 0.02s idle and over 30s while spans were being walked -- so the
timeout branch is the common one, not the rare one.
"""

from __future__ import annotations

import asyncio

import httpx

from cogniverse_dashboard.telemetry_gate import (
    TelemetryProbe,
    classify_telemetry_probe,
    decide_telemetry_gate,
)

TIMEOUT_S = 5.0


def test_answering_store_is_reachable_and_renders():
    probe = classify_telemetry_probe(None, timeout_s=TIMEOUT_S)
    assert (probe.reachable, probe.transient, probe.detail) == (True, False, "")

    decision = decide_telemetry_gate(probe)
    assert (decision.render, decision.error, decision.cacheable) == (True, "", True)


def test_timeout_is_transient_and_never_cached():
    probe = classify_telemetry_probe(asyncio.TimeoutError(), timeout_s=TIMEOUT_S)
    assert (probe.reachable, probe.transient) == (False, True)
    assert "5.0s" in probe.detail and "TimeoutError" in probe.detail

    decision = decide_telemetry_gate(probe)
    assert decision.render is False
    # Caching a timeout keeps the tab blank for the TTL after the store has
    # recovered, so the next rerun must probe again.
    assert decision.cacheable is False
    assert "did not answer within 5.0s" in decision.error
    assert "not available" not in decision.error


def test_connection_error_is_transient_and_names_the_cause():
    probe = classify_telemetry_probe(
        httpx.ConnectError("connection refused"), timeout_s=TIMEOUT_S
    )
    assert (probe.reachable, probe.transient) == (False, True)

    decision = decide_telemetry_gate(probe)
    assert decision.cacheable is False
    assert "ConnectError" in decision.error


def test_configuration_error_is_a_verdict_and_is_cached():
    probe = classify_telemetry_probe(
        ValueError("PHOENIX_ENDPOINT is not configured"), timeout_s=TIMEOUT_S
    )
    assert (probe.reachable, probe.transient) == (False, False)

    decision = decide_telemetry_gate(probe)
    assert decision.render is False
    # A misconfiguration will not fix itself between reruns, so re-probing it
    # every interaction buys nothing.
    assert decision.cacheable is True
    assert "ValueError" in decision.error
    assert "PHOENIX_ENDPOINT is not configured" in decision.error


def test_transient_and_permanent_errors_report_different_causes():
    """The two branches must be distinguishable to a reader of the UI."""
    transient = decide_telemetry_gate(
        classify_telemetry_probe(asyncio.TimeoutError(), timeout_s=TIMEOUT_S)
    )
    permanent = decide_telemetry_gate(
        classify_telemetry_probe(ValueError("bad config"), timeout_s=TIMEOUT_S)
    )
    assert transient.error != permanent.error
    assert (transient.cacheable, permanent.cacheable) == (False, True)


def test_classification_is_by_exception_type_not_message():
    """A permanent error whose message mentions a timeout stays permanent.

    Classifying on message substrings is how an outage whose URL or body
    happens to carry the phrase gets misrouted; the type is the signal.
    """
    probe = classify_telemetry_probe(
        ValueError("connection timed out while parsing config"), timeout_s=TIMEOUT_S
    )
    assert probe.transient is False
    assert decide_telemetry_gate(probe).cacheable is True


def test_probe_detail_carries_the_exception_type_for_every_branch():
    for exc in (
        asyncio.TimeoutError(),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        ValueError("bad"),
        RuntimeError("boom"),
    ):
        probe = classify_telemetry_probe(exc, timeout_s=TIMEOUT_S)
        assert type(exc).__name__ in probe.detail, probe

    assert TelemetryProbe(reachable=True).detail == ""


def test_probe_budget_cannot_outlast_its_own_cache_window():
    """A probe that runs longer than it is cached for is not rate limited.

    The probe blocks the render, and Streamlit executes every tab body on
    every rerun, so an unbounded probe rate makes a slow store block every
    interaction. The cache is the only rate limit, and it only limits
    anything while the budget fits inside the window.
    """
    from cogniverse_dashboard.tabs import optimization

    assert optimization._TELEMETRY_PROBE_TIMEOUT_S < optimization._PROBE_CACHE_TTL_S, (
        optimization._TELEMETRY_PROBE_TIMEOUT_S,
        optimization._PROBE_CACHE_TTL_S,
    )
    # Short enough that a blank tab recovers within a couple of interactions
    # rather than the minute that motivated classifying the probe at all.
    assert optimization._PROBE_CACHE_TTL_S <= 10.0


def test_metrics_tab_keeps_a_transient_probe_cached(monkeypatch):
    """Transient outcomes stay cached, so the retry rate stays bounded.

    An earlier version evicted the entry on every transient branch. That made
    the cache inert: Streamlit reruns on each interaction and renders every
    tab body, so each rerun paid a fresh probe and a slow store blocked the
    whole page for the probe budget -- worse in exactly the condition the
    classification was written for. A short TTL recovers quickly without
    giving up the rate limit.
    """
    import streamlit as st

    from cogniverse_dashboard.tabs import optimization

    cleared: list[str] = []

    class _FakeProbe:
        def __init__(self, probe: TelemetryProbe) -> None:
            self._probe = probe

        def __call__(self, tenant_id: str) -> TelemetryProbe:
            return self._probe

        def clear(self) -> None:
            cleared.append("cleared")

    warnings: list[str] = []
    monkeypatch.setattr(st, "warning", lambda msg: warnings.append(str(msg)))
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setitem(st.session_state, "current_tenant", "acme")

    transient = classify_telemetry_probe(asyncio.TimeoutError(), timeout_s=TIMEOUT_S)
    monkeypatch.setattr(optimization, "_probe_telemetry", _FakeProbe(transient))
    optimization._render_metrics_dashboard_tab()
    assert cleared == [], "a transient probe must stay cached to bound the retry rate"
    assert len(warnings) == 1 and "rerun to try again" in warnings[0], warnings

    # A configuration verdict will not change between reruns, so it stays.
    cleared.clear()
    warnings.clear()
    permanent = classify_telemetry_probe(ValueError("no endpoint"), timeout_s=TIMEOUT_S)
    monkeypatch.setattr(optimization, "_probe_telemetry", _FakeProbe(permanent))
    optimization._render_metrics_dashboard_tab()
    assert cleared == [], "a configuration verdict must stay cached"
    assert len(warnings) == 1 and "no endpoint" in warnings[0], warnings
