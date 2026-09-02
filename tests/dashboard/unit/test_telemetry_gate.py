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
import time

import httpx
import pytest

from cogniverse_dashboard import telemetry_gate
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

    assert probe.timed_out is True

    decision = decide_telemetry_gate(probe)
    # A timeout renders the body with a caveat rather than blanking the tab:
    # the store is measurably slower under load than any budget that fits
    # inside a render, so failing closed hides the tab exactly when it is
    # wanted. Slowness only -- a refused connection still blocks.
    assert decision.render is True
    assert decision.cacheable is False
    assert decision.error == ""
    assert "did not answer within 5.0s" in decision.caveat
    assert "not available" not in decision.caveat


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


def test_a_busy_store_renders_the_body_with_a_caveat():
    """A slow store must not blank the tab.

    Measured 2026-09-02 during a sweep: Phoenix answered its simplest
    endpoint in 11s, 11s and 37s while sitting at 270m CPU, so it is I/O
    bound on its store, not busy. No probe budget short enough to keep the
    page responsive can succeed against that, and failing closed on a
    timeout means the tab is permanently blank under exactly the load an
    operator most wants to look at it. A timeout is not evidence the store
    is unusable - only that it did not answer inside a render.
    """
    probe = classify_telemetry_probe(asyncio.TimeoutError(), timeout_s=TIMEOUT_S)
    decision = decide_telemetry_gate(probe)
    assert decision.render is True
    assert decision.error == ""
    assert "did not answer" in decision.caveat and str(TIMEOUT_S) in decision.caveat


def test_a_rejected_query_still_blocks_the_body():
    """Fail-open applies to slowness only, never to a verdict.

    A configuration error will not fix itself between reruns, and rendering
    a body whose every call raises would replace one clear message with a
    page of tracebacks.
    """
    probe = classify_telemetry_probe(ValueError("no such project"), timeout_s=TIMEOUT_S)
    decision = decide_telemetry_gate(probe)
    assert decision.render is False
    assert decision.caveat == ""
    assert "rejected the query" in decision.error


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

    transient = classify_telemetry_probe(
        ConnectionError("connection refused"), timeout_s=TIMEOUT_S
    )
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


@pytest.mark.parametrize("tab", ["metrics", "profile"])
def test_a_slow_store_renders_the_body_behind_a_caveat(monkeypatch, tab):
    """A timeout must not blank a tab -- it renders, carrying the warning.

    The store answers its simplest endpoint in 11-36s while spans are being
    walked, so no budget that fits inside a page render can succeed under
    load. Failing closed on slowness hides the tab in exactly the condition
    it exists to report on. Each case raises from the first call past its
    gate, which proves the body was reached rather than skipped; the sentinel
    derives from BaseException so the profile tab's ``except Exception``
    cannot swallow it and turn a skipped body into a passing test.

    Parametrized over both gate sites: covering one left the other free to
    drop the caveat with every test still green.
    """
    import streamlit as st

    from cogniverse_dashboard.tabs import optimization

    class _PastTheGate(BaseException):
        pass

    def _raise(*_a, **_k):
        raise _PastTheGate()

    warnings: list[str] = []
    monkeypatch.setattr(st, "warning", lambda msg: warnings.append(str(msg)))
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setitem(st.session_state, "current_tenant", "acme")

    if tab == "metrics":
        render = optimization._render_metrics_dashboard_tab
        monkeypatch.setattr(st, "columns", _raise)
    else:
        render = optimization._render_profile_selection_tab
        monkeypatch.setattr(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager", _raise
        )

    slow = classify_telemetry_probe(asyncio.TimeoutError(), timeout_s=TIMEOUT_S)
    monkeypatch.setattr(optimization, "_probe_telemetry", lambda tenant_id: slow)

    with pytest.raises(_PastTheGate):
        render()

    assert len(warnings) == 1, warnings
    assert "did not answer within 5.0s" in warnings[0], warnings[0]


def test_every_gate_verdict_carries_exactly_one_message():
    """No branch may emit a bare warning marker with nothing after it.

    Call sites format ``f"warning {decision.error}"``; a verdict that leaves
    both fields empty renders a warning icon and no cause. Fail-open added a
    branch where ``error`` is empty by design, so the invariant is now that
    exactly one of the two is populated.
    """
    verdicts = [
        classify_telemetry_probe(asyncio.TimeoutError(), timeout_s=TIMEOUT_S),
        classify_telemetry_probe(ConnectionError("refused"), timeout_s=TIMEOUT_S),
        classify_telemetry_probe(ValueError("no endpoint"), timeout_s=TIMEOUT_S),
        TelemetryProbe(reachable=True),
    ]
    populated = [
        (bool(d.error), bool(d.caveat))
        for d in (decide_telemetry_gate(p) for p in verdicts)
    ]
    assert populated == [(False, True), (True, False), (True, False), (False, False)]


def test_a_wrapped_timeout_is_still_classified_as_slow():
    """A busy store re-raised as another type must not read as misconfigured.

    The routing evaluator wraps every failure in ``RuntimeError(...) from e``.
    Classifying the outermost type alone sent a timeout down the permanent
    branch -- "check the telemetry configuration" for a store that was merely
    busy, with the fail-open path unreachable behind it.
    """
    wrapped = RuntimeError("Failed to query routing spans: ")
    wrapped.__cause__ = asyncio.TimeoutError()

    probe = classify_telemetry_probe(wrapped, timeout_s=TIMEOUT_S)
    assert (probe.reachable, probe.transient, probe.timed_out) == (False, True, True)
    assert probe.detail == "TimeoutError: did not answer within 5.0s"

    # A wrapper with no transient cause stays a configuration verdict.
    plain = RuntimeError("Failed to query routing spans: bad project")
    plain.__cause__ = ValueError("bad project")
    assert classify_telemetry_probe(plain, timeout_s=TIMEOUT_S).transient is False


def test_the_cause_walk_terminates_on_a_cyclic_chain():
    """A self-referential chain must not hang the render thread."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a

    probe = classify_telemetry_probe(a, timeout_s=TIMEOUT_S)
    assert (probe.reachable, probe.transient) == (False, False)


class TestRenderSpanQueryIsBounded:
    """A span read on Streamlit's render path must not run unbounded.

    Streamlit executes every tab body on every rerun, so one unbounded Phoenix
    read stalls every later tab. Measured 2026-09-02: a healthy read is
    1.22-1.81s, while an abandoned one kept a full scan running server-side and
    took the host disk to 90% utilisation until Phoenix was restarted.
    """

    def test_returns_the_frame_when_the_query_completes(self):
        async def _q():
            return "FRAME"

        out = telemetry_gate.run_render_span_query(_q, timeout_s=5.0)
        assert out.frame == "FRAME"
        assert out.timed_out is False
        assert out.error == ""

    def test_a_slow_query_times_out_instead_of_blocking_the_render(self):
        import asyncio as _a

        async def _q():
            await _a.sleep(30)
            return "NEVER"

        started = time.monotonic()
        out = telemetry_gate.run_render_span_query(_q, timeout_s=0.25)
        elapsed = time.monotonic() - started

        assert out.timed_out is True
        assert out.frame is None
        assert "0.25" in out.error, out.error
        assert elapsed < 5.0, elapsed

    def test_a_failing_query_reports_its_cause_and_is_not_a_timeout(self):
        async def _q():
            raise ValueError("phoenix exploded")

        out = telemetry_gate.run_render_span_query(_q, timeout_s=5.0)
        assert out.timed_out is False
        assert out.frame is None
        assert "ValueError" in out.error and "phoenix exploded" in out.error, out.error

    def test_every_outcome_carries_exactly_one_non_empty_signal(self):
        import asyncio as _a

        async def _ok():
            return "F"

        async def _slow():
            await _a.sleep(30)

        async def _boom():
            raise RuntimeError("x")

        outs = [
            telemetry_gate.run_render_span_query(_ok, timeout_s=1.0),
            telemetry_gate.run_render_span_query(_slow, timeout_s=0.2),
            telemetry_gate.run_render_span_query(_boom, timeout_s=1.0),
        ]
        assert [o.frame is not None for o in outs] == [True, False, False]
        assert [o.error != "" for o in outs] == [False, True, True]

    def test_the_default_budget_clears_the_measured_healthy_latency(self):
        # A budget below the measured healthy latency fails every healthy read.
        assert telemetry_gate.RENDER_SPAN_QUERY_TIMEOUT_S >= 3 * 1.81
