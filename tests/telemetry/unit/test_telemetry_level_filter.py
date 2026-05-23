"""Verify TelemetryConfig.level actually filters spans by component.

Two layers tested:
1. ``TelemetryConfig.should_instrument_component`` — pure filter logic.
2. ``TelemetryManager.span()`` — when the filter says no, yields a
   NoOpSpan unconditionally (no tracer lookup, no OTel call).
"""

from unittest.mock import MagicMock

import pytest

from cogniverse_foundation.telemetry.config import (
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_foundation.telemetry.manager import NoOpSpan, TelemetryManager

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Filter logic — pure, no I/O
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level,component,expected",
    [
        # DISABLED — nothing emits
        (TelemetryLevel.DISABLED, "search_service", False),
        (TelemetryLevel.DISABLED, "backend", False),
        (TelemetryLevel.DISABLED, "encoder", False),
        (TelemetryLevel.DISABLED, "pipeline", False),
        (TelemetryLevel.DISABLED, "agents", False),
        # BASIC — only search_service
        (TelemetryLevel.BASIC, "search_service", True),
        (TelemetryLevel.BASIC, "backend", False),
        (TelemetryLevel.BASIC, "encoder", False),
        (TelemetryLevel.BASIC, "pipeline", False),
        (TelemetryLevel.BASIC, "agents", False),
        # DETAILED — adds backend + encoder
        (TelemetryLevel.DETAILED, "search_service", True),
        (TelemetryLevel.DETAILED, "backend", True),
        (TelemetryLevel.DETAILED, "encoder", True),
        (TelemetryLevel.DETAILED, "pipeline", False),
        (TelemetryLevel.DETAILED, "agents", False),
        # VERBOSE — adds pipeline + agents (everything)
        (TelemetryLevel.VERBOSE, "search_service", True),
        (TelemetryLevel.VERBOSE, "backend", True),
        (TelemetryLevel.VERBOSE, "encoder", True),
        (TelemetryLevel.VERBOSE, "pipeline", True),
        (TelemetryLevel.VERBOSE, "agents", True),
    ],
)
def test_should_instrument_component(level, component, expected):
    """Locks the documented level→component admission matrix.

    Any drift between the docstring and the implementation fails here.
    """
    cfg = TelemetryConfig(enabled=True, level=level)
    assert cfg.should_instrument_component(component) is expected, (
        f"level={level.value} component={component!r} expected "
        f"{expected} but got {not expected}"
    )


def test_enabled_false_filters_everything():
    """``enabled=False`` short-circuits even VERBOSE — operators can
    kill telemetry without changing level."""
    cfg = TelemetryConfig(enabled=False, level=TelemetryLevel.VERBOSE)
    for comp in ("search_service", "backend", "encoder", "pipeline", "agents"):
        assert cfg.should_instrument_component(comp) is False


def test_unknown_component_defaults_to_detailed_tier():
    """Unknown component name fails open — admits on DETAILED+VERBOSE,
    drops on BASIC+DISABLED. So a typo in a new instrumentation site
    doesn't silently disappear telemetry at every level."""
    for level, expected in [
        (TelemetryLevel.DISABLED, False),
        (TelemetryLevel.BASIC, False),
        (TelemetryLevel.DETAILED, True),
        (TelemetryLevel.VERBOSE, True),
    ]:
        cfg = TelemetryConfig(enabled=True, level=level)
        assert cfg.should_instrument_component("brand_new_thing") is expected


# ---------------------------------------------------------------------------
# Manager integration — filtered components yield NoOpSpan WITHOUT touching
# the tracer provider
# ---------------------------------------------------------------------------


def test_span_yields_noop_when_filter_says_no():
    """When the level filter rejects the component, ``span()`` MUST
    return a NoOpSpan immediately — no tracer lookup, no OTel call.

    Patches the tracer-lookup helper to a Mock to prove it's never
    invoked when the filter short-circuits.
    """
    cfg = TelemetryConfig(enabled=True, level=TelemetryLevel.BASIC)
    mgr = TelemetryManager(cfg)
    tracer_lookup = MagicMock()
    mgr._get_tracer_for_project = tracer_lookup  # type: ignore[assignment]

    with mgr.span("t", tenant_id="t1", component="agents") as span:
        assert isinstance(span, NoOpSpan)
    tracer_lookup.assert_not_called()


def test_span_consults_tracer_when_filter_says_yes():
    """When the filter admits the component, ``span()`` MUST call the
    tracer-lookup path."""
    cfg = TelemetryConfig(enabled=True, level=TelemetryLevel.BASIC)
    mgr = TelemetryManager(cfg)
    tracer_lookup = MagicMock(return_value=None)
    mgr._get_tracer_for_project = tracer_lookup  # type: ignore[assignment]

    with mgr.span("t", tenant_id="t1", component="search_service") as _:
        pass
    tracer_lookup.assert_called_once()


def test_span_default_component_is_agents():
    """``.span()`` without an explicit component defaults to 'agents' —
    a VERBOSE-only tier. Callers that want their spans emitted at lower
    levels MUST opt in via ``component=``."""
    cfg = TelemetryConfig(enabled=True, level=TelemetryLevel.BASIC)
    mgr = TelemetryManager(cfg)
    mgr._get_tracer_for_project = MagicMock()  # type: ignore[assignment]

    with mgr.span("t", tenant_id="t1") as span:
        assert isinstance(span, NoOpSpan)
    # Verify the lookup was never called — filter short-circuited.
    mgr._get_tracer_for_project.assert_not_called()
