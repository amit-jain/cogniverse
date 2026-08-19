"""Fail any test during which a telemetry-loss WARNING was logged.

Serving telemetry is best effort on the request path: a missing telemetry
manager, a failed enqueue, or an enabled manager that could not build a
tracer logs a WARNING and the request continues. The test environment is
managed, so nothing may be lost there: a test that produced one of these
warnings fails, unless the warning is the test's own subject — then it
carries ``@pytest.mark.expects_telemetry_loss_warning`` and asserts the
exact message itself.
"""

from __future__ import annotations

import logging

import pytest

LOSS_MARKERS: tuple[str, ...] = (
    "has no telemetry_manager",
    "Failed to emit ",
    "No tracer for span",
)
MARKER = "expects_telemetry_loss_warning"


class _LossCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if any(marker in message for marker in LOSS_MARKERS):
            self.records.append(f"{record.name}: {message}")


@pytest.fixture(autouse=True)
def _telemetry_loss_guard(request: pytest.FixtureRequest):
    handler = _LossCollector()
    root = logging.getLogger()
    root.addHandler(handler)
    request.node._telemetry_loss_records = handler.records
    try:
        yield
    finally:
        root.removeHandler(handler)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.passed:
        return
    records = getattr(item, "_telemetry_loss_records", None) or []
    if not records or item.get_closest_marker(MARKER):
        return
    report.outcome = "failed"
    report.longrepr = (
        "telemetry loss during test — a best-effort serving path dropped a span:\n  - "
        + "\n  - ".join(records)
        + f"\nAttach a telemetry manager in the fixture. Mark @pytest.mark.{MARKER} "
        "only when the warning is the test's subject and the test asserts its exact text."
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}: the test's subject is a telemetry-loss WARNING; the loss guard does not fail it",
    )
