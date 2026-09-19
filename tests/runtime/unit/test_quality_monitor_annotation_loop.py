"""_annotation_loop sidecar and its launch/cancel in one monitor iteration.

The infinite annotation loop had no coverage: these pin that one failing
cycle does not stop the loop (the per-iteration ``except Exception`` swallow),
that the inter-cycle sleep equals ``annotation_interval_minutes * 60`` seconds,
and that the monitor iteration launches the loop as a task and cancels it in
the shutdown finally after ``monitor.run()`` returns.

Every loop here is bounded: the patched cycle raises ``_BreakLoop`` (a
BaseException the loop's ``except Exception`` cannot swallow) to exit cleanly,
so no test can hang on the real ``while True``.
"""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_runtime import quality_monitor_cli as qm

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _BreakLoop(BaseException):
    """Sentinel to break the otherwise-infinite sidecar loop.

    Subclasses BaseException (not Exception) so the loop's ``except Exception``
    swallow does not catch it — the same mechanism by which a real
    asyncio.CancelledError breaks the loop on shutdown.
    """


def _rules_with_interval(minutes: int):
    from cogniverse_agents.routing.config import AutomationRulesConfig, IntervalConfig

    return AutomationRulesConfig(
        intervals=IntervalConfig(annotation_interval_minutes=minutes)
    )


async def _run_bounded_loop(monkeypatch, *, rules, cycle):
    """Patch the loop's collaborators and run _annotation_loop until ``cycle``
    raises _BreakLoop. Returns the list of sleep-delays the loop awaited."""
    sleeps: list = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        await real_sleep(0)  # yield without a real delay

    monkeypatch.setattr(qm, "_load_automation_rules", lambda tenant_id: rules)
    monkeypatch.setattr(qm, "run_annotation_cycle", cycle)
    monkeypatch.setattr(qm.asyncio, "sleep", fake_sleep)

    with pytest.raises(_BreakLoop):
        await qm._annotation_loop("acme:acme", "http://runtime:28000")
    return sleeps


@pytest.mark.asyncio
async def test_failing_cycle_does_not_stop_loop(monkeypatch):
    """A cycle that raises on iteration 1 must not kill the loop: iteration 2
    still runs, and the loop completed iteration 1 through its sleep."""
    calls: list = []

    async def cycle(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("cycle boom on iteration 1")
        raise _BreakLoop  # iteration 2 was reached — break out

    rules = _rules_with_interval(30)
    sleeps = await _run_bounded_loop(monkeypatch, rules=rules, cycle=cycle)

    assert len(calls) == 2  # iteration 2 ran despite iteration 1 raising
    assert sleeps == [
        30 * 60
    ]  # iteration 1 finished through its sleep, then re-entered
    assert [c["tenant_id"] for c in calls] == ["acme:acme", "acme:acme"]
    assert [c["runtime_url"] for c in calls] == ["http://runtime:28000"] * 2
    assert calls[1]["automation_rules"] is rules


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes,expected_seconds", [(1, 60), (7, 420), (30, 1800)])
async def test_sleep_equals_interval_minutes_times_60(
    monkeypatch, minutes, expected_seconds
):
    """The inter-cycle sleep is exactly annotation_interval_minutes * 60 seconds."""
    calls: list = []

    async def cycle(**kwargs):
        calls.append(kwargs)
        if len(calls) >= 2:
            raise _BreakLoop  # break before iteration 2's sleep
        return {"identified": 0, "already_annotated": 0, "enqueued": 0}

    rules = _rules_with_interval(minutes)
    sleeps = await _run_bounded_loop(monkeypatch, rules=rules, cycle=cycle)

    assert sleeps == [expected_seconds]
    assert len(calls) == 2


class _StubTelemetry:
    def __init__(self):
        self.config = type("_Cfg", (), {"provider_config": {}})()


@pytest.mark.asyncio
async def test_monitor_iteration_launches_and_cancels_annotation_task(monkeypatch):
    """_run_quality_monitor_iteration launches _annotation_loop as a task and
    cancels it in the shutdown finally once monitor.run() returns."""
    started = {"v": False}
    cancelled = {"v": False}
    # close() runs in the finally right after annotation_task.cancel() and
    # before asyncio.run's own teardown-cancel — reading cancelling() there
    # proves the finally (not teardown) requested the cancellation.
    close_saw_cancel_requested = {"v": False}

    async def stub_loop(tenant_id, runtime_url):
        started["v"] = True
        try:
            await asyncio.Event().wait()  # block until cancelled on shutdown
        except asyncio.CancelledError:
            cancelled["v"] = True
            raise

    created: list = []

    class _StubMonitor:
        last = None

        def __init__(self, **kwargs):
            self.closed = False
            type(self).last = self

        async def run(self):
            # Yield until the annotation task has actually started, then return
            # so the default branch reaches its cancel-in-finally.
            for _ in range(1000):
                if started["v"]:
                    return None
                await asyncio.sleep(0)
            return None

        async def close(self):
            if created:
                close_saw_cancel_requested["v"] = created[0].cancelling() > 0
            self.closed = True

    real_create_task = asyncio.create_task

    def spy_create_task(coro, *args, **kwargs):
        task = real_create_task(coro, *args, **kwargs)
        created.append(task)
        return task

    monkeypatch.setattr(qm, "_annotation_loop", stub_loop)
    monkeypatch.setattr(qm.asyncio, "create_task", spy_create_task)
    monitor = _StubMonitor()
    result = await qm._run_quality_monitor_iteration(
        monitor, "acme:acme", "http://localhost:28000"
    )

    # The iteration returns to its caller instead of ending the process: the
    # retry loop in main() decides what happens next. Pinned here because the
    # sidecar shares a pod with the serving API, so a terminating iteration
    # would drop the runtime from the Service's endpoints.
    assert result is None
    assert started["v"] is True  # the loop was launched in the default branch
    assert cancelled["v"] is True  # and cancelled on shutdown
    assert len(created) == 1  # exactly the annotation task, nothing else
    assert created[0].cancelled() is True  # the task ended in a cancelled state
    # cancel() ran in the finally itself, before asyncio.run's teardown-cancel.
    assert close_saw_cancel_requested["v"] is True
    assert monitor.closed is True  # monitor.close() ran in the finally


@pytest.mark.asyncio
async def test_missing_golden_keeps_annotation_sibling_until_shutdown(monkeypatch):
    """Expected golden absence cannot reconstruct the monitor process.

    The annotation sibling remains alive while the same monitor reaches live
    evaluation. External cancellation then owns both task and client cleanup.
    """
    from datetime import datetime

    from cogniverse_agents.optimizer.golden_set_ground_truth import (
        GoldenSetGroundTruthMissingError,
    )
    from cogniverse_evaluation.quality_monitor import LiveEvalResult, QualityMonitor

    annotation_started = asyncio.Event()
    annotation_cancelled = asyncio.Event()
    live_ran = asyncio.Event()

    async def annotation_loop(_tenant_id, _runtime_url):
        annotation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            annotation_cancelled.set()
            raise

    class _Client:
        closed = False

        async def aclose(self):
            self.closed = True

    monitor = QualityMonitor(
        tenant_id="acme:monitor",
        runtime_url="http://runtime",
        phoenix_http_endpoint="http://phoenix",
        llm_base_url="http://llm",
        llm_model="judge",
        golden_dataset_path="",
    )
    client = _Client()
    monitor._http_client = client

    async def missing_golden():
        await annotation_started.wait()
        raise GoldenSetGroundTruthMissingError("golden set is not configured")

    async def exact_live():
        result = LiveEvalResult(
            timestamp=datetime(2024, 1, 1), tenant_id="acme:monitor"
        )
        live_ran.set()
        return result

    monitor.evaluate_golden_set = missing_golden
    monitor.evaluate_live_traffic = exact_live
    monitor.check_thresholds = lambda golden, live: {}
    monkeypatch.setattr(qm, "_annotation_loop", annotation_loop)

    iteration = asyncio.create_task(
        qm._run_quality_monitor_iteration(monitor, "acme:monitor", "http://runtime")
    )
    live_wait = asyncio.create_task(live_ran.wait())
    done, _ = await asyncio.wait(
        {iteration, live_wait}, timeout=2, return_when=asyncio.FIRST_COMPLETED
    )

    assert live_wait in done
    assert iteration.done() is False
    assert annotation_cancelled.is_set() is False
    assert client.closed is False

    live_wait.cancel()
    iteration.cancel()
    with pytest.raises(asyncio.CancelledError):
        await iteration

    assert annotation_cancelled.is_set() is True
    assert client.closed is True
