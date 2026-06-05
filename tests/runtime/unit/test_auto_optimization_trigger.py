"""Coverage for the auto-optimization trigger CLI script.

``scripts/auto_optimization_trigger.py`` is run by the Argo CronWorkflow to
conditionally trigger module optimization. It had zero test coverage. These
exercise the real decision logic: the marker-file time gate, the optimization
subprocess command it builds, the conditions tree in ``run()``, and the
UTC-aware trace-count window.
"""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "auto_optimization_trigger.py"
_spec = importlib.util.spec_from_file_location("auto_optimization_trigger", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AutoOptimizationTrigger = _mod.AutoOptimizationTrigger


def _bare_trigger(tenant_id: str = "aot_test", module: str = "routing"):
    """Construct without the heavy __init__ (config manager + telemetry)."""
    t = AutoOptimizationTrigger.__new__(AutoOptimizationTrigger)
    t.tenant_id = tenant_id
    t.module = module
    t.config_manager = MagicMock()
    t.provider = MagicMock()
    return t


def _marker_path(tenant_id: str, module: str) -> Path:
    return Path(f"/tmp/auto_opt_{tenant_id}_{module}.marker")


@pytest.fixture
def clean_marker():
    """Yield a (tenant, module) whose marker file is absent and cleaned up."""
    tenant, module = "aot_marker_rt", "routing"
    path = _marker_path(tenant, module)
    path.unlink(missing_ok=True)
    yield tenant, module
    path.unlink(missing_ok=True)


@pytest.mark.unit
class TestMarkerTimeGate:
    def test_no_marker_allows_run(self, clean_marker):
        tenant, module = clean_marker
        t = _bare_trigger(tenant, module)
        should_run, reason = t.check_last_optimization_time(3600)
        assert should_run is True
        assert reason == "No previous optimization found"

    def test_recent_marker_blocks_run(self, clean_marker):
        tenant, module = clean_marker
        _marker_path(tenant, module).write_text(datetime.now().isoformat())
        t = _bare_trigger(tenant, module)
        should_run, reason = t.check_last_optimization_time(3600)
        assert should_run is False
        assert "elapsed" in reason

    def test_old_marker_allows_run(self, clean_marker):
        tenant, module = clean_marker
        old = (datetime.now() - timedelta(hours=2)).isoformat()
        _marker_path(tenant, module).write_text(old)
        t = _bare_trigger(tenant, module)
        should_run, reason = t.check_last_optimization_time(3600)
        assert should_run is True
        assert "elapsed" in reason

    def test_corrupt_marker_allows_run(self, clean_marker):
        tenant, module = clean_marker
        _marker_path(tenant, module).write_text("not-a-timestamp")
        t = _bare_trigger(tenant, module)
        should_run, reason = t.check_last_optimization_time(3600)
        assert should_run is True
        assert reason.startswith("Error checking marker file")

    def test_update_marker_writes_parseable_timestamp(self, clean_marker):
        tenant, module = clean_marker
        t = _bare_trigger(tenant, module)
        t.update_marker_file()
        written = _marker_path(tenant, module).read_text().strip()
        # Round-trips through fromisoformat (what check_last_optimization_time reads).
        assert datetime.fromisoformat(written)


@pytest.mark.unit
class TestTriggerOptimizationCommand:
    def test_builds_optimization_cli_command_and_updates_marker(
        self, clean_marker, monkeypatch
    ):
        tenant, module = clean_marker
        t = _bare_trigger(tenant, module)

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["timeout"] = kwargs.get("timeout")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)

        assert t.trigger_optimization(lookback_hours=12) is True
        assert captured["cmd"] == [
            "python",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            module,
            "--tenant-id",
            tenant,
            "--lookback-hours",
            "12",
        ]
        assert captured["timeout"] == 3600
        # Success path stamps the marker so the next run respects the interval.
        assert _marker_path(tenant, module).exists()

    def test_nonzero_returncode_is_failure_and_no_marker(
        self, clean_marker, monkeypatch
    ):
        tenant, module = clean_marker
        t = _bare_trigger(tenant, module)
        monkeypatch.setattr(
            _mod.subprocess,
            "run",
            lambda cmd, **kw: MagicMock(returncode=1, stdout="boom", stderr="err"),
        )
        assert t.trigger_optimization() is False
        assert not _marker_path(tenant, module).exists()

    def test_timeout_is_failure(self, clean_marker, monkeypatch):
        tenant, module = clean_marker
        t = _bare_trigger(tenant, module)

        def fake_run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 3600)

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        assert t.trigger_optimization() is False
        assert not _marker_path(tenant, module).exists()


@pytest.mark.unit
class TestRunDecisionTree:
    @pytest.mark.asyncio
    async def test_disabled_returns_1(self):
        t = _bare_trigger()
        t.check_auto_optimization_enabled = MagicMock(return_value=(False, {}))
        assert await t.run() == 1

    @pytest.mark.asyncio
    async def test_time_gate_skip_returns_1(self):
        t = _bare_trigger()
        t.check_auto_optimization_enabled = MagicMock(
            return_value=(True, {"interval_seconds": 3600, "min_samples": 10})
        )
        t.check_last_optimization_time = MagicMock(return_value=(False, "too soon"))
        assert await t.run() == 1

    @pytest.mark.asyncio
    async def test_insufficient_traces_returns_1(self):
        t = _bare_trigger()
        t.check_auto_optimization_enabled = MagicMock(
            return_value=(True, {"interval_seconds": 3600, "min_samples": 10})
        )
        t.check_last_optimization_time = MagicMock(return_value=(True, "ok"))
        t.check_trace_count = AsyncMock(return_value=(False, 3))
        assert await t.run() == 1

    @pytest.mark.asyncio
    async def test_all_conditions_met_triggers_and_returns_0(self):
        t = _bare_trigger()
        t.check_auto_optimization_enabled = MagicMock(
            return_value=(True, {"interval_seconds": 3600, "min_samples": 10})
        )
        t.check_last_optimization_time = MagicMock(return_value=(True, "ok"))
        t.check_trace_count = AsyncMock(return_value=(True, 50))
        t.trigger_optimization = MagicMock(return_value=True)
        assert await t.run() == 0
        t.trigger_optimization.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_failure_returns_2(self):
        t = _bare_trigger()
        t.check_auto_optimization_enabled = MagicMock(
            return_value=(True, {"interval_seconds": 3600, "min_samples": 10})
        )
        t.check_last_optimization_time = MagicMock(return_value=(True, "ok"))
        t.check_trace_count = AsyncMock(return_value=(True, 50))
        t.trigger_optimization = MagicMock(return_value=False)
        assert await t.run() == 2


@pytest.mark.unit
class TestTraceCountWindow:
    @pytest.mark.asyncio
    async def test_trace_window_is_utc_aware(self):
        """The Phoenix lookback window must be UTC-aware so a non-UTC host
        doesn't shift it (a naive datetime.now() would drop all traces)."""
        t = _bare_trigger()
        captured = {}

        async def fake_get_spans(**kwargs):
            captured.update(kwargs)
            import pandas as pd

            return pd.DataFrame([{"x": 1}] * 5)

        t.provider.traces.get_spans = fake_get_spans

        sufficient, count = await t.check_trace_count(min_samples=3, lookback_hours=24)
        assert (sufficient, count) == (True, 5)
        assert captured["start_time"].tzinfo is not None
        assert captured["end_time"].tzinfo is not None
        # Window spans exactly the requested lookback.
        assert captured["end_time"] - captured["start_time"] == timedelta(hours=24)

    @pytest.mark.asyncio
    async def test_empty_spans_is_insufficient(self):
        t = _bare_trigger()
        import pandas as pd

        t.provider.traces.get_spans = AsyncMock(return_value=pd.DataFrame())
        assert await t.check_trace_count(min_samples=1) == (False, 0)
