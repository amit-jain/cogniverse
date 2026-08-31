from __future__ import annotations

import gc
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_runtime.optimization_cli import run_monthly_reports


class _Tracker:
    def __init__(self):
        self.live_rows = 0
        self.peak_live_rows = 0
        self.client_calls: list[dict] = []


class _TrackingRow:
    __slots__ = ("start_time", "end_time", "status_code", "_tracker")

    def __init__(
        self,
        tracker: _Tracker,
        *,
        start_time: datetime,
        end_time: datetime,
        status_code: str | None,
    ) -> None:
        self._tracker = tracker
        self.start_time = start_time
        self.end_time = end_time
        self.status_code = status_code
        tracker.live_rows += 1
        tracker.peak_live_rows = max(tracker.peak_live_rows, tracker.live_rows)

    def __del__(self):
        tracker = getattr(self, "_tracker", None)
        if tracker is not None:
            tracker.live_rows -= 1


class _TrackingSeries:
    def __init__(self, values):
        self._values = list(values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def dropna(self):
        return type(self)(value for value in self._values if value is not None)


class _TrackingBoolSeries:
    def __init__(self, values):
        self._values = [bool(value) for value in values]

    def sum(self):
        return sum(self._values)


class _TrackingStatusStringAccessor:
    def __init__(self, values):
        self._values = list(values)

    def upper(self):
        return _TrackingStatusSeries(
            value.upper() if value is not None else None for value in self._values
        )


class _TrackingStatusSeries:
    def __init__(self, values):
        self._values = list(values)

    def fillna(self, value):
        return type(self)(
            value if current is None else current for current in self._values
        )

    @property
    def str(self):
        return _TrackingStatusStringAccessor(self._values)

    def ne(self, other):
        return _TrackingBoolSeries(value != other for value in self._values)


class _TrackingDataFrame:
    def __init__(self, tracker: _Tracker, rows: list[_TrackingRow], columns):
        self._tracker = tracker
        self._rows = rows
        self.columns = tuple(columns)

    @property
    def empty(self):
        return not self._rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if key == "start_time":
            return _TrackingSeries(row.start_time for row in self._rows)
        if key == "end_time":
            return _TrackingSeries(row.end_time for row in self._rows)
        if key == "status_code":
            return _TrackingStatusSeries(row.status_code for row in self._rows)
        raise KeyError(key)

    def itertuples(self, index=False):  # noqa: ARG002
        return iter(self._rows)


class _FakeTraceStore:
    def __init__(self, tracker: _Tracker, row_specs: list[dict]):
        self._tracker = tracker
        self._row_specs = list(row_specs)

    def _filter_specs(self, start_time, end_time):
        rows = []
        for spec in self._row_specs:
            row_start = spec["start_time"]
            if start_time is not None and row_start < start_time:
                continue
            if end_time is not None and row_start >= end_time:
                continue
            rows.append(spec)
        return rows

    async def get_spans(self, **kwargs):
        selected_columns = tuple(
            kwargs.get("columns") or ("start_time", "end_time", "status_code")
        )
        self._tracker.client_calls.append(
            {
                "project": kwargs["project"],
                "start_time": kwargs["start_time"],
                "end_time": kwargs["end_time"],
                "limit": kwargs["limit"],
                "columns": selected_columns,
            }
        )

        rows = self._filter_specs(kwargs["start_time"], kwargs["end_time"])
        rows = rows[: kwargs["limit"]]
        tracking_rows = [
            _TrackingRow(
                self._tracker,
                start_time=row["start_time"],
                end_time=row["end_time"],
                status_code=row["status_code"]
                if "status_code" in selected_columns
                else None,
            )
            for row in rows
        ]
        return _TrackingDataFrame(self._tracker, tracking_rows, selected_columns)


class _FakeProvider:
    def __init__(self, trace_store: _FakeTraceStore):
        self.traces = trace_store


class _FakeTelemetryManager:
    def __init__(self, trace_store: _FakeTraceStore):
        self.config = SimpleNamespace(get_project_name=lambda tid: f"proj-{tid}")
        self._provider = _FakeProvider(trace_store)

    def get_provider(self, tenant_id=None):  # noqa: ARG002
        return self._provider


def _make_span_specs(
    *,
    count: int,
    base_time: datetime,
    step: timedelta,
    latencies_ms: list[float],
    statuses: list[str | None] | None = None,
):
    statuses = statuses or ["OK"] * count
    specs = []
    for index in range(count):
        start_time = base_time + (step * index)
        latency_ms = latencies_ms[index % len(latencies_ms)]
        specs.append(
            {
                "start_time": start_time,
                "end_time": start_time + timedelta(milliseconds=latency_ms),
                "status_code": statuses[index % len(statuses)],
            }
        )
    return specs


def _install_monthly_reports_tenant_manager(monkeypatch, *, org_id, tenant_ids):
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: SimpleNamespace(name="config_manager"),
    )
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.set_schema_loader",
        lambda loader: None,
    )
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.set_config_manager",
        lambda config_manager: None,
    )

    async def _list_organizations_internal():
        return [org_id]

    async def _list_tenants_for_org_internal(_org_id):
        return [
            SimpleNamespace(
                tenant_full_id=tenant_id,
                tenant_name=tenant_id.split(":", 1)[1],
                status="active",
                schemas_deployed=["agent_memories"],
            )
            for tenant_id in tenant_ids
        ]

    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.list_organizations_internal",
        _list_organizations_internal,
    )
    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.list_tenants_for_org_internal",
        _list_tenants_for_org_internal,
    )


def _make_monthly_reports_environment(monkeypatch, row_specs):
    tracker = _Tracker()
    trace_store = _FakeTraceStore(tracker, row_specs)
    manager = _FakeTelemetryManager(trace_store)
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda otlp_endpoint=None: manager,
    )
    return tracker, trace_store


@pytest.mark.asyncio
async def test_monthly_reports_surfaces_phoenix_outages(monkeypatch, tmp_path):
    org_id = "monthly_rep_fault_org"
    tenant_ids = [f"{org_id}:alpha", f"{org_id}:beta"]
    _install_monthly_reports_tenant_manager(
        monkeypatch,
        org_id=org_id,
        tenant_ids=tenant_ids,
    )

    tracker, trace_store = _make_monthly_reports_environment(
        monkeypatch,
        [
            {
                "start_time": datetime.now(timezone.utc) - timedelta(minutes=10),
                "end_time": datetime.now(timezone.utc)
                - timedelta(minutes=10)
                + timedelta(milliseconds=5),
                "status_code": "OK",
            }
        ],
    )
    assert tracker.peak_live_rows == 0

    async def _raise_get_spans(**kwargs):  # noqa: ARG001
        project = kwargs["project"]
        raise RuntimeError(
            f"Failed to query every span from Phoenix project {project}"
        ) from ConnectionError("phoenix unreachable")

    trace_store.get_spans = _raise_get_spans

    result = await run_monthly_reports(
        output_dir=str(tmp_path / "reports"),
        lookback_hours=1.0,
    )

    assert result["failed"] == tenant_ids
    for tid in tenant_ids:
        error = result["failed_details"][tid]
        assert "Failed to query every span from Phoenix project" in error
        assert "phoenix unreachable" in error


@pytest.mark.asyncio
async def test_monthly_reports_pins_exact_span_columns(monkeypatch, tmp_path):
    org_id = "monthly_rep_org"
    tenant_ids = [f"{org_id}:prod"]
    _install_monthly_reports_tenant_manager(
        monkeypatch,
        org_id=org_id,
        tenant_ids=tenant_ids,
    )

    now = datetime.now(timezone.utc)
    row_specs = _make_span_specs(
        count=4,
        base_time=now - timedelta(minutes=30),
        step=timedelta(minutes=1),
        latencies_ms=[10.0, 20.0, 30.0, 40.0],
        statuses=["OK", "OK", "ERROR", "OK"],
    )
    tracker, _trace_store = _make_monthly_reports_environment(monkeypatch, row_specs)

    result = await run_monthly_reports(
        output_dir=str(tmp_path / "reports"),
        lookback_hours=1.0,
    )

    assert result["summary"]["perf_tenants_with_data"] == 1
    assert len(result["files_written"]) == 2
    assert len(tracker.client_calls) == 1
    call = tracker.client_calls[0]
    assert call["project"] == "proj-monthly_rep_org:prod"
    assert call["limit"] == 512
    assert call["columns"] == (
        "start_time",
        "end_time",
        "status_code",
    )


@pytest.mark.parametrize(
    "span_count, expected_peak_live_rows",
    [(100, 100), (1000, 512)],
)
@pytest.mark.asyncio
async def test_monthly_reports_bounds_retained_span_rows(
    monkeypatch, tmp_path, span_count, expected_peak_live_rows
):
    org_id = "monthly_rep_scale_org"
    tenant_ids = [f"{org_id}:prod"]
    _install_monthly_reports_tenant_manager(
        monkeypatch,
        org_id=org_id,
        tenant_ids=tenant_ids,
    )

    now = datetime.now(timezone.utc)
    row_specs = _make_span_specs(
        count=span_count,
        base_time=now - timedelta(minutes=55),
        step=timedelta(seconds=3),
        latencies_ms=[10.0],
        statuses=["OK"] * span_count,
    )
    tracker, _trace_store = _make_monthly_reports_environment(monkeypatch, row_specs)

    result = await run_monthly_reports(
        output_dir=str(tmp_path / "reports"),
        lookback_hours=1.0,
    )

    perf_path = Path(result["files_written"][1])
    perf = json.loads(perf_path.read_text())
    entry = perf["tenants"][tenant_ids[0]]
    assert entry["span_count"] == span_count
    assert tracker.peak_live_rows == expected_peak_live_rows
    gc.collect()
    assert tracker.live_rows == 0


@pytest.mark.asyncio
async def test_monthly_reports_preserves_exact_values_under_small_input(
    monkeypatch, tmp_path
):
    org_id = "monthly_rep_exact_org"
    tenant_ids = [f"{org_id}:prod"]
    _install_monthly_reports_tenant_manager(
        monkeypatch,
        org_id=org_id,
        tenant_ids=tenant_ids,
    )

    now = datetime.now(timezone.utc)
    row_specs = _make_span_specs(
        count=6,
        base_time=now - timedelta(minutes=50),
        step=timedelta(minutes=1),
        latencies_ms=[50.0, 10.0, 40.0, 20.0, 60.0, 30.0],
        statuses=["OK", "ERROR", "OK", "OK", "ERROR", "OK"],
    )
    tracker, _trace_store = _make_monthly_reports_environment(monkeypatch, row_specs)

    result = await run_monthly_reports(
        output_dir=str(tmp_path / "reports"),
        lookback_hours=1.0,
    )

    perf_path = Path(result["files_written"][1])
    perf = json.loads(perf_path.read_text())
    entry = perf["tenants"][tenant_ids[0]]
    assert entry == {
        "span_count": 6,
        "latency_ms_mean": 35.0,
        "latency_ms_p50": 30.0,
        "latency_ms_p95": 50.0,
        "error_rate": 0.3333,
    }

    usage_path = Path(result["files_written"][0])
    usage = json.loads(usage_path.read_text())
    assert usage["summary"] == {
        "org_count": 1,
        "tenant_count": 1,
        "schema_count": 1,
    }
    assert usage["organizations"][org_id]["tenant_count"] == 1

    gc.collect()
    assert tracker.live_rows == 0
