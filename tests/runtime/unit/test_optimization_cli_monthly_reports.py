"""Unit tests for the monthly-report contracts."""

from __future__ import annotations

import gc
import heapq
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_runtime import optimization_cli as cli

pytestmark = pytest.mark.unit


class _OpenTracker:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    def track(self, handle):
        self.active += 1
        self.peak = max(self.peak, self.active)
        return _TrackedHandle(handle, self)


class _TrackedHandle:
    def __init__(self, handle, tracker: _OpenTracker) -> None:
        self._handle = handle
        self._tracker = tracker
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._handle)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                return self._handle.close()
            finally:
                self._tracker.active -= 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._handle.__exit__(exc_type, exc, tb)
        finally:
            self.close()


def _write_sorted_chunk(path: Path, values: list[float]) -> Path:
    path.write_text("\n".join(f"{value:.17g}" for value in values))
    return path


def _reference_percentiles(
    chunk_paths: list[Path],
    latency_count: int,
) -> tuple[float | None, float | None]:
    if latency_count <= 0:
        return None, None

    p50_index = cli._monthly_report_percentile_index(latency_count, 0.50)
    p95_index = cli._monthly_report_percentile_index(latency_count, 0.95)
    p50 = None
    p95 = None
    merged = heapq.merge(
        *(cli._iter_monthly_report_sorted_latencies(path) for path in chunk_paths)
    )
    for index, value in enumerate(merged):
        if index == p50_index:
            p50 = round(float(value), 3)
        if index == p95_index:
            p95 = round(float(value), 3)
        if p50 is not None and p95 is not None:
            break
    return p50, p95


def test_monthly_report_percentiles_bound_open_files(tmp_path, monkeypatch):
    chunk_paths = [
        _write_sorted_chunk(tmp_path / f"chunk-{index:02d}.txt", [float(index + 1)])
        for index in range(7)
    ]
    tracker = _OpenTracker()
    original_open = Path.open

    def counting_open(self, *args, **kwargs):
        return tracker.track(original_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", counting_open)
    monkeypatch.setattr(cli, "_MONTHLY_REPORT_MERGE_FAN_IN", 3, raising=False)

    result = cli._monthly_report_percentiles(chunk_paths, len(chunk_paths))

    assert result == (4.0, 6.0)
    assert tracker.peak <= 4, (
        "monthly-report merges must keep file descriptors bounded; "
        f"observed peak={tracker.peak}"
    )


def test_monthly_report_percentiles_match_all_at_once_merge(tmp_path, monkeypatch):
    chunk_paths = [
        _write_sorted_chunk(tmp_path / f"chunk-{index:02d}.txt", values)
        for index, values in enumerate(
            (
                [1.0, 10.0],
                [2.0, 8.0],
                [3.0, 7.0],
                [4.0, 6.0],
                [5.0, 9.0],
            )
        )
    ]
    monkeypatch.setattr(cli, "_MONTHLY_REPORT_MERGE_FAN_IN", 3, raising=False)

    bounded = cli._monthly_report_percentiles(chunk_paths, 10)
    reference = _reference_percentiles(chunk_paths, 10)

    assert bounded == reference == (5.0, 9.0)


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

    def _build_frame(self, rows: list[dict], selected_columns):
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

    async def get_spans(self, **kwargs):
        selected_columns = tuple(
            kwargs.get("columns") or ("start_time", "end_time", "status_code")
        )

        rows = self._filter_specs(kwargs["start_time"], kwargs["end_time"])
        rows = rows[: kwargs["limit"]]
        self._tracker.client_calls.append(
            {
                "project": kwargs["project"],
                "start_time": kwargs["start_time"],
                "end_time": kwargs["end_time"],
                "page_size": kwargs["limit"],
                "columns": selected_columns,
                "span_ids": [row["span_id"] for row in rows],
            }
        )
        return self._build_frame(rows, selected_columns)

    async def iter_spans(self, **kwargs):
        selected_columns = tuple(
            kwargs.get("columns") or ("start_time", "end_time", "status_code")
        )
        page_size = kwargs["page_size"]
        rows = self._filter_specs(kwargs["start_time"], kwargs["end_time"])

        for offset in range(0, len(rows), page_size):
            page_rows = rows[offset : offset + page_size]
            self._tracker.client_calls.append(
                {
                    "project": kwargs["project"],
                    "start_time": kwargs["start_time"],
                    "end_time": kwargs["end_time"],
                    "page_size": page_size,
                    "columns": selected_columns,
                    "span_ids": [row["span_id"] for row in page_rows],
                }
            )
            yield self._build_frame(page_rows, selected_columns)


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
                "span_id": f"span-{index:05d}",
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

    async def _raise_iter_spans(**kwargs):  # noqa: ARG001
        project = kwargs["project"]
        raise RuntimeError(
            f"Failed to query every span from Phoenix project {project}"
        ) from ConnectionError("phoenix unreachable")
        if False:  # pragma: no cover
            yield None

    trace_store.iter_spans = _raise_iter_spans

    result = await cli.run_monthly_reports(
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

    result = await cli.run_monthly_reports(
        output_dir=str(tmp_path / "reports"),
        lookback_hours=1.0,
    )

    assert result["summary"]["perf_tenants_with_data"] == 1
    assert len(result["files_written"]) == 2
    assert len(tracker.client_calls) == 1
    call = tracker.client_calls[0]
    assert call["project"] == "proj-monthly_rep_org:prod"
    assert call["page_size"] == cli._MONTHLY_REPORT_SPAN_PAGE_SIZE
    assert len(call["span_ids"]) == 4
    assert call["columns"] == (
        "start_time",
        "end_time",
        "status_code",
    )


@pytest.mark.asyncio
async def test_monthly_reports_streams_each_page_once(monkeypatch, tmp_path):
    org_id = "monthly_rep_page_org"
    tenant_ids = [f"{org_id}:prod"]
    _install_monthly_reports_tenant_manager(
        monkeypatch,
        org_id=org_id,
        tenant_ids=tenant_ids,
    )

    now = datetime.now(timezone.utc)
    row_specs = _make_span_specs(
        count=1_005,
        base_time=now - timedelta(minutes=55),
        step=timedelta(seconds=2),
        latencies_ms=[10.0, 20.0, 30.0, 40.0, 50.0],
        statuses=["OK", "ERROR", "OK", "OK", "ERROR"],
    )
    tracker, _trace_store = _make_monthly_reports_environment(monkeypatch, row_specs)

    result = await cli.run_monthly_reports(
        output_dir=str(tmp_path / "reports"),
        lookback_hours=1.0,
    )

    perf_path = Path(result["files_written"][1])
    perf = json.loads(perf_path.read_text())
    entry = perf["tenants"][tenant_ids[0]]
    assert entry["span_count"] == 1_005

    page_size = cli._MONTHLY_REPORT_SPAN_PAGE_SIZE
    expected_call_count = (len(row_specs) + page_size - 1) // page_size
    assert len(tracker.client_calls) == expected_call_count

    page_lengths = [len(call["span_ids"]) for call in tracker.client_calls]
    assert page_lengths == [512, 493]

    flattened_span_ids = [
        span_id for call in tracker.client_calls for span_id in call["span_ids"]
    ]
    expected_span_ids = [spec["span_id"] for spec in row_specs]
    assert flattened_span_ids == expected_span_ids
    assert len(flattened_span_ids) == len(set(flattened_span_ids))


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

    result = await cli.run_monthly_reports(
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

    result = await cli.run_monthly_reports(
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
