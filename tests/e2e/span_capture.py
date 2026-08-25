"""Capture and replay Phoenix spans for batch optimizer seeding."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind as OTelSpanKind
from opentelemetry.trace import Status, StatusCode
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.trace.span_json_decoder import json_to_span

SpanRecord = dict[str, Any]

_DEFAULT_LOOKBACK_HOURS = 24.0
_REPLAY_STEP_NS = 1_000_000
_QUERY_BATCH_SIZE = 200


class SpanCaptureError(RuntimeError):
    """Raised when span capture or replay data is invalid."""


class SpanCaptureFileError(SpanCaptureError):
    """Raised when a capture file cannot be loaded."""


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _project_name(tenant_id: str) -> str:
    return f"cogniverse-{tenant_id}"


def _normalize_name_filter(
    span_names: str | Sequence[str] | None,
) -> str | Sequence[str] | None:
    if span_names is None or isinstance(span_names, str):
        return span_names
    return list(span_names)


def _capture_identity(record: SpanRecord) -> str:
    context = record.get("context")
    if isinstance(context, dict):
        span_id = context.get("span_id")
    else:
        span_id = record.get("context.span_id")
    if not span_id:
        raise SpanCaptureError("captured span is missing context.span_id")
    return str(span_id)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise SpanCaptureError(f"unsupported timestamp value: {value!r}")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _span_sort_key(record: SpanRecord, position: int) -> tuple[datetime, datetime, int]:
    start_time = _parse_timestamp(record["start_time"])
    end_time = _parse_timestamp(record.get("end_time") or record["start_time"])
    return start_time, end_time, position


def _validate_record(record: SpanRecord, *, path: Path | None = None) -> None:
    try:
        json_to_span(copy.deepcopy(record))
    except Exception as exc:  # noqa: BLE001
        if path is None:
            raise SpanCaptureError("invalid captured span record") from exc
        raise SpanCaptureFileError(f"capture file is invalid: {path}") from exc


def _ensure_non_empty(records: list[SpanRecord], *, path: Path | None = None) -> None:
    if records:
        return
    if path is None:
        raise SpanCaptureError("capture is empty")
    raise SpanCaptureFileError(f"capture file is empty: {path}")


def load_capture_json(path: str | Path) -> list[SpanRecord]:
    """Load a span capture file and return the raw span records."""
    capture_path = Path(path)
    try:
        raw_text = capture_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SpanCaptureFileError(f"capture file is missing: {capture_path}") from exc
    except OSError as exc:
        raise SpanCaptureFileError(
            f"capture file is unreadable: {capture_path}"
        ) from exc

    if not raw_text.strip():
        raise SpanCaptureFileError(f"capture file is empty: {capture_path}")

    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SpanCaptureFileError(
            f"capture file is unreadable: {capture_path}"
        ) from exc

    if not isinstance(loaded, list):
        raise SpanCaptureFileError(f"capture file is unreadable: {capture_path}")
    _ensure_non_empty(loaded, path=capture_path)
    for record in loaded:
        if not isinstance(record, dict):
            raise SpanCaptureFileError(f"capture file is unreadable: {capture_path}")
        _validate_record(record, path=capture_path)
    return loaded


def write_capture_json(path: str | Path, spans: Sequence[SpanRecord]) -> None:
    capture_path = Path(path)
    records = list(spans)
    _ensure_non_empty(records)
    for record in records:
        if not isinstance(record, dict):
            raise SpanCaptureError("capture records must be dicts")
        _validate_record(record)

    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def capture_spans(
    *,
    phoenix_http_endpoint: str,
    tenant_id: str,
    project_name: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    lookback_hours: float | None = _DEFAULT_LOOKBACK_HOURS,
    span_names: str | Sequence[str] | None = None,
    limit: int = 100_000,
    timeout_s: int = 120,
) -> list[SpanRecord]:
    """Read spans for a tenant from Phoenix and return JSON-ready records."""
    project = project_name or _project_name(tenant_id)
    if start_time is None and lookback_hours is not None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    client = Client(base_url=phoenix_http_endpoint)
    raw_spans = client.spans.get_spans(
        project_identifier=project,
        start_time=start_time,
        end_time=end_time,
        name=_normalize_name_filter(span_names),
        limit=limit,
        timeout=timeout_s,
    )

    captured: list[tuple[int, SpanRecord]] = []
    for position, span in enumerate(raw_spans):
        record = _json_round_trip(span)
        _validate_record(record)
        captured.append((position, record))

    if not captured:
        raise SpanCaptureError(
            f"no spans captured for project {project!r} from {phoenix_http_endpoint!r}"
        )

    ordered = []
    seen_span_ids: set[str] = set()
    for position, record in sorted(
        captured, key=lambda item: _span_sort_key(item[1], item[0])
    ):
        span_id = _capture_identity(record)
        if span_id in seen_span_ids:
            continue
        seen_span_ids.add(span_id)
        ordered.append(record)
    return ordered


def _existing_span_ids(
    *,
    phoenix_http_endpoint: str,
    project_name: str,
    span_ids: Sequence[str],
    start_time: Any | None = None,
) -> set[str]:
    if not span_ids:
        return set()

    client = Client(base_url=phoenix_http_endpoint)
    existing: set[str] = set()
    for start in range(0, len(span_ids), _QUERY_BATCH_SIZE):
        batch = span_ids[start : start + _QUERY_BATCH_SIZE]
        predicate = "span_id in [" + ", ".join(repr(span_id) for span_id in batch) + "]"
        spans = client.spans.get_spans_dataframe(
            project_identifier=project_name,
            query=SpanQuery().where(predicate),
            limit=len(batch),
            timeout=120,
            start_time=start_time,
        )
        if spans.empty or "context.span_id" not in spans.columns:
            continue
        existing.update(str(span_id) for span_id in spans["context.span_id"].dropna())
    return existing


def _flatten_attributes(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_attributes(child_prefix, item))
        return flattened
    if isinstance(value, (list, tuple)):
        return {prefix: list(value)}
    if value is None:
        return {}
    return {prefix: value}


def _span_attributes(record: SpanRecord) -> dict[str, Any]:
    attrs = record.get("attributes")
    if not isinstance(attrs, dict):
        raise SpanCaptureError(
            f"captured span {record.get('name')!r} has attributes of type "
            f"{type(attrs).__name__}, expected a mapping"
        )
    flattened: dict[str, Any] = {}
    for key, value in attrs.items():
        flattened.update(_flatten_attributes(key, value))
    return flattened


def _parse_ns(value: Any) -> int:
    dt = _parse_timestamp(value)
    return int(dt.timestamp() * 1_000_000_000)


def _event_timestamp_ns(
    *,
    event: dict[str, Any],
    original_start_ns: int,
    replay_start_ns: int,
    replay_end_ns: int,
    original_end_ns: int | None,
) -> int:
    timestamp = event.get("timestamp")
    if timestamp is None:
        return replay_start_ns
    try:
        event_ns = _parse_ns(timestamp)
    except SpanCaptureError:
        return replay_start_ns
    original_span_end_ns = original_end_ns or original_start_ns
    original_duration_ns = max(original_span_end_ns - original_start_ns, 0)
    replay_duration_ns = max(replay_end_ns - replay_start_ns, 0)
    if original_duration_ns == 0:
        return replay_start_ns
    offset_ns = event_ns - original_start_ns
    scaled_ns = replay_start_ns + min(
        max(offset_ns, 0),
        min(original_duration_ns, replay_duration_ns),
    )
    return min(scaled_ns, replay_end_ns)


class _FailureRecordingExporter(SpanExporter):
    """Surface a dropped export instead of letting the SDK swallow it.

    SimpleSpanProcessor discards the export result, so a Phoenix that is
    down, wrong-ported or speaking another protocol produces a full return
    value and an empty project.
    """

    def __init__(self, inner: SpanExporter, failures: list[str]) -> None:
        self._inner = inner
        self._failures = failures

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        result = self._inner.export(spans)
        if result is not SpanExportResult.SUCCESS:
            self._failures.append(
                f"{len(list(spans))} span(s) -> {getattr(result, 'name', result)}"
            )
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


def replay_spans(
    *,
    capture_path: str | Path,
    phoenix_http_endpoint: str,
    tenant_id: str,
    project_name: str | None = None,
    span_exporter: SpanExporter | None = None,
    existing_span_ids: Sequence[str] | None = None,
    existing_since: Any | None = None,
) -> list[SpanRecord]:
    """Replay a capture file into Phoenix with current timestamps."""
    records = load_capture_json(capture_path)
    project = project_name or _project_name(tenant_id)
    span_ids = [_capture_identity(record) for record in records]
    if existing_span_ids is None:
        # Scoped to the window the caller will read: the corpus was recorded
        # from this same project, so every recorded id still matches its own
        # original. Unscoped, that original suppresses the replay it came
        # from and the readable window stays empty.
        existing_ids = _existing_span_ids(
            phoenix_http_endpoint=phoenix_http_endpoint,
            project_name=project,
            span_ids=span_ids,
            start_time=existing_since,
        )
    else:
        existing_ids = {str(span_id) for span_id in existing_span_ids}
    pending_records = [
        record for record in records if _capture_identity(record) not in existing_ids
    ]
    if not pending_records:
        return []

    resource = Resource.create({"openinference.project.name": project})
    provider = TracerProvider(resource=resource)

    exporter = span_exporter
    if exporter is None:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter as _otlp_http

        # phoenix_http_endpoint is Phoenix's HTTP API. The gRPC exporter
        # cannot speak to it, and OTLP exporters log-and-drop rather than
        # raise, so the wrong protocol replays as a silent no-op.
        traces_url = phoenix_http_endpoint.rstrip("/") + "/v1/traces"
        exporter = _otlp_http.OTLPSpanExporter(endpoint=traces_url)

    failures: list[str] = []
    exporter = _FailureRecordingExporter(exporter, failures)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("cogniverse.span_capture")

    ordered_records = [
        record
        for _, record in sorted(
            enumerate(pending_records),
            key=lambda item: _span_sort_key(item[1], item[0]),
        )
    ]

    started_spans: dict[str, Any] = {}
    replay_start_ns = time.time_ns()
    total = len(ordered_records)
    start_base_ns = replay_start_ns - (total * _REPLAY_STEP_NS)

    try:
        for index, record in enumerate(ordered_records):
            span_id = _capture_identity(record)
            parent_id = record.get("parent_id")
            parent_span = started_spans.get(str(parent_id)) if parent_id else None
            parent_context = (
                trace.set_span_in_context(parent_span)
                if parent_span is not None
                else None
            )
            start_ns = start_base_ns + (index * _REPLAY_STEP_NS)
            end_ns = start_ns + _REPLAY_STEP_NS
            span = tracer.start_span(
                record["name"],
                context=parent_context,
                kind=OTelSpanKind.INTERNAL,
                attributes=_span_attributes(record),
                start_time=start_ns,
            )

            status_code = str(record.get("status_code") or "UNSET").upper()
            if status_code == "OK":
                otel_status = StatusCode.OK
            elif status_code == "ERROR":
                otel_status = StatusCode.ERROR
            else:
                otel_status = StatusCode.UNSET
            span.set_status(
                Status(otel_status, description=str(record.get("status_message") or ""))
            )

            original_start_ns = _parse_ns(record["start_time"])
            original_end_ns = None
            if record.get("end_time") is not None:
                original_end_ns = _parse_ns(record["end_time"])
            for event in record.get("events") or []:
                if not isinstance(event, dict):
                    continue
                span.add_event(
                    event.get("name") or "event",
                    attributes=_flatten_attributes("", event.get("attributes") or {}),
                    timestamp=_event_timestamp_ns(
                        event=event,
                        original_start_ns=original_start_ns,
                        replay_start_ns=start_ns,
                        replay_end_ns=end_ns,
                        original_end_ns=original_end_ns,
                    ),
                )

            started_spans[span_id] = span

        replayed: list[SpanRecord] = []
        for index, record in enumerate(ordered_records):
            span_id = _capture_identity(record)
            span = started_spans[span_id]
            start_ns = start_base_ns + (index * _REPLAY_STEP_NS)
            span.end(end_time=start_ns + _REPLAY_STEP_NS)
            replayed.append(record)
        provider.force_flush()
        if failures:
            raise SpanCaptureError(
                "span replay export failed; Phoenix received nothing for "
                f"{len(failures)} batch(es) at {phoenix_http_endpoint}: "
                f"{failures[:3]}"
            )
        return replayed
    finally:
        provider.shutdown()
