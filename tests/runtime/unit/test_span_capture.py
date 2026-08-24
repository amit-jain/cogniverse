from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from tests.e2e.span_capture import (
    SpanCaptureError,
    SpanCaptureFileError,
    _span_attributes,
    capture_spans,
    load_capture_json,
    replay_spans,
    write_capture_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _span_record(
    *,
    name: str,
    trace_id: str,
    span_id: str,
    start_time: str,
    end_time: str,
    attributes: dict[str, object],
    parent_id: str | None = None,
    span_kind: str = "LLM",
    status_code: str = "OK",
    status_message: str = "",
    events: list[dict[str, object]] | None = None,
    conversation_id: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "context": {"trace_id": trace_id, "span_id": span_id},
        "span_kind": span_kind,
        "parent_id": parent_id,
        "start_time": start_time,
        "end_time": end_time,
        "status_code": status_code,
        "status_message": status_message,
        "attributes": attributes,
        "events": events or [],
        "conversation": (
            None if conversation_id is None else {"conversation_id": conversation_id}
        ),
    }


class _FakeSpans:
    def __init__(self, spans: list[dict[str, object]]) -> None:
        self.spans = spans
        self.calls: list[dict[str, object]] = []

    def get_spans(self, **kwargs):
        self.calls.append(kwargs)
        return self.spans


class _FakeClient:
    last_instance: "_FakeClient | None" = None
    spans_data: list[dict[str, object]] = []

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.spans = _FakeSpans(type(self).spans_data)
        type(self).last_instance = self


def test_capture_spans_sorts_and_dedupes_by_span_identity(monkeypatch):
    from tests.e2e import span_capture as span_capture_mod

    records = [
        _span_record(
            name="later",
            trace_id="trace-b",
            span_id="span-b",
            start_time="2026-08-24T10:00:10Z",
            end_time="2026-08-24T10:00:11Z",
            attributes={
                "input.value": "later query",
                "output.value": "later output",
                "operation": "routing",
                "modality": "text",
            },
        ),
        _span_record(
            name="earlier",
            trace_id="trace-a",
            span_id="span-a",
            start_time="2026-08-24T10:00:05Z",
            end_time="2026-08-24T10:00:06Z",
            attributes={
                "input.value": "earlier query",
                "output.value": "earlier output",
                "operation": "profile_selection",
                "modality": "video",
            },
        ),
        _span_record(
            name="duplicate",
            trace_id="trace-c",
            span_id="span-a",
            start_time="2026-08-24T10:00:15Z",
            end_time="2026-08-24T10:00:16Z",
            attributes={
                "input.value": "duplicate query",
                "output.value": "duplicate output",
                "operation": "profile_selection",
                "modality": "video",
            },
        ),
    ]
    _FakeClient.spans_data = records
    monkeypatch.setattr(span_capture_mod, "Client", _FakeClient)

    start_time = datetime(2026, 8, 24, 9, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 8, 24, 11, 0, tzinfo=timezone.utc)
    captured = capture_spans(
        phoenix_http_endpoint="http://phoenix.example",
        tenant_id="tenant-x",
        start_time=start_time,
        end_time=end_time,
        lookback_hours=None,
    )

    expected = [records[1], records[0]]
    assert captured == expected
    client_instance = _FakeClient.last_instance
    if client_instance is None:
        pytest.fail("Client was not constructed")
    assert client_instance.base_url == "http://phoenix.example"
    assert client_instance.spans.calls == [
        {
            "project_identifier": "cogniverse-tenant-x",
            "start_time": start_time,
            "end_time": end_time,
            "name": None,
            "limit": 100000,
            "timeout": 120,
        }
    ]


def test_capture_json_round_trip_preserves_full_record(tmp_path):
    path = tmp_path / "capture.json"
    records = [
        _span_record(
            name="cogniverse.profile_selection",
            trace_id="trace-a",
            span_id="span-a",
            start_time="2026-08-24T10:00:05Z",
            end_time="2026-08-24T10:00:06Z",
            attributes={
                "input.value": "show me cat videos",
                "output.value": '{"selected_profile":"video_colpali"}',
                "operation": "profile_selection",
                "modality": "video",
                "available_profiles": "video_colpali,text_bm25",
                "prompt": {"role": "user", "text": "show me cat videos"},
                "scores": [0.9, 0.8],
            },
            events=[
                {
                    "name": "checkpoint",
                    "timestamp": "2026-08-24T10:00:05.500000Z",
                    "attributes": {"phase": "mid", "step": 1},
                }
            ],
            conversation_id="11111111-1111-1111-1111-111111111111",
        ),
        _span_record(
            name="cogniverse.query_enhancement",
            trace_id="trace-b",
            span_id="span-b",
            parent_id="span-a",
            start_time="2026-08-24T10:00:07Z",
            end_time="2026-08-24T10:00:09Z",
            attributes={
                "input.value": "robot tutorials",
                "output.value": '{"enhanced_query":"robot tutorials"}',
                "operation": "query_enhancement",
                "modality": "text",
                "available_profiles": "video_colpali,text_bm25",
                "extra": {"nested": "value"},
                "choices": ["a", "b"],
            },
            status_message="ok",
        ),
    ]

    write_capture_json(path, records)
    loaded = load_capture_json(path)

    assert loaded == records


def test_replay_spans_emits_exact_name_and_attribute_mapping(tmp_path):
    path = tmp_path / "capture.json"
    record = _span_record(
        name="cogniverse.profile_selection",
        trace_id="trace-a",
        span_id="span-a",
        start_time="2026-08-24T10:00:05Z",
        end_time="2026-08-24T10:00:06Z",
        attributes={
            "input.value": "show me cat videos",
            "output.value": '{"selected_profile":"video_colpali"}',
            "operation": "profile_selection",
            "modality": "video",
            "available_profiles": "video_colpali,text_bm25",
            "confidence": 0.91,
        },
    )
    write_capture_json(path, [record])

    exporter = InMemorySpanExporter()
    replay_spans(
        capture_path=path,
        phoenix_http_endpoint="http://phoenix.example",
        tenant_id="tenant-x",
        span_exporter=exporter,
        existing_span_ids=(),
    )

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == record["name"]
    assert dict(finished[0].attributes) == record["attributes"]
    assert (
        finished[0].resource.attributes["openinference.project.name"]
        == "cogniverse-tenant-x"
    )


def test_replay_spans_rewrites_timestamps_and_preserves_order(tmp_path):
    path = tmp_path / "capture.json"
    records = [
        _span_record(
            name="first",
            trace_id="trace-a",
            span_id="span-a",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:00:02Z",
            attributes={
                "input.value": "first",
                "output.value": "first",
                "operation": "routing",
                "modality": "text",
            },
        ),
        _span_record(
            name="second",
            trace_id="trace-b",
            span_id="span-b",
            parent_id="span-a",
            start_time="2024-01-01T00:00:05Z",
            end_time="2024-01-01T00:00:06Z",
            attributes={
                "input.value": "second",
                "output.value": "second",
                "operation": "routing",
                "modality": "text",
            },
        ),
    ]
    write_capture_json(path, records)

    exporter = InMemorySpanExporter()
    before_ns = time.time_ns()
    replay_spans(
        capture_path=path,
        phoenix_http_endpoint="http://phoenix.example",
        tenant_id="tenant-x",
        span_exporter=exporter,
        existing_span_ids=(),
    )
    after_ns = time.time_ns()

    finished = exporter.get_finished_spans()
    assert len(finished) == 2
    assert [span.name for span in finished] == ["first", "second"]
    assert finished[0].start_time < finished[1].start_time
    assert finished[0].end_time < finished[1].end_time
    assert before_ns <= finished[1].end_time <= after_ns
    assert (
        finished[0].resource.attributes["openinference.project.name"]
        == "cogniverse-tenant-x"
    )


def test_replay_spans_skips_existing_span_ids(tmp_path):
    path = tmp_path / "capture.json"
    records = [
        _span_record(
            name="first",
            trace_id="trace-a",
            span_id="span-a",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:00:02Z",
            attributes={
                "input.value": "first",
                "output.value": "first",
                "operation": "routing",
                "modality": "text",
            },
        ),
        _span_record(
            name="second",
            trace_id="trace-b",
            span_id="span-b",
            start_time="2024-01-01T00:00:05Z",
            end_time="2024-01-01T00:00:06Z",
            attributes={
                "input.value": "second",
                "output.value": "second",
                "operation": "routing",
                "modality": "text",
            },
        ),
    ]
    write_capture_json(path, records)

    exporter = InMemorySpanExporter()
    replay_spans(
        capture_path=path,
        phoenix_http_endpoint="http://phoenix.example",
        tenant_id="tenant-x",
        span_exporter=exporter,
        existing_span_ids={"span-a"},
    )

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "second"
    assert dict(finished[0].attributes) == records[1]["attributes"]


def test_load_capture_missing_raises_with_path(tmp_path):
    path = tmp_path / "missing.json"

    with pytest.raises(SpanCaptureFileError) as excinfo:
        load_capture_json(path)

    assert str(path) in str(excinfo.value)


def test_load_capture_empty_raises_with_path(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")

    with pytest.raises(SpanCaptureFileError) as excinfo:
        load_capture_json(path)

    assert str(path) in str(excinfo.value)


def test_replay_refuses_a_record_whose_attributes_are_not_a_mapping():
    """A span with no usable attributes must raise, never replay empty.

    The optimizer reads its training population out of span attributes, so a
    record that replays with none of them is indistinguishable from a span
    the agents never produced. Silently emitting it seeds a corpus the
    optimizer cannot learn from and reports success.
    """
    record = _span_record(
        name="cogniverse.query_enhancement",
        trace_id="trace-attrs",
        span_id="span-attrs",
        start_time="2026-08-24T10:00:00.000000+00:00",
        end_time="2026-08-24T10:00:01.000000+00:00",
        attributes={"input.value": "seeded"},
    )
    record["attributes"] = None

    with pytest.raises(SpanCaptureError) as error:
        _span_attributes(record)

    assert str(error.value) == (
        "captured span 'cogniverse.query_enhancement' has attributes of type "
        "NoneType, expected a mapping"
    )
