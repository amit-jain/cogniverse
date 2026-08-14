import asyncio
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

pytestmark = pytest.mark.unit


class _Breaker:
    async def acall(self, operation, *args, **kwargs):
        return await operation(*args, **kwargs)


def _store(spans):
    store = PhoenixTraceStore.__new__(PhoenixTraceStore)
    store.http_endpoint = "http://phoenix.test"
    store._breaker = _Breaker()
    store._get_client = lambda: SimpleNamespace(spans=spans)
    return store


@pytest.mark.asyncio
async def test_get_all_spans_returns_records_beyond_one_page():
    calls = []

    class Spans:
        async def get_spans(self, **kwargs):
            calls.append(kwargs)
            return [
                {
                    "name": "approval_batch",
                    "context": {
                        "trace_id": f"trace-{index}",
                        "span_id": f"span-{index}",
                    },
                    "attributes": {
                        "batch_id": f"batch-{index}",
                        "metadata.agent_type": "profile_selection",
                    },
                    "start_time": "2026-08-04T00:00:00+00:00",
                    "end_time": "2026-08-04T00:00:01+00:00",
                    "status_code": "OK",
                    "span_kind": "CHAIN",
                }
                for index in range(1_005)
            ]

    frame = await _store(Spans()).get_all_spans(
        project="cogniverse-acme-approval",
        filters={"name": ["approval_batch", "approval_item"]},
    )

    assert len(frame) == 1_005
    assert frame["attributes.batch_id"].tolist() == [
        f"batch-{index}" for index in range(1_005)
    ]
    assert frame.iloc[0]["attributes.metadata"] == {"agent_type": "profile_selection"}
    assert frame.iloc[0]["attributes.metadata.agent_type"] == "profile_selection"
    assert frame["start_time"].dtype == pd.DatetimeTZDtype(tz="UTC", unit="ns")
    assert frame["end_time"].dtype == pd.DatetimeTZDtype(tz="UTC", unit="ns")
    assert frame.iloc[0]["start_time"] == pd.Timestamp("2026-08-04T00:00:00+00:00")
    assert frame.iloc[0]["end_time"] == pd.Timestamp("2026-08-04T00:00:01+00:00")
    assert calls == [
        {
            "project_identifier": "cogniverse-acme-approval",
            "start_time": None,
            "end_time": None,
            "name": ["approval_batch", "approval_item"],
            "limit": 2_147_483_647,
            "timeout": 120,
        }
    ]


@pytest.mark.asyncio
async def test_get_all_spans_parses_mixed_iso8601_precision_and_offsets():
    class Spans:
        async def get_spans(self, **kwargs):
            return [
                {
                    "name": "approval_batch",
                    "context": {
                        "trace_id": "trace-0",
                        "span_id": "span-0",
                    },
                    "attributes": {"batch_id": "batch-0"},
                    "start_time": "2026-08-14T18:55:30.123456+00:00",
                    "end_time": "2026-08-14T18:55:31.123456+00:00",
                    "status_code": "OK",
                    "span_kind": "CHAIN",
                },
                {
                    "name": "approval_batch",
                    "context": {
                        "trace_id": "trace-1",
                        "span_id": "span-1",
                    },
                    "attributes": {"batch_id": "batch-1"},
                    "start_time": "2026-08-14T18:55:30+05:30",
                    "end_time": "2026-08-14T18:55:31+05:30",
                    "status_code": "OK",
                    "span_kind": "CHAIN",
                },
            ]

    frame = await _store(Spans()).get_all_spans(project="cogniverse-acme")

    assert frame["start_time"].tolist() == [
        pd.Timestamp("2026-08-14T18:55:30.123456+00:00"),
        pd.Timestamp("2026-08-14T13:25:30+00:00"),
    ]
    assert frame["end_time"].tolist() == [
        pd.Timestamp("2026-08-14T18:55:31.123456+00:00"),
        pd.Timestamp("2026-08-14T13:25:31+00:00"),
    ]


@pytest.mark.asyncio
async def test_get_all_spans_keeps_concurrent_queries_isolated():
    entered = 0
    both_entered = asyncio.Event()

    class Spans:
        async def get_spans(self, **kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            project = kwargs["project_identifier"]
            return [
                {
                    "name": "approval_batch",
                    "context": {"trace_id": project, "span_id": project},
                    "attributes": {"batch_id": project},
                    "start_time": "2026-08-04T00:00:00+00:00",
                    "end_time": "2026-08-04T00:00:01+00:00",
                    "status_code": "OK",
                    "span_kind": "CHAIN",
                }
            ]

    store = _store(Spans())
    alpha, beta = await asyncio.gather(
        store.get_all_spans(project="alpha", filters={"name": "approval_batch"}),
        store.get_all_spans(project="beta", filters={"name": "approval_batch"}),
    )

    assert alpha.iloc[0]["attributes.batch_id"] == "alpha"
    assert beta.iloc[0]["attributes.batch_id"] == "beta"


@pytest.mark.asyncio
async def test_get_all_spans_surfaces_page_failure_with_project_context():
    class Spans:
        async def get_spans(self, **kwargs):
            raise TimeoutError("cursor page timed out")

    with pytest.raises(
        RuntimeError,
        match="Failed to query every span from Phoenix project cogniverse-acme-approval",
    ) as captured:
        await _store(Spans()).get_all_spans(project="cogniverse-acme-approval")

    assert isinstance(captured.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_get_all_spans_returns_empty_for_explicit_missing_project():
    class Spans:
        async def get_spans(self, **kwargs):
            request = httpx.Request(
                "GET", "http://phoenix.test/v1/projects/missing/spans"
            )
            response = httpx.Response(
                404,
                request=request,
                text="Project with name missing not found",
            )
            raise httpx.HTTPStatusError(
                "project not found", request=request, response=response
            )

    frame = await _store(Spans()).get_all_spans(project="missing")

    assert frame.empty


@pytest.mark.asyncio
async def test_get_all_spans_rejects_unrelated_404_as_boundary_failure():
    class Spans:
        async def get_spans(self, **kwargs):
            request = httpx.Request("GET", "http://proxy.test/v1/projects/p/spans")
            response = httpx.Response(
                404, request=request, json={"detail": "route not found"}
            )
            raise httpx.HTTPStatusError(
                "proxy route not found", request=request, response=response
            )

    with pytest.raises(RuntimeError) as captured:
        await _store(Spans()).get_all_spans(project="p")

    assert isinstance(captured.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.asyncio
async def test_get_all_spans_rejects_invalid_timestamp_with_project_context():
    class Spans:
        async def get_spans(self, **kwargs):
            return [
                {
                    "name": "cogniverse.orchestration",
                    "context": {"trace_id": "trace-a", "span_id": "span-a"},
                    "attributes": {},
                    "start_time": "not-a-timestamp",
                    "end_time": "2026-08-04T00:00:01+00:00",
                    "status_code": "OK",
                    "span_kind": "CHAIN",
                }
            ]

    with pytest.raises(
        RuntimeError,
        match=(
            "Phoenix project cogniverse-acme returned an invalid "
            "ISO-8601 timestamp in start_time: Time data not-a-timestamp is not "
            "ISO8601 format, at position 0"
        ),
    ) as captured:
        await _store(Spans()).get_all_spans(project="cogniverse-acme")

    assert isinstance(captured.value.__cause__, ValueError)
