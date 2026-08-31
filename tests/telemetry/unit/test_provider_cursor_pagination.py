import ast
import asyncio
from datetime import datetime, timedelta, timezone
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


def _cursor_store(client):
    store = PhoenixTraceStore.__new__(PhoenixTraceStore)
    store.http_endpoint = "http://phoenix.test"
    store._breaker = _Breaker()
    store._get_client = lambda: SimpleNamespace(_client=client)
    return store


class _CursorResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _CursorClient:
    def __init__(self, spans):
        self._spans = list(spans)
        self.calls = []

    async def get(self, **kwargs):
        params = kwargs["params"]
        cursor = params.get("cursor")
        offset = int(cursor) if cursor is not None else 0
        limit = params["limit"]
        page = self._spans[offset : offset + limit]
        next_cursor = (
            str(offset + len(page)) if offset + len(page) < len(self._spans) else None
        )
        self.calls.append(
            {
                "url": kwargs["url"],
                "params": dict(params),
                "timeout": kwargs["timeout"],
                "page_span_ids": [span["context"]["span_id"] for span in page],
            }
        )
        return _CursorResponse({"data": page, "next_cursor": next_cursor})


class _ProjectedClient:
    def __init__(self, spans):
        self._spans = list(spans)
        self.calls = []

    @staticmethod
    def _excluded_span_ids(condition):
        if not condition or "span_id not in " not in condition:
            return set()
        excluded_clause = condition.split("span_id not in ", 1)[1]
        return set(ast.literal_eval(excluded_clause))

    async def get_spans_dataframe(self, **kwargs):
        query_dict = kwargs["query"].to_dict()
        condition = query_dict.get("filter", {}).get("condition")
        excluded_span_ids = self._excluded_span_ids(condition)
        selected_columns = tuple(query_dict["select"].keys())
        start_time = kwargs["start_time"]
        end_time = kwargs["end_time"]

        rows = [
            span
            for span in self._spans
            if (start_time is None or start_time <= span["start_time"])
            and (end_time is None or span["start_time"] <= end_time)
            and span["context.span_id"] not in excluded_span_ids
        ]
        page_rows = rows[: kwargs["limit"]]

        self.calls.append(
            {
                "project_identifier": kwargs["project_identifier"],
                "start_time": kwargs["start_time"],
                "end_time": kwargs["end_time"],
                "limit": kwargs["limit"],
                "condition": condition,
                "selected_columns": selected_columns,
                "page_span_ids": [span["context.span_id"] for span in page_rows],
            }
        )

        if not page_rows:
            frame = pd.DataFrame(
                columns=["context.span_id", *selected_columns]
            ).set_index("context.span_id")
            return frame.reindex(columns=selected_columns)

        frame = pd.DataFrame(page_rows).set_index("context.span_id")
        return frame.reindex(columns=selected_columns)


@pytest.mark.asyncio
async def test_get_spans_forwards_requested_columns():
    calls = []

    class Spans:
        async def get_spans_dataframe(self, **kwargs):
            calls.append(
                {
                    "project_identifier": kwargs["project_identifier"],
                    "start_time": kwargs["start_time"],
                    "end_time": kwargs["end_time"],
                    "limit": kwargs["limit"],
                    "timeout": kwargs["timeout"],
                    "query": kwargs["query"].to_dict(),
                }
            )
            return pd.DataFrame(
                [
                    {
                        "start_time": "2026-08-04T00:00:00+00:00",
                        "end_time": "2026-08-04T00:00:01+00:00",
                        "status_code": "OK",
                    }
                ]
            )

    frame = await _store(Spans()).get_spans(
        project="cogniverse-acme-approval",
        columns=("start_time", "end_time", "status_code"),
        limit=7,
    )

    assert len(frame) == 1
    assert calls == [
        {
            "project_identifier": "cogniverse-acme-approval",
            "start_time": None,
            "end_time": None,
            "limit": 7,
            "timeout": 120,
            "query": {
                "select": {
                    "start_time": {"key": "start_time"},
                    "end_time": {"key": "end_time"},
                    "status_code": {"key": "status_code"},
                },
                "index": {"key": "context.span_id"},
            },
        }
    ]


@pytest.mark.asyncio
async def test_get_all_spans_returns_records_beyond_one_page():
    spans = [
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

    client = _CursorClient(spans)
    frame = await _cursor_store(client).get_all_spans(
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
    assert len(client.calls) == 2
    assert [call["params"]["limit"] for call in client.calls] == [1000, 1000]
    assert [len(call["page_span_ids"]) for call in client.calls] == [1000, 5]
    assert [call["params"].get("cursor") for call in client.calls] == [None, "1000"]
    assert [call["params"]["name"] for call in client.calls] == [
        ["approval_batch", "approval_item"],
        ["approval_batch", "approval_item"],
    ]
    assert [span_id for call in client.calls for span_id in call["page_span_ids"]] == [
        f"span-{index}" for index in range(1_005)
    ]


@pytest.mark.asyncio
async def test_iter_spans_uses_projected_window_pages_without_duplicates():
    base_time = datetime(2026, 8, 4, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "name": "approval_batch",
            "context.span_id": "span-4",
            "context.trace_id": "trace-4",
            "start_time": base_time + timedelta(hours=1, minutes=5),
            "end_time": base_time + timedelta(hours=1, minutes=5, seconds=1),
            "status_code": "OK",
        },
        {
            "name": "approval_batch",
            "context.span_id": "span-3",
            "context.trace_id": "trace-3",
            "start_time": base_time + timedelta(minutes=25),
            "end_time": base_time + timedelta(minutes=25, seconds=1),
            "status_code": "OK",
        },
        {
            "name": "approval_batch",
            "context.span_id": "span-0",
            "context.trace_id": "trace-0",
            "start_time": base_time + timedelta(minutes=5),
            "end_time": base_time + timedelta(minutes=5, seconds=1),
            "status_code": "OK",
        },
        {
            "name": "approval_batch",
            "context.span_id": "span-5",
            "context.trace_id": "trace-5",
            "start_time": base_time + timedelta(hours=1),
            "end_time": base_time + timedelta(hours=1, seconds=1),
            "status_code": "WARN",
        },
        {
            "name": "approval_batch",
            "context.span_id": "span-1",
            "context.trace_id": "trace-1",
            "start_time": base_time + timedelta(minutes=15),
            "end_time": base_time + timedelta(minutes=15, seconds=1),
            "status_code": "ERROR",
        },
        {
            "name": "approval_batch",
            "context.span_id": "span-2",
            "context.trace_id": "trace-2",
            "start_time": base_time + timedelta(minutes=15),
            "end_time": base_time + timedelta(minutes=15, seconds=2),
            "status_code": "OK",
        },
    ]

    client = _ProjectedClient(rows)
    store = _store(client)

    pages = []
    async for frame in store.iter_spans(
        project="cogniverse-acme-approval",
        start_time=base_time,
        end_time=base_time + timedelta(hours=2),
        filters={"name": "approval_batch"},
        page_size=2,
        columns=("start_time", "end_time", "status_code"),
    ):
        pages.append(frame)

    combined = pd.concat(pages, ignore_index=True)
    root_window_end = base_time + timedelta(hours=2)

    assert list(combined.columns) == ["start_time", "end_time", "status_code"]
    assert len(combined) == len(rows)
    assert {tuple(row) for row in combined.itertuples(index=False, name=None)} == {
        (
            base_time + timedelta(minutes=5),
            base_time + timedelta(minutes=5, seconds=1),
            "OK",
        ),
        (
            base_time + timedelta(minutes=15),
            base_time + timedelta(minutes=15, seconds=1),
            "ERROR",
        ),
        (
            base_time + timedelta(minutes=15),
            base_time + timedelta(minutes=15, seconds=2),
            "OK",
        ),
        (
            base_time + timedelta(minutes=25),
            base_time + timedelta(minutes=25, seconds=1),
            "OK",
        ),
        (
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=1, seconds=1),
            "WARN",
        ),
        (
            base_time + timedelta(hours=1, minutes=5),
            base_time + timedelta(hours=1, minutes=5, seconds=1),
            "OK",
        ),
    }
    assert len(client.calls) >= 4
    assert [call["project_identifier"] for call in client.calls] == [
        "cogniverse-acme-approval",
    ] * len(client.calls)
    assert all(
        call["selected_columns"] == ("start_time", "end_time", "status_code")
        for call in client.calls
    )
    assert any(call["end_time"] < root_window_end for call in client.calls)
    assert client.calls[0]["start_time"] == base_time
    assert client.calls[0]["end_time"] == root_window_end
    assert {span_id for call in client.calls for span_id in call["page_span_ids"]} == {
        f"span-{index}" for index in range(len(rows))
    }
    assert client.calls[0]["condition"] == "name == 'approval_batch'"


@pytest.mark.asyncio
async def test_get_all_spans_parses_mixed_iso8601_precision_and_offsets():
    client = _CursorClient(
        [
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
    )

    frame = await _cursor_store(client).get_all_spans(project="cogniverse-acme")

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
        async def get(self, **kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            project = kwargs["url"].split("/")[2]
            return _CursorResponse(
                {
                    "data": [
                        {
                            "name": "approval_batch",
                            "context": {"trace_id": project, "span_id": project},
                            "attributes": {"batch_id": project},
                            "start_time": "2026-08-04T00:00:00+00:00",
                            "end_time": "2026-08-04T00:00:01+00:00",
                            "status_code": "OK",
                            "span_kind": "CHAIN",
                        }
                    ],
                    "next_cursor": None,
                }
            )

    store = _cursor_store(Spans())
    alpha, beta = await asyncio.gather(
        store.get_all_spans(project="alpha", filters={"name": "approval_batch"}),
        store.get_all_spans(project="beta", filters={"name": "approval_batch"}),
    )

    assert alpha.iloc[0]["attributes.batch_id"] == "alpha"
    assert beta.iloc[0]["attributes.batch_id"] == "beta"


@pytest.mark.asyncio
async def test_get_all_spans_surfaces_page_failure_with_project_context():
    class Spans:
        async def get(self, **kwargs):
            raise TimeoutError("cursor page timed out")

    with pytest.raises(
        RuntimeError,
        match="Failed to query every span from Phoenix project cogniverse-acme-approval",
    ) as captured:
        await _cursor_store(Spans()).get_all_spans(project="cogniverse-acme-approval")

    assert isinstance(captured.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_get_all_spans_returns_empty_for_explicit_missing_project():
    class Spans:
        async def get(self, **kwargs):
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

    frame = await _cursor_store(Spans()).get_all_spans(project="missing")

    assert frame.empty


@pytest.mark.asyncio
async def test_get_all_spans_rejects_unrelated_404_as_boundary_failure():
    class Spans:
        async def get(self, **kwargs):
            request = httpx.Request("GET", "http://proxy.test/v1/projects/p/spans")
            response = httpx.Response(
                404, request=request, json={"detail": "route not found"}
            )
            raise httpx.HTTPStatusError(
                "proxy route not found", request=request, response=response
            )

    with pytest.raises(RuntimeError) as captured:
        await _cursor_store(Spans()).get_all_spans(project="p")

    assert isinstance(captured.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.asyncio
async def test_get_all_spans_rejects_invalid_timestamp_with_project_context():
    class Spans:
        async def get(self, **kwargs):
            return _CursorResponse(
                {
                    "data": [
                        {
                            "name": "cogniverse.orchestration",
                            "context": {"trace_id": "trace-a", "span_id": "span-a"},
                            "attributes": {},
                            "start_time": "not-a-timestamp",
                            "end_time": "2026-08-04T00:00:01+00:00",
                            "status_code": "OK",
                            "span_kind": "CHAIN",
                        }
                    ],
                    "next_cursor": None,
                }
            )

    with pytest.raises(
        RuntimeError,
        match=(
            "Phoenix project cogniverse-acme returned an invalid "
            "ISO-8601 timestamp in start_time: Time data not-a-timestamp is not "
            "ISO8601 format, at position 0"
        ),
    ) as captured:
        await _cursor_store(Spans()).get_all_spans(project="cogniverse-acme")

    assert isinstance(captured.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_projected_frames_survive_phoenix_named_index():
    """Real get_spans_dataframe frames are INDEXED BY context.span_id while
    also carrying a context.span_id column; reset_index() must not collide."""
    import pandas as pd

    calls = []

    class _RealShapedClient:
        def __init__(self):
            self.spans = self

        async def get_spans_dataframe(self, **kwargs):
            calls.append(kwargs)
            frame = pd.DataFrame(
                {
                    "context.span_id": ["s1", "s2"],
                    "start_time": [
                        "2026-08-04T00:00:00+00:00",
                        "2026-08-04T00:01:00+00:00",
                    ],
                    "end_time": [
                        "2026-08-04T00:00:01+00:00",
                        "2026-08-04T00:01:01+00:00",
                    ],
                }
            )
            frame.index = pd.Index(frame["context.span_id"], name="context.span_id")
            return frame

    store = _store([])
    client = _RealShapedClient()
    store._get_client = lambda: client

    frames = []
    async for frame in store.iter_spans(
        "proj",
        start_time=datetime(2026, 8, 4, tzinfo=timezone.utc),
        end_time=datetime(2026, 8, 5, tzinfo=timezone.utc),
        columns=("start_time", "end_time", "context.span_id"),
    ):
        frames.append(frame)

    assert len(frames) == 1
    assert list(frames[0].columns) == [
        "start_time",
        "end_time",
        "context.span_id",
    ]
    assert list(frames[0]["context.span_id"]) == ["s1", "s2"]
