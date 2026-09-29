"""PhoenixDatasetStore keeps its append contract against a real Phoenix.

``create_dataset`` on a name that already exists appends its rows, including
when several writers create the same fresh name at once. ``append_to_dataset``
stores every row it is given, including one identical to a row already stored.
Row values come back as the CSV strings they were written as.
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pandas as pd
import pytest

from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.requires_docker,
]

_QA_KEYS = {"input_keys": ["question"], "output_keys": ["answer"]}


def _qa(question: str, answer: str) -> pd.DataFrame:
    return pd.DataFrame({"question": [question], "answer": [answer]})


def _rows(frame: pd.DataFrame) -> list[tuple[dict, dict]]:
    return [(row["input"], row["output"]) for _, row in frame.iterrows()]


@pytest.fixture
def store(phoenix_container) -> PhoenixDatasetStore:
    return PhoenixDatasetStore(phoenix_container["http_endpoint"])


@pytest.mark.asyncio
async def test_create_on_existing_name_appends_rows(store):
    name = f"create-append-{uuid.uuid4().hex[:8]}"

    first_id = await store.create_dataset(name, _qa("q1", "a1"), metadata=_QA_KEYS)
    second_id = await store.create_dataset(name, _qa("q2", "a2"), metadata=_QA_KEYS)

    assert second_id == first_id
    assert _rows(await store.get_dataset(name)) == [
        ({"question": "q1"}, {"answer": "a1"}),
        ({"question": "q2"}, {"answer": "a2"}),
    ]


@pytest.mark.asyncio
async def test_append_keeps_a_row_identical_to_a_stored_one(store):
    name = f"append-duplicate-{uuid.uuid4().hex[:8]}"

    await store.create_dataset(name, _qa("q1", "a1"), metadata=_QA_KEYS)
    await store.append_to_dataset(name, _qa("q1", "a1"), metadata=_QA_KEYS)

    assert _rows(await store.get_dataset(name)) == [
        ({"question": "q1"}, {"answer": "a1"}),
        ({"question": "q1"}, {"answer": "a1"}),
    ]


@pytest.mark.asyncio
async def test_create_repeating_an_earlier_row_appends_it_last(store):
    """A last-row-wins log that returns to an earlier value must end on it."""
    name = f"create-repeat-{uuid.uuid4().hex[:8]}"
    keys = {"input_keys": ["payload"], "output_keys": []}

    for payload in ('{"score": 0.8}', '{"score": 0.7}', '{"score": 0.8}'):
        await store.create_dataset(
            name, pd.DataFrame({"payload": [payload]}), metadata=keys
        )

    frame = await store.get_dataset(name)
    assert [row["input"]["payload"] for _, row in frame.iterrows()] == [
        '{"score": 0.8}',
        '{"score": 0.7}',
        '{"score": 0.8}',
    ]


@pytest.mark.asyncio
async def test_row_values_round_trip_as_written_csv_strings(store):
    """JSON-object text stays text, an empty cell stays an empty string, and a
    number comes back as its CSV rendering — also under ``metadata.``-prefixed
    column names and with keys inferred from the frame."""
    name = f"csv-values-{uuid.uuid4().hex[:8]}"
    frame = pd.DataFrame(
        {
            "item_id": ["i1"],
            "metadata.record_json": ['{"k": 1, "nested": {"x": [1, 2]}}'],
            "label": [""],
            "score": [0.5],
        }
    )

    await store.create_dataset(name, frame)
    await store.append_to_dataset(name, frame)

    loaded = await store.get_dataset(name)
    expected = {
        "item_id": "i1",
        "metadata.record_json": '{"k": 1, "nested": {"x": [1, 2]}}',
        "label": "",
        "score": "0.5",
    }
    assert [row["input"] for _, row in loaded.iterrows()] == [expected, expected]
    assert [row["output"] for _, row in loaded.iterrows()] == [{}, {}]
    assert [row["metadata"] for _, row in loaded.iterrows()] == [{}, {}]


@pytest.mark.asyncio
async def test_concurrent_creates_of_a_fresh_name_keep_every_row(store):
    name = f"create-race-{uuid.uuid4().hex[:8]}"
    writers = 8

    ids = await asyncio.gather(
        *[
            store.create_dataset(name, _qa(f"q{i}", f"a{i}"), metadata=_QA_KEYS)
            for i in range(writers)
        ]
    )

    assert len(set(ids)) == 1
    assert sorted(
        (inp["question"], out["answer"])
        for inp, out in _rows(await store.get_dataset(name))
    ) == [(f"q{i}", f"a{i}") for i in range(writers)]


@pytest.mark.asyncio
async def test_create_against_an_unreachable_phoenix_raises():
    store = PhoenixDatasetStore("http://127.0.0.1:1")

    with pytest.raises(httpx.ConnectError):
        await store.create_dataset("unreachable", _qa("q", "a"), metadata=_QA_KEYS)
