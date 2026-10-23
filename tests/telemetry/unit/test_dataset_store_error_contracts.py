"""PhoenixDatasetStore must classify errors by TYPE/STATUS, not by substring
matching on messages that embed the request URL or server body.

A dataset is found through Phoenix's name-filtered listing: an empty listing
is a missing dataset, a refused listing is an outage, and a 404 counts as a
deleted dataset only when Phoenix's own detail names that dataset's id. A
duplicate-name conflict is HTTP 409. None of these may be confused with each
other — the failure mode this pins is an outage being read as "no dataset"
(silently disabling baseline reads) or a failed create being read as a
successful append.
"""

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from cogniverse_foundation.telemetry.providers.base import (
    DatasetNotFoundError,
    DatasetStoreUnavailableError,
)

pytestmark = pytest.mark.unit


_ENDPOINT = "http://phoenix:6006"
_DATASET_ID = "RGF0YXNldDox"


def _store():
    from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

    return PhoenixDatasetStore(http_endpoint=_ENDPOINT)


def _http_error(
    status: int, url: str = "http://phoenix:6006/v1/datasets", **body
) -> httpx.HTTPStatusError:
    """A realistic httpx error whose message matches raise_for_status() —
    i.e. it embeds the status ('404 Not Found') and the URL, exactly the text
    the old substring sniff misread."""
    req = httpx.Request("GET", url)
    resp = httpx.Response(status, request=req, **(body or {"text": "backend detail"}))
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return e


def _listing(*, status: int = 200, example_count: int | None = None) -> httpx.Response:
    """Phoenix's answer to ``GET /v1/datasets?name=ds1``: one row when
    ``example_count`` is given, none otherwise, or a refusal."""
    request = httpx.Request("GET", f"{_ENDPOINT}/v1/datasets")
    if status != 200:
        return httpx.Response(status, request=request, text="backend detail")
    rows = (
        []
        if example_count is None
        else [
            {
                "id": _DATASET_ID,
                "name": "ds1",
                "example_count": example_count,
                "created_at": "2026-10-23T00:00:00+00:00",
                "description": "",
                "metadata": {},
            }
        ]
    )
    return httpx.Response(
        200, request=request, json={"data": rows, "next_cursor": None}
    )


def _listed(response: httpx.Response):
    return patch(
        "cogniverse_telemetry_phoenix.provider.httpx.get", return_value=response
    )


class TestGetDataset:
    @pytest.mark.asyncio
    async def test_genuine_not_found_maps_to_subclass(self):
        with _listed(_listing()), patch("phoenix.client.Client") as client_cls:
            with pytest.raises(DatasetNotFoundError) as excinfo:
                await _store().get_dataset("ds1")
        assert str(excinfo.value) == "Dataset not found: ds1"
        assert client_cls.call_count == 0

    @pytest.mark.asyncio
    async def test_http_404_from_outage_is_not_read_as_missing(self):
        """A 404 whose text embeds a URL/port containing '404', or a proxy 404
        during an outage, must surface as an error — never DatasetNotFoundError."""
        with _listed(_listing(status=404)):
            with pytest.raises(DatasetStoreUnavailableError) as excinfo:
                await _store().get_dataset("ds1")
        assert excinfo.value.dataset == "ds1"
        assert excinfo.value.endpoint == _ENDPOINT
        assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
        assert excinfo.value.__cause__.response.status_code == 404

    @pytest.mark.asyncio
    async def test_503_outage_raises_not_missing(self):
        with _listed(_listing(status=503)):
            with pytest.raises(DatasetStoreUnavailableError) as excinfo:
                await _store().get_dataset("ds1")
        assert excinfo.value.dataset == "ds1"
        assert excinfo.value.endpoint == _ENDPOINT
        assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
        assert excinfo.value.__cause__.response.status_code == 503

    @pytest.mark.asyncio
    async def test_name_containing_404_still_raises_on_outage(self):
        """Dataset name 'quality-baseline-20260404' + a 500 must not be read
        as not-found just because '404' appears in the name/message."""
        with _listed(_listing(status=500)):
            with pytest.raises(DatasetStoreUnavailableError) as excinfo:
                await _store().get_dataset("quality-baseline-20260404")
        assert excinfo.value.dataset == "quality-baseline-20260404"
        assert excinfo.value.endpoint == _ENDPOINT
        assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
        assert excinfo.value.__cause__.response.status_code == 500

    @pytest.mark.asyncio
    async def test_a_dataset_holding_no_rows_reads_empty_without_describing_it(self):
        """Phoenix answers 500 to a description of a dataset with no version,
        so a listed dataset with no examples is answered from the listing."""
        with (
            _listed(_listing(example_count=0)),
            patch("phoenix.client.Client") as client_cls,
        ):
            df = await _store().get_dataset("ds1")
        assert (list(df.columns), df.index.name, len(df)) == (
            ["input", "output", "metadata"],
            "example_id",
            0,
        )
        assert client_cls.call_count == 0

    @pytest.mark.asyncio
    async def test_rows_are_read_from_the_listed_dataset_id(self):
        frame = pd.DataFrame({"input": [{"q": "a"}]})
        client = MagicMock()
        client.datasets.get_dataset.return_value.to_dataframe.return_value = frame
        with (
            _listed(_listing(example_count=1)),
            patch("phoenix.client.Client", return_value=client),
        ):
            df = await _store().get_dataset("ds1")
        assert df is frame
        assert client.datasets.get_dataset.call_args.kwargs == {
            "dataset": _DATASET_ID,
            "timeout": 120,
        }

    @pytest.mark.asyncio
    async def test_phoenix_404_naming_the_dataset_reads_as_deleted(self):
        """Deleted between the listing and the read: Phoenix's own 404 names
        the dataset id."""
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(
            404, text=f"Dataset with ID {_DATASET_ID} not found"
        )
        with (
            _listed(_listing(example_count=1)),
            patch("phoenix.client.Client", return_value=client),
        ):
            with pytest.raises(DatasetNotFoundError) as excinfo:
                await _store().get_dataset("ds1")
        assert str(excinfo.value) == "Dataset not found: ds1"
        assert excinfo.value.__cause__.response.status_code == 404

    @pytest.mark.asyncio
    async def test_a_404_not_naming_the_dataset_is_an_outage(self):
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(
            404, text=f"Cannot GET /v1/datasets/{_DATASET_ID}"
        )
        with (
            _listed(_listing(example_count=1)),
            patch("phoenix.client.Client", return_value=client),
        ):
            with pytest.raises(DatasetStoreUnavailableError) as excinfo:
                await _store().get_dataset("ds1")
        assert (excinfo.value.dataset, excinfo.value.endpoint) == ("ds1", _ENDPOINT)
        assert excinfo.value.__cause__.response.status_code == 404


class TestAppendToDataset:
    @pytest.mark.asyncio
    async def test_missing_raises_dataset_not_found_subclass(self):
        with (
            _listed(_listing()),
            patch("cogniverse_telemetry_phoenix.provider.httpx.post") as post,
        ):
            with pytest.raises(DatasetNotFoundError):
                await _store().append_to_dataset("ds1", pd.DataFrame([{"a": 1}]))
        assert post.call_count == 0

    @pytest.mark.asyncio
    async def test_outage_during_lookup_raises_not_missing(self):
        with (
            _listed(_listing(status=503)),
            patch("cogniverse_telemetry_phoenix.provider.httpx.post") as post,
        ):
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await _store().append_to_dataset("ds1", pd.DataFrame([{"a": 1}]))
        assert excinfo.value.response.status_code == 503
        assert post.call_count == 0


def _upload_response(status: int, **kwargs) -> httpx.Response:
    request = httpx.Request("POST", f"{_ENDPOINT}/v1/datasets/upload")
    return httpx.Response(status, request=request, **kwargs)


def _posted(post) -> list[dict]:
    return [call.kwargs["json"] for call in post.call_args_list]


class TestCreateDataset:
    @pytest.mark.asyncio
    async def test_409_conflict_appends_new_version(self):
        created = {"data": {"dataset_id": "ds1", "version_id": "v2"}}
        with patch(
            "cogniverse_telemetry_phoenix.provider.httpx.post",
            side_effect=[
                _upload_response(409, text="already exists"),
                _upload_response(200, json=created),
            ],
        ) as post:
            result = await _store().create_dataset("ds1", pd.DataFrame([{"a": 1}]))
        assert result == "ds1"
        assert [body["action"] for body in _posted(post)] == ["create", "append"]

    @pytest.mark.asyncio
    async def test_500_with_already_exists_body_is_not_appended(self):
        """A non-conflict 500 whose body text merely contains 'already exists'
        must fail loudly, never be silently rerouted to append."""
        with patch(
            "cogniverse_telemetry_phoenix.provider.httpx.post",
            side_effect=[_upload_response(500, text="WAL says already exists")],
        ) as post:
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await _store().create_dataset("ds1", pd.DataFrame([{"a": 1}]))
        assert excinfo.value.response.status_code == 500
        assert [body["action"] for body in _posted(post)] == ["create"]

    @pytest.mark.asyncio
    async def test_append_failure_after_conflict_raises(self):
        """The name exists, and the append that follows fails: the failure
        surfaces instead of reading as a created dataset."""
        with patch(
            "cogniverse_telemetry_phoenix.provider.httpx.post",
            side_effect=[
                _upload_response(409, text="already exists"),
                _upload_response(503, text="unavailable"),
            ],
        ) as post:
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await _store().create_dataset("ds1", pd.DataFrame([{"a": 1}]))
        assert excinfo.value.response.status_code == 503
        assert [body["action"] for body in _posted(post)] == ["create", "append"]

    @pytest.mark.asyncio
    async def test_keys_split_at_the_first_output_named_column(self):
        """Without explicit keys, columns before the first output-named one
        are inputs, it is the output, and the rest are metadata — every value
        sent as its CSV text."""
        created = {"data": {"dataset_id": "ds1", "version_id": "v1"}}
        frame = pd.DataFrame(
            [{"question": "q", "turns": 2, "answer": "a", "source": "", "ok": True}]
        )
        with patch(
            "cogniverse_telemetry_phoenix.provider.httpx.post",
            side_effect=[_upload_response(200, json=created)],
        ) as post:
            await _store().create_dataset("ds1", frame)
        [body] = _posted(post)
        assert (body["inputs"], body["outputs"], body["metadata"]) == (
            [{"question": "q", "turns": "2"}],
            [{"answer": "a"}],
            [{"source": "", "ok": "True"}],
        )
        assert len(body["example_ids"]) == 1
        assert len(body["example_ids"][0]) == 32


class TestDeleteDataset:
    @pytest.mark.asyncio
    async def test_genuine_missing_returns_false(self):
        with (
            _listed(_listing()),
            patch("cogniverse_telemetry_phoenix.provider.httpx.delete") as delete,
        ):
            assert await _store().delete_dataset("ds1") is False
        assert delete.call_count == 0

    @pytest.mark.asyncio
    async def test_outage_raises_not_false(self):
        """A dead/hung backend must not be read as 'nothing to delete' — that
        turns replace-dataset's delete-then-create into a silent append."""
        with (
            _listed(_listing(status=503)),
            patch("cogniverse_telemetry_phoenix.provider.httpx.delete") as delete,
        ):
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await _store().delete_dataset("ds1")
        assert excinfo.value.response.status_code == 503
        assert delete.call_count == 0

    @pytest.mark.asyncio
    async def test_the_listed_dataset_id_is_deleted(self):
        request = httpx.Request("DELETE", f"{_ENDPOINT}/v1/datasets/{_DATASET_ID}")
        with (
            _listed(_listing(example_count=0)),
            patch(
                "cogniverse_telemetry_phoenix.provider.httpx.delete",
                return_value=httpx.Response(204, request=request),
            ) as delete,
        ):
            assert await _store().delete_dataset("ds1") is True
        assert delete.call_args.args == (f"{_ENDPOINT}/v1/datasets/{_DATASET_ID}",)
