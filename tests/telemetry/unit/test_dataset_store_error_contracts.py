"""PhoenixDatasetStore must classify errors by TYPE/STATUS, not by substring
matching on messages that embed the request URL or server body.

A genuine missing dataset is a plain ValueError raised by phoenix's own name
resolution after a successful HTTP call; an outage is an httpx error. A
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


def _store():
    from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

    return PhoenixDatasetStore(http_endpoint=_ENDPOINT)


def _http_error(
    status: int, url: str = "http://phoenix:6006/v1/datasets"
) -> httpx.HTTPStatusError:
    """A realistic httpx error whose message matches raise_for_status() —
    i.e. it embeds the status ('404 Not Found') and the URL, exactly the text
    the old substring sniff misread."""
    req = httpx.Request("GET", url)
    resp = httpx.Response(status, request=req, text="backend detail")
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return e


class TestGetDataset:
    @pytest.mark.asyncio
    async def test_genuine_not_found_maps_to_subclass(self):
        client = MagicMock()
        client.datasets.get_dataset.side_effect = ValueError("Dataset not found: ds1")
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(DatasetNotFoundError):
                await _store().get_dataset("ds1")

    @pytest.mark.asyncio
    async def test_http_404_from_outage_is_not_read_as_missing(self):
        """A 404 whose text embeds a URL/port containing '404', or a proxy 404
        during an outage, must surface as an error — never DatasetNotFoundError."""
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(404)
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(DatasetStoreUnavailableError) as excinfo:
                await _store().get_dataset("ds1")
        assert excinfo.value.dataset == "ds1"
        assert excinfo.value.endpoint == _ENDPOINT
        assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
        assert excinfo.value.__cause__.response.status_code == 404

    @pytest.mark.asyncio
    async def test_503_outage_raises_not_missing(self):
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(503)
        with patch("phoenix.client.Client", return_value=client):
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
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(500)
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(DatasetStoreUnavailableError) as excinfo:
                await _store().get_dataset("quality-baseline-20260404")
        assert excinfo.value.dataset == "quality-baseline-20260404"
        assert excinfo.value.endpoint == _ENDPOINT
        assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
        assert excinfo.value.__cause__.response.status_code == 500


class TestAppendToDataset:
    @pytest.mark.asyncio
    async def test_missing_raises_dataset_not_found_subclass(self):
        client = MagicMock()
        client.datasets.get_dataset.side_effect = ValueError("Dataset not found: ds1")
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(DatasetNotFoundError):
                await _store().append_to_dataset("ds1", pd.DataFrame([{"a": 1}]))

    @pytest.mark.asyncio
    async def test_outage_during_lookup_raises_not_missing(self):
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(503)
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(httpx.HTTPStatusError):
                await _store().append_to_dataset("ds1", pd.DataFrame([{"a": 1}]))


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
        client = MagicMock()
        client.datasets.get_dataset.side_effect = ValueError("Dataset not found: ds1")
        with patch("phoenix.client.Client", return_value=client):
            assert await _store().delete_dataset("ds1") is False

    @pytest.mark.asyncio
    async def test_outage_raises_not_false(self):
        """A dead/hung backend must not be read as 'nothing to delete' — that
        turns replace-dataset's delete-then-create into a silent append."""
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(503)
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(httpx.HTTPStatusError):
                await _store().delete_dataset("ds1")
