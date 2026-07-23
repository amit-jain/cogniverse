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

from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

pytestmark = pytest.mark.unit


def _store():
    from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

    return PhoenixDatasetStore(http_endpoint="http://phoenix:6006", tenant_id="acme:a")


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
            with pytest.raises(httpx.HTTPStatusError):
                await _store().get_dataset("ds1")

    @pytest.mark.asyncio
    async def test_503_outage_raises_not_missing(self):
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(503)
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(httpx.HTTPStatusError):
                await _store().get_dataset("ds1")

    @pytest.mark.asyncio
    async def test_name_containing_404_still_raises_on_outage(self):
        """Dataset name 'quality-baseline-20260404' + a 500 must not be read
        as not-found just because '404' appears in the name/message."""
        client = MagicMock()
        client.datasets.get_dataset.side_effect = _http_error(500)
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(httpx.HTTPStatusError):
                await _store().get_dataset("quality-baseline-20260404")


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


class TestCreateDataset:
    @pytest.mark.asyncio
    async def test_409_conflict_appends_new_version(self):
        from phoenix.client.resources.datasets import DatasetUploadError

        client = MagicMock()
        conflict = DatasetUploadError("Dataset upload failed: already exists")
        conflict.__cause__ = _http_error(409)
        client.datasets.create_dataset.side_effect = conflict
        client.datasets.add_examples_to_dataset.return_value = MagicMock(id="ds1")
        with patch("phoenix.client.Client", return_value=client):
            result = await _store().create_dataset("ds1", pd.DataFrame([{"a": 1}]))
        assert result == "ds1"
        client.datasets.add_examples_to_dataset.assert_called_once()

    @pytest.mark.asyncio
    async def test_500_with_already_exists_body_is_not_appended(self):
        """A non-conflict 500 whose body text merely contains 'already exists'
        must fail loudly, never be silently rerouted to append."""
        from phoenix.client.resources.datasets import DatasetUploadError

        client = MagicMock()
        err = DatasetUploadError("Dataset upload failed: WAL says already exists")
        err.__cause__ = _http_error(500)
        client.datasets.create_dataset.side_effect = err
        with patch("phoenix.client.Client", return_value=client):
            with pytest.raises(DatasetUploadError):
                await _store().create_dataset("ds1", pd.DataFrame([{"a": 1}]))
        client.datasets.add_examples_to_dataset.assert_not_called()


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
