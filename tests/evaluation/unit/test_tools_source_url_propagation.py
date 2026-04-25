"""Tests that source_url propagates through the eval result-dict normalization in tools.py."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_search_service():
    """Patch SearchService so we control the to_dict() shape returned to the normalizer."""
    with patch("cogniverse_agents.search.service.SearchService") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_config():
    with (
        patch(
            "cogniverse_foundation.config.utils.create_default_config_manager"
        ) as mock_cm,
        patch("cogniverse_foundation.config.utils.get_config") as mock_get,
    ):
        mock_cm.return_value = MagicMock()
        mock_get.return_value = {}
        yield


def _make_result(result_dict):
    obj = MagicMock()
    obj.to_dict.return_value = result_dict
    return obj


@pytest.mark.asyncio
async def test_source_url_at_top_level_propagates(mock_search_service, mock_config):
    from cogniverse_evaluation.core.tools import video_search_tool

    mock_search_service.search.return_value = [
        _make_result(
            {
                "source_id": "v_abc",
                "score": 0.9,
                "document_id": "v_abc_segment_0",
                "content": "alpha",
                "source_url": "s3://corpus/v_abc.mp4",
                "metadata": {},
            }
        )
    ]

    tool_fn = video_search_tool()
    results = await tool_fn(query="dogs running", profile="p", strategy="s", top_k=1)

    assert len(results) == 1
    assert results[0]["source_url"] == "s3://corpus/v_abc.mp4"
    assert results[0]["video_id"] == "v_abc"


@pytest.mark.asyncio
async def test_source_url_in_metadata_propagates(mock_search_service, mock_config):
    from cogniverse_evaluation.core.tools import video_search_tool

    mock_search_service.search.return_value = [
        _make_result(
            {
                "source_id": "v_xyz",
                "score": 0.7,
                "document_id": "v_xyz_segment_0",
                "metadata": {"source_url": "file:///abs/v_xyz.mp4"},
            }
        )
    ]

    tool_fn = video_search_tool()
    results = await tool_fn(query="q", profile="p", strategy="s", top_k=1)

    assert results[0]["source_url"] == "file:///abs/v_xyz.mp4"


@pytest.mark.asyncio
async def test_missing_source_url_yields_empty_string(mock_search_service, mock_config):
    from cogniverse_evaluation.core.tools import video_search_tool

    mock_search_service.search.return_value = [
        _make_result(
            {
                "source_id": "v_abc",
                "score": 0.5,
                "document_id": "v_abc_segment_0",
                "metadata": {},
            }
        )
    ]

    tool_fn = video_search_tool()
    results = await tool_fn(query="q", profile="p", strategy="s", top_k=1)

    assert results[0]["source_url"] == ""


@pytest.mark.asyncio
async def test_top_level_takes_precedence_over_metadata(
    mock_search_service, mock_config
):
    from cogniverse_evaluation.core.tools import video_search_tool

    mock_search_service.search.return_value = [
        _make_result(
            {
                "source_id": "v_abc",
                "score": 0.5,
                "document_id": "v_abc_segment_0",
                "source_url": "top_level",
                "metadata": {"source_url": "in_metadata"},
            }
        )
    ]

    tool_fn = video_search_tool()
    results = await tool_fn(query="q", profile="p", strategy="s", top_k=1)

    assert results[0]["source_url"] == "top_level"
