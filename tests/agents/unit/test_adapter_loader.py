"""get_active_adapter_path resolves the adapter URI through the cache dir.

Returning ``adapter.adapter_path`` verbatim only worked for locally trained
adapters; a cloud-backed adapter (s3://, modal://) has to be downloaded under
``SystemConfig.adapter_cache_dir`` first. These pin that the loader routes
through ``resolve_adapter_path`` with the cache dir — a local file:// adapter
resolves to its path, and a cloud URI downloads under the cache dir.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from cogniverse_agents.adapter_loader import get_active_adapter_path


def _adapter(effective_uri: str, adapter_path: str = "/stale/unused/path"):
    adapter = Mock()
    adapter.get_effective_uri.return_value = effective_uri
    adapter.adapter_path = adapter_path
    adapter.name = "routing"
    adapter.version = "2.0.0"
    return adapter


@pytest.mark.unit
def test_cloud_uri_downloads_under_cache_dir(tmp_path):
    """A cloud adapter is downloaded under adapter_cache_dir, NOT read from the
    stale local adapter_path."""
    registry = Mock()
    registry.get_active_adapter.return_value = _adapter(
        "s3://bucket/adapters/routing_v2", adapter_path="/stale/unused/path"
    )
    expected_local = str(tmp_path / "routing_v2")

    with (
        patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry),
        patch(
            "cogniverse_finetuning.registry.download_adapter",
            return_value=expected_local,
        ) as mock_download,
    ):
        path = get_active_adapter_path("t1", "routing", adapter_cache_dir=str(tmp_path))

    assert path == expected_local
    # Downloaded to a path under the configured cache dir (not adapter_path).
    called_uri, called_local = mock_download.call_args.args
    assert called_uri == "s3://bucket/adapters/routing_v2"
    assert called_local == expected_local


@pytest.mark.unit
def test_local_file_uri_resolves_to_its_path():
    """A file:// adapter resolves to the on-disk path without downloading."""
    registry = Mock()
    registry.get_active_adapter.return_value = _adapter("file:///models/routing_lora")

    with (
        patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry),
        patch("cogniverse_finetuning.registry.download_adapter") as mock_download,
    ):
        path = get_active_adapter_path("t1", "routing", adapter_cache_dir="/cache")

    assert path == "/models/routing_lora"
    mock_download.assert_not_called()


@pytest.mark.unit
def test_empty_cache_dir_is_rejected_for_a_cloud_adapter():
    """resolve_adapter_path requires a non-empty cache dir; the loader swallows
    the resulting error and returns None rather than crashing the agent."""
    registry = Mock()
    registry.get_active_adapter.return_value = _adapter("s3://bucket/adapters/x")

    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        assert get_active_adapter_path("t1", "routing", adapter_cache_dir="") is None


@pytest.mark.unit
def test_no_active_adapter_returns_none():
    registry = Mock()
    registry.get_active_adapter.return_value = None
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        assert (
            get_active_adapter_path("t1", "routing", adapter_cache_dir="/cache") is None
        )
