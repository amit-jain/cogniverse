"""Tests that source_url propagates through Phoenix experiment result normalization."""

from unittest.mock import MagicMock, patch

import pytest


def _make_result(result_dict):
    obj = MagicMock()
    obj.to_dict.return_value = result_dict
    return obj


@pytest.fixture
def patched_search():
    with (
        patch("cogniverse_agents.search.service.SearchService") as mock_service_cls,
        patch(
            "cogniverse_foundation.config.utils.create_default_config_manager"
        ) as mock_cm,
        patch("cogniverse_foundation.config.utils.get_config") as mock_get_config,
    ):
        instance = MagicMock()
        mock_service_cls.return_value = instance
        mock_cm.return_value = MagicMock()
        mock_get_config.return_value = {}
        yield instance


def _run_phoenix_task(patched_search, results):
    from cogniverse_telemetry_phoenix.evaluation.experiments import (
        PhoenixExperimentPlugin,
    )

    patched_search.search.return_value = results

    task = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
        inspect_solver=MagicMock(),
        profiles=["video_colpali"],
        strategies=["binary_binary"],
        config={"top_k": 10},
    )
    example = MagicMock()
    example.input = {"query": "anything"}
    return task(example)


class TestSourceUrlPropagation:
    def test_top_level_source_url_emitted(self, patched_search):
        out = _run_phoenix_task(
            patched_search,
            [
                _make_result(
                    {
                        "source_id": "v_abc",
                        "score": 0.9,
                        "document_id": "v_abc_segment_0",
                        "content": "alpha",
                        "source_url": "s3://corpus/v_abc.mp4",
                    }
                )
            ],
        )

        config_result = out["results"]["video_colpali_binary_binary"]
        assert config_result["success"], config_result.get("error")
        formatted = config_result["results"]
        assert len(formatted) == 1
        assert formatted[0]["source_url"] == "s3://corpus/v_abc.mp4"
        assert formatted[0]["video_id"] == "v_abc"

    def test_source_url_in_metadata_emitted(self, patched_search):
        out = _run_phoenix_task(
            patched_search,
            [
                _make_result(
                    {
                        "source_id": "v_xyz",
                        "score": 0.5,
                        "document_id": "v_xyz_segment_0",
                        "metadata": {"source_url": "file:///abs/v_xyz.mp4"},
                    }
                )
            ],
        )

        config_result = out["results"]["video_colpali_binary_binary"]
        assert config_result["success"], config_result.get("error")
        formatted = config_result["results"]
        assert formatted[0]["source_url"] == "file:///abs/v_xyz.mp4"

    def test_missing_source_url_marks_config_failed(self, patched_search):
        """source_url is required; the per-config result must record failure
        when a search backend ever omits the field rather than emit empty
        downstream."""
        out = _run_phoenix_task(
            patched_search,
            [
                _make_result(
                    {
                        "source_id": "v_abc",
                        "score": 0.5,
                        "document_id": "v_abc_segment_0",
                    }
                )
            ],
        )

        config_result = out["results"]["video_colpali_binary_binary"]
        assert config_result["success"] is False
        assert "missing source_url" in config_result.get("error", "")

    def test_top_level_takes_precedence(self, patched_search):
        out = _run_phoenix_task(
            patched_search,
            [
                _make_result(
                    {
                        "source_id": "v_abc",
                        "score": 0.5,
                        "document_id": "v_abc_segment_0",
                        "source_url": "top_level",
                        "metadata": {"source_url": "in_metadata"},
                    }
                )
            ],
        )

        config_result = out["results"]["video_colpali_binary_binary"]
        assert config_result["success"], config_result.get("error")
        formatted = config_result["results"]
        assert formatted[0]["source_url"] == "top_level"
