"""Unit tests for ingestion_worker._summarise.

Regression: the function previously read ``schema_name``/``tenant_id``/
``duration_seconds`` from the pipeline envelope, but the pipeline emits
``duration`` (not ``duration_seconds``) and never sets ``schema_name``/
``tenant_id`` at the top level. Status events silently lost the duration.
"""

from __future__ import annotations

import pytest

from cogniverse_runtime.ingestion_worker.worker import _summarise


@pytest.mark.unit
class TestSummariseKeys:
    def _envelope(self, **overrides):
        # Mirrors pipeline._prepare_base_results' top-level keys.
        base = {
            "video_id": "vid_123",
            "video_path": "/data/vid_123.mp4",
            "source_url": "s3://bucket/vid_123.mp4",
            "duration": 12.5,
            "pipeline_config": {},
            "results": {},
            "started_at": "2026-05-28T00:00:00+00:00",
            "status": "completed",
        }
        base.update(overrides)
        return base

    def test_reads_duration_not_duration_seconds(self):
        out = _summarise(self._envelope())
        assert out["duration"] == 12.5
        assert "duration_seconds" not in out

    def test_carries_video_id_and_source_url(self):
        out = _summarise(self._envelope())
        assert out["video_id"] == "vid_123"
        assert out["source_url"] == "s3://bucket/vid_123.mp4"

    def test_omits_top_level_schema_name_and_tenant_id(self):
        """Pipeline never sets these at the envelope top level — _summarise must
        not pretend they did. Callers merge them from the job context."""
        out = _summarise(self._envelope())
        assert "schema_name" not in out
        assert "tenant_id" not in out

    def test_handles_keyframes_and_documents_fed(self):
        env = self._envelope(
            results={
                "keyframes": [1, 2, 3],
                "embeddings": {"documents_fed": 7},
            }
        )
        out = _summarise(env)
        assert out["keyframes"] == 3
        assert out["documents_fed"] == 7

    def test_non_dict_input_returns_raw_type(self):
        assert _summarise("not a dict") == {"raw_type": "str"}
