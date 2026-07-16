"""REST ingestion must discover files by the profile's content type.

The /ingestion/start path hard-globbed ``**/*.mp4`` for every profile, so a
document/audio/image profile found zero files and then crashed the pipeline on
an empty batch (``total_time / len(video_files)`` -> ZeroDivisionError, reported
as a cryptic "failed"). These pin the content-type derivation, the per-type
discovery, and the empty-batch short-circuit.
"""

import logging
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_runtime.ingestion.strategies import (
    content_type_for_profile,
    discover_ingestible_files,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class TestContentTypeForProfile:
    @pytest.mark.parametrize(
        "profile,expected",
        [
            ("document_text_semantic", "document"),
            ("document_visual_colpali", "document"),
            ("audio_clap_semantic", "audio"),
            ("image_colpali_mv", "image"),
            ("video_colpali_smol500_mv_frame", "video"),
            ("something_unlabelled", "video"),
        ],
    )
    def test_derives_content_type_from_profile_name(self, profile, expected):
        assert content_type_for_profile(profile) == expected

    def test_schema_name_disambiguates_when_profile_name_is_bare(self):
        assert (
            content_type_for_profile("clap_v2", {"schema_name": "audio_content"})
            == "audio"
        )


class TestDiscoverIngestibleFiles:
    def test_document_profile_yields_the_directory_not_zero_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.md").write_text("world")
        # Document content is processed as a whole-directory batch; the key
        # regression is that this is NOT empty (the mp4 glob returned []).
        assert discover_ingestible_files(tmp_path, "document") == [tmp_path]

    def test_video_profile_yields_individual_video_files(self, tmp_path):
        (tmp_path / "one.mp4").write_bytes(b"\x00")
        (tmp_path / "two.mov").write_bytes(b"\x00")
        (tmp_path / "notes.txt").write_text("ignore me")
        found = discover_ingestible_files(tmp_path, "video")
        assert {p.name for p in found} == {"one.mp4", "two.mov"}

    def test_empty_video_directory_yields_no_files(self, tmp_path):
        assert discover_ingestible_files(tmp_path, "video") == []


class TestEmptyBatchDoesNotCrash:
    @pytest.mark.asyncio
    async def test_process_videos_concurrent_empty_returns_clean_zero(self):
        pipeline = object.__new__(VideoIngestionPipeline)
        pipeline.logger = logging.getLogger("test-pipeline")
        pipeline.tenant_id = "acme:acme"

        result = await pipeline.process_videos_concurrent([])

        # No ZeroDivisionError; a clean "nothing to do" result the router can
        # report as completed with 0 processed.
        assert result["total_videos"] == 0
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["status"] == "completed"
        assert result["results"] == []


class TestRunIngestionRouteDiscovery:
    @pytest.mark.asyncio
    async def test_document_profile_ingests_the_document_dir_not_empty(
        self, tmp_path, monkeypatch
    ):
        """The route must feed the discovered document dir to the pipeline; the
        old ``**/*.mp4`` glob fed an empty batch for a document profile."""
        from cogniverse_runtime.routers import ingestion as ing

        (tmp_path / "a.txt").write_text("hello")

        captured: dict = {}

        class _StubPipeline:
            def __init__(self, **kwargs):
                pass

            async def process_videos_concurrent(self, video_files, max_concurrent=None):
                captured["files"] = list(video_files)
                return {"successful": 1, "errors": [], "results": []}

        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline",
            _StubPipeline,
        )

        job_id = "job-doc-1"
        ing.ingestion_jobs[job_id] = ing.IngestionStatus(
            job_id=job_id,
            status="pending",
            videos_processed=0,
            videos_total=0,
            errors=[],
        )
        request = ing.IngestionRequest(
            video_dir=str(tmp_path),
            profile="document_text_semantic",
            tenant_id="acme:acme",
        )
        try:
            await ing.run_ingestion(
                job_id, request, config_manager=None, schema_loader=None
            )

            assert captured["files"] == [tmp_path]
            assert ing.ingestion_jobs[job_id].status == "completed"
        finally:
            ing.ingestion_jobs.pop(job_id, None)
