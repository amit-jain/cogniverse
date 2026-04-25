"""Wiring tests for the MediaLocator integration in VideoIngestionPipeline.

Phase 2 of the unified-MediaLocator rollout: file:// only, processors still
receive Path objects. These tests cover the boundaries we changed:
- PipelineConfig.media_root_uri propagates through the builder
- Pipeline construction wires up self.locator
- get_video_files returns URIs from the locator when media_root_uri is set
- _canonical_uri / _display_name normalize URIs and Paths consistently
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from cogniverse_core.common.media import MediaConfig, MediaLocator
from cogniverse_runtime.ingestion.pipeline import (
    PipelineConfig,
    VideoIngestionPipeline,
)
from cogniverse_runtime.ingestion.pipeline_builder import (
    PipelineConfigBuilder,
    VideoIngestionPipelineBuilder,
    create_config,
)


@pytest.fixture
def stub_pipeline(tmp_path):
    """Pipeline shell with locator + config + logger, no I/O."""
    pipeline = VideoIngestionPipeline.__new__(VideoIngestionPipeline)
    pipeline.tenant_id = "acme"
    pipeline.config = PipelineConfig(video_dir=tmp_path)
    pipeline.locator = MediaLocator(
        tenant_id="acme",
        config=MediaConfig(),
        cache_root=tmp_path / "cache",
    )
    pipeline.logger = Mock()
    return pipeline


class TestPipelineConfigMediaRootUri:
    def test_default_is_none(self):
        assert PipelineConfig().media_root_uri is None

    def test_explicit_value_set(self):
        cfg = PipelineConfig(media_root_uri="s3://corpus/")
        assert cfg.media_root_uri == "s3://corpus/"


class TestBuilderPropagation:
    def test_video_pipeline_builder_threads_media_root_uri(self):
        builder = VideoIngestionPipelineBuilder()
        builder.with_media_root_uri("s3://corpus/")
        assert builder._media_root_uri == "s3://corpus/"

    def test_pipeline_config_builder_threads_media_root_uri(self):
        cfg = (
            PipelineConfigBuilder()
            .video_dir(Path("data/videos"))
            .media_root_uri("pvc://media/")
            .build()
        )
        assert cfg.media_root_uri == "pvc://media/"

    def test_create_config_fluent_chain(self):
        cfg = create_config().video_dir(Path("/x")).media_root_uri("s3://b/").build()
        assert cfg.media_root_uri == "s3://b/"
        assert cfg.video_dir == Path("/x")


class TestGetVideoFilesLegacyMode:
    def test_no_media_root_uri_globs_directory(self, stub_pipeline, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"")
        (tmp_path / "b.mkv").write_bytes(b"")
        (tmp_path / "ignore.txt").write_bytes(b"")

        files = stub_pipeline.get_video_files(tmp_path)

        assert all(isinstance(f, Path) for f in files)
        assert sorted(f.name for f in files) == ["a.mp4", "b.mkv"]

    def test_explicit_argument_overrides_config(self, stub_pipeline, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        (other / "x.mp4").write_bytes(b"")

        files = stub_pipeline.get_video_files(other)

        assert [f.name for f in files] == ["x.mp4"]


class TestGetVideoFilesUriMode:
    def test_returns_uris_when_media_root_set(self, stub_pipeline, tmp_path):
        media_root = tmp_path / "corpus"
        media_root.mkdir()
        (media_root / "a.mp4").write_bytes(b"")
        (media_root / "sub").mkdir()
        (media_root / "sub" / "b.mkv").write_bytes(b"")

        stub_pipeline.config = PipelineConfig(
            video_dir=tmp_path, media_root_uri=f"file://{media_root}"
        )

        files = stub_pipeline.get_video_files()

        assert all(isinstance(f, str) for f in files)
        assert all(f.startswith("file://") for f in files)
        assert any(f.endswith("/a.mp4") for f in files)
        assert any(f.endswith("/b.mkv") for f in files)

    def test_uris_localizable_through_locator(self, stub_pipeline, tmp_path):
        media_root = tmp_path / "corpus"
        media_root.mkdir()
        clip = media_root / "v.mp4"
        clip.write_bytes(b"video")

        stub_pipeline.config = PipelineConfig(
            video_dir=tmp_path, media_root_uri=f"file://{media_root}"
        )

        files = stub_pipeline.get_video_files()
        assert len(files) == 1
        local = stub_pipeline.locator.localize(files[0])
        assert local == clip
        assert local.read_bytes() == b"video"


class TestCanonicalUriHelper:
    def test_passthrough_uri(self, stub_pipeline):
        assert stub_pipeline._canonical_uri("s3://b/v.mp4") == "s3://b/v.mp4"

    def test_path_becomes_file_uri(self, stub_pipeline, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"")

        result = stub_pipeline._canonical_uri(f)
        assert result == f"file://{f.resolve()}"

    def test_string_path_canonicalized(self, stub_pipeline, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"")
        assert stub_pipeline._canonical_uri(str(f)) == f"file://{f.resolve()}"


class TestDisplayNameHelper:
    def test_path(self, stub_pipeline, tmp_path):
        assert stub_pipeline._display_name(tmp_path / "v.mp4") == "v.mp4"

    def test_file_uri(self, stub_pipeline):
        assert stub_pipeline._display_name("file:///abs/path/clip.mkv") == "clip.mkv"

    def test_s3_uri(self, stub_pipeline):
        assert stub_pipeline._display_name("s3://corpus/v.mp4") == "v.mp4"


class TestPrepareBaseResultsIncludesSourceUrl:
    def test_source_url_present(self, stub_pipeline, tmp_path):
        clip = tmp_path / "v.mp4"
        clip.write_bytes(b"data")
        stub_pipeline.config = PipelineConfig(video_dir=tmp_path)

        result = stub_pipeline._prepare_base_results(clip)

        assert result["source_url"] == f"file://{clip.resolve()}"
        assert result["video_path"] == str(clip)
        assert result["video_id"] == "v"
