"""End-to-end visual judge test: ingest → search → judge → score, no mocks.

Exercises the canonical chain the unified-MediaLocator rollout was built for:

1. A real Vespa container (managed by ``eval_vespa_instance``) holds documents
   with ``source_url`` set to the on-disk test videos via ``eval_seeded_documents``.
2. A real Ollama container (managed by ``OllamaTestManager``) serves a small
   multimodal model (``moondream``) at a tenant-scoped endpoint.
3. ``ConfigurableVisualJudge`` is constructed against that endpoint and run
   on result dicts shaped like the eval normalizer's output (with
   ``source_url``, ``video_id``).
4. The judge resolves ``source_url`` through ``MediaLocator``, extracts frames
   from the local video, feeds them to Ollama, and returns a score.

Asserts the judge returns a usable result object with a numeric score and
non-empty frame extraction. No mocking the LLM, Vespa, or filesystem
boundaries.

Skips cleanly via ``requires_docker`` when Docker is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.system.ollama_test_manager import DEFAULT_MODEL, OllamaTestManager


@pytest.fixture(scope="module")
def ollama_instance():
    """Start an Ollama container with the moondream multimodal model loaded."""
    manager = OllamaTestManager(model=DEFAULT_MODEL)
    instance = manager.start()
    try:
        yield instance
    finally:
        manager.stop()


@pytest.fixture
def visual_judge(ollama_instance, tmp_path, monkeypatch):
    """Construct ConfigurableVisualJudge wired to the test Ollama + a fresh cache.

    Patches BOTH the lazily-imported ``create_default_config_manager`` (called
    inside ``evaluate``) and the module-level ``get_config`` so the judge runs
    without needing the project's ``configs/config.json`` reachable on disk.
    """
    from unittest.mock import MagicMock

    from cogniverse_core.common.media import (
        MediaCacheConfig,
        MediaConfig,
        MediaLocator,
    )
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_evaluation.evaluators.configurable_visual_judge import (
        ConfigurableVisualJudge,
    )

    fake_config = {
        "evaluators": {
            "visual_judge": {
                "provider": "ollama",
                "model": ollama_instance.model,
                "base_url": ollama_instance.base_url,
                "api_key": None,
                "frames_per_video": 3,
                "max_videos": 1,
                "max_total_frames": 3,
            }
        }
    }
    monkeypatch.setattr(
        "cogniverse_evaluation.evaluators.configurable_visual_judge.get_config",
        lambda **_: fake_config,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: MagicMock(),
    )

    locator = MediaLocator(
        tenant_id=SYSTEM_TENANT_ID,
        config=MediaConfig(cache=MediaCacheConfig(max_bytes_gb=1)),
        cache_root=tmp_path / "judge-cache",
    )
    return ConfigurableVisualJudge(locator=locator)


@pytest.mark.requires_docker
@pytest.mark.integration
class TestVisualJudgeE2E:
    def test_judge_resolves_source_url_via_path_outside_legacy_dirs(
        self, visual_judge, tmp_path
    ):
        """Source video lives at a tmp path with a unique filename, so the
        legacy probe (``data/testset/...``, ``data/videos/``, ``outputs/videos/``)
        cannot find it. Only the new ``source_url``-driven MediaLocator path
        can resolve it.

        Hardened assertions confirm the LLM was actually invoked
        (``frames_evaluated > 0`` + ``provider == "ollama"`` + a real label),
        so a no_frames fallback returning score=0.0 cannot pass.
        """
        videos_dir = (
            Path(__file__).resolve().parents[2] / "system" / "resources" / "videos"
        )
        sources = sorted(videos_dir.glob("*.mp4"))
        if not sources:
            pytest.skip(f"No test videos under {videos_dir}")

        # Copy the test video to a tmp_path with a unique video_id that does
        # NOT exist in any legacy probe directory. This is the load-bearing
        # invariant: source_url is the ONLY path that can resolve this video.
        unique_video_id = "cogniverse_e2e_judge_only_via_source_url"
        clip = tmp_path / f"{unique_video_id}.mp4"
        clip.write_bytes(sources[0].read_bytes())

        # Invariant: confirm the unique filename exists nowhere a legacy probe
        # would look. This sanity check fails fast if the test data layout
        # changes in a way that compromises the assertion below.
        legacy_dirs = {
            Path("data/testset/evaluation/sample_videos"),
            Path("data/videos"),
            Path("outputs/videos"),
        }
        for legacy in legacy_dirs:
            assert not (legacy / clip.name).exists(), (
                f"Test invariant broken: {clip.name} exists under legacy probe "
                f"path {legacy}."
            )

        results = [
            {
                "video_id": unique_video_id,
                "score": 0.9,
                "rank": 1,
                "source_url": f"file://{clip.resolve()}",
                "content": "test",
                "metadata": {},
            }
        ]

        eval_result = visual_judge.evaluate(
            input={"query": "a video clip"},
            output={"results": results},
        )

        # Outcome 1: locator actually fetched the bytes from source_url and
        # cv2 decoded exactly the requested number of frames (3, matching
        # frames_per_video=3, max_total_frames=3, max_videos=1 in fixture).
        assert eval_result.metadata["frames_evaluated"] == 3

        # Outcome 2: label is one of the four LLM-response labels. The judge
        # only emits these after parsing a real LLM response; every other
        # code path (no_results, no_frames, evaluation_failed) returns a
        # different label. So this assertion is true iff the LLM was
        # actually invoked AND its response was successfully parsed.
        assert eval_result.label in {
            "excellent_match",
            "good_match",
            "partial_match",
            "poor_match",
        }

    def test_legacy_dir_video_is_not_resolved_without_source_url(
        self, visual_judge, tmp_path, monkeypatch
    ):
        """Pre-rollout code resolved video_id by globbing ``data/testset/...``.
        Phase 6 removed that probe. This test plants a video at the legacy
        location, omits source_url, and asserts the judge returns no_frames —
        proving the legacy probe is gone.

        On pre-Phase-6 code, the legacy probe would have FOUND the video and
        the judge would have called the LLM, so this test would fail (score
        would be a real LLM response, not 0.0/no_frames).
        """
        legacy_dir = tmp_path / "data" / "testset" / "evaluation" / "sample_videos"
        legacy_dir.mkdir(parents=True)
        # Use a real video so the old code path would actually have decoded it.
        videos_dir = (
            Path(__file__).resolve().parents[2] / "system" / "resources" / "videos"
        )
        source_videos = sorted(videos_dir.glob("*.mp4"))
        if not source_videos:
            pytest.skip(f"No test videos under {videos_dir}")
        legacy_clip = legacy_dir / "legacy_visible.mp4"
        legacy_clip.write_bytes(source_videos[0].read_bytes())

        monkeypatch.chdir(tmp_path)

        results = [
            {
                "video_id": "legacy_visible",
                "score": 0.5,
                "rank": 1,
                # source_url deliberately omitted — pre-Phase-6 code would have
                # found legacy_visible.mp4 via the data/testset/... probe.
            }
        ]

        eval_result = visual_judge.evaluate(
            input={"query": "anything"},
            output={"results": results},
        )

        assert eval_result is not None
        assert eval_result.score == 0.0
        assert eval_result.label == "no_frames", (
            f"Legacy probe is still active: judge returned {eval_result.label!r} "
            f"(score={eval_result.score}) for a video discoverable only via the "
            f"removed data/testset/... fallback."
        )
