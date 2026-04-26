"""End-to-end visual judge test: ingest → search → judge → score, no mocks.

Exercises the canonical chain the unified-MediaLocator rollout was built for:

1. Test source video lives at a tmp path with a unique filename so neither
   the (already-removed) legacy probe nor any other resolution path could
   surface it — ``source_url`` on the result dict is the only thing that
   works.
2. ``ConfigurableVisualJudge`` is constructed against a user-supplied LLM
   endpoint (provided by the ``llm_endpoint`` fixture in conftest) and run
   on result dicts shaped like the eval normalizer's output.
3. The judge resolves ``source_url`` through ``MediaLocator``, extracts
   frames from the local video, feeds them to the configured LLM, and
   returns a score.

The test class does not reference any specific LLM provider, model, or
container manager. Set ``COGNIVERSE_TEST_LLM_PROVIDER_URI`` (and optionally
``COGNIVERSE_TEST_LLM_BASE_URL``) to point at a vision-capable model — see
``conftest.py`` for the resolution chain. The test skips when no endpoint
is configured.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def visual_judge(llm_endpoint, tmp_path, monkeypatch):
    """Wire ``ConfigurableVisualJudge`` to ``llm_endpoint`` + a fresh cache.

    Patches BOTH the lazily-imported ``create_default_config_manager`` (called
    inside ``evaluate``) and the module-level ``get_config`` so the judge
    runs without needing the project's ``configs/config.json`` reachable on
    disk.
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

    provider, model = llm_endpoint["provider_uri"].split("/", 1)

    fake_config = {
        "evaluators": {
            "visual_judge": {
                "provider": provider,
                "model": model,
                "base_url": llm_endpoint["base_url"],
                # Leave api_key unset; provider SDKs (litellm, openai,
                # anthropic, ollama) auto-resolve from their standard env
                # vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
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


@pytest.mark.integration
class TestVisualJudgeE2E:
    def test_judge_resolves_source_url_via_path_outside_legacy_dirs(
        self, visual_judge, tmp_path
    ):
        """Pin the two outcomes the new MediaLocator path uniquely produces:
        an exact frame count and a label drawn from the LLM-response set.

        Source video lives at a tmp path with a unique filename so neither
        the (already-removed) legacy probe nor any other resolution path
        could have surfaced it — ``source_url`` is the only thing that
        works.
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
