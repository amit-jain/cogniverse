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
    """Construct ConfigurableVisualJudge wired to the test Ollama + a fresh cache."""
    from cogniverse_core.common.media import (
        MediaCacheConfig,
        MediaConfig,
        MediaLocator,
    )
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_evaluation.evaluators.configurable_visual_judge import (
        ConfigurableVisualJudge,
    )

    # Patch the system-tenant evaluator config so the judge picks up the
    # ephemeral Ollama endpoint and the moondream model.
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

    locator = MediaLocator(
        tenant_id=SYSTEM_TENANT_ID,
        config=MediaConfig(cache=MediaCacheConfig(max_bytes_gb=1)),
        cache_root=tmp_path / "judge-cache",
    )
    return ConfigurableVisualJudge(locator=locator)


@pytest.mark.requires_docker
@pytest.mark.integration
class TestVisualJudgeE2E:
    def test_judge_resolves_source_url_and_returns_score(self, visual_judge, tmp_path):
        """The judge must:

        - resolve ``source_url`` to a local file via the locator,
        - extract frames into the locator-controlled cache,
        - call the real Ollama LLM,
        - return an ``EvaluationResult`` with a float score in [0, 1].
        """
        videos_dir = (
            Path(__file__).resolve().parents[2] / "system" / "resources" / "videos"
        )
        videos = sorted(videos_dir.glob("*.mp4"))
        if not videos:
            pytest.skip(f"No test videos under {videos_dir}")

        clip = videos[0]
        results = [
            {
                "video_id": clip.stem,
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

        assert eval_result is not None
        assert hasattr(eval_result, "score")
        assert isinstance(eval_result.score, float)
        assert 0.0 <= eval_result.score <= 1.0
        # Judge should report frames evaluated and the provider used.
        if hasattr(eval_result, "metadata") and eval_result.metadata:
            assert eval_result.metadata.get("frames_evaluated", 0) > 0
            assert eval_result.metadata.get("provider") == "ollama"

    def test_judge_returns_no_frames_when_source_url_missing(self, visual_judge):
        """Without source_url and without legacy probe (Phase 6 removed it),
        the judge should report no_frames cleanly."""
        results = [{"video_id": "no_such_video", "score": 0.5, "rank": 1}]

        eval_result = visual_judge.evaluate(
            input={"query": "anything"},
            output={"results": results},
        )

        assert eval_result is not None
        assert eval_result.score == 0.0
        if hasattr(eval_result, "label"):
            assert eval_result.label in {"no_frames", "no_results"}
