"""The visual judge must delete the temp JPEGs it extracts.

extract_frames writes NamedTemporaryFile(delete=False) JPEGs; the judge encoded
them and never unlinked, so a large eval run filled the temp dir. evaluate()
must clean them up whether scoring succeeds or raises.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from cogniverse_evaluation.evaluators.configurable_visual_judge import (
    ConfigurableVisualJudge,
)


def _judge():
    j = object.__new__(ConfigurableVisualJudge)
    j.evaluator_name = "visual"
    j.provider = "test"
    j.model = "test-model"
    return j


def _make_temp_frames(n: int) -> list[str]:
    paths = []
    for _ in range(n):
        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f.write(b"frame")
        f.close()
        paths.append(f.name)
    return paths


def _run_with_frames(judge, frames, score_side_effect=None):
    with (
        patch.object(judge, "_get_video_path", return_value="/tmp/vid.mp4"),
        patch.object(judge, "_extract_frames_from_video", return_value=frames),
        patch.object(
            judge,
            "_score_frames",
            side_effect=score_side_effect,
            return_value=(0.5, "ok"),
        ),
        patch(
            "cogniverse_evaluation.evaluators.configurable_visual_judge.get_config",
            return_value={},
        ),
        patch(
            "cogniverse_foundation.config.utils.create_default_config_manager",
            return_value=object(),
        ),
    ):
        return judge.evaluate(
            input={"query": "q"}, output={"results": [{"video_id": "v1"}]}
        )


def test_frames_unlinked_after_successful_scoring():
    frames = _make_temp_frames(3)
    result = _run_with_frames(_judge(), frames)
    assert result is not None
    assert all(not Path(f).exists() for f in frames), "temp frames must be cleaned up"


def test_frames_unlinked_even_when_scoring_raises():
    frames = _make_temp_frames(3)
    _run_with_frames(_judge(), frames, score_side_effect=RuntimeError("vlm down"))
    assert all(not Path(f).exists() for f in frames), (
        "temp frames must be cleaned up on the error path too"
    )
