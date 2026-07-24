"""The shipped config samples keyframes at 0.5 fps, not the old 1.0.

At 1.0 fps a 10-minute video yields ~600 keyframes, each of which becomes an
embedding, a VLM description call, and an object-store write — 5-10x the
pipeline cost of a lower rate. The default was deliberately lowered to 0.5 fps
(one keyframe every two seconds). This pins that value so it can't silently
drift back to 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs" / "config.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_top_level_pipeline_default_is_half_fps():
    cfg = _config()
    pipeline = cfg["pipeline_config"]
    assert pipeline["keyframe_extraction_method"] == "fps"
    assert pipeline["keyframe_fps"] == 0.5


def test_default_video_profile_samples_at_half_fps():
    cfg = _config()
    backend = cfg["backend"]
    default_video_profile = backend["default_profiles"]["video"]["profile"]
    profile = backend["profiles"][default_video_profile]
    # pipeline_config.keyframe_fps feeds the keyframe CACHE KEY...
    assert profile["pipeline_config"]["keyframe_fps"] == 0.5
    # ...while strategies.segmentation.params.fps is what actually drives
    # extraction (FrameSegmentationStrategy -> KeyframeProcessor). Both must
    # agree, or the cache key describes a different rate than what was extracted.
    seg_params = profile["strategies"]["segmentation"]["params"]
    assert seg_params["fps"] == 0.5


def test_example_config_matches_the_shipped_default():
    example = json.loads(
        (REPO_ROOT / "configs" / "examples" / "config.example.json").read_text()
    )
    assert example["pipeline_config"]["keyframe_fps"] == 0.5


def test_chart_config_matches_the_shipped_default():
    """The Helm chart ships its OWN config.json (mounted over
    /app/configs/config.json in the runtime and ingestor pods), so a default
    changed only in configs/config.json never reaches a chart deployment.
    Pin the chart copy to the same sampling values as the repo config."""
    import re

    raw = (REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json").read_text()
    # The chart copy embeds Helm expressions ({{ include "..." . }}) whose inner
    # quotes break plain JSON parsing — neutralise them; the fps values under
    # pin are literal numbers untouched by templating.
    chart = json.loads(re.sub(r"\{\{.*?\}\}", "HELM", raw, flags=re.S))
    assert chart["pipeline_config"]["keyframe_fps"] == 0.5

    backend = chart["backend"]
    default_video_profile = backend["default_profiles"]["video"]["profile"]
    profile = backend["profiles"][default_video_profile]
    assert profile["pipeline_config"]["keyframe_fps"] == 0.5
    assert profile["strategies"]["segmentation"]["params"]["fps"] == 0.5
