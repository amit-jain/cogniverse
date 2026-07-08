"""The shipped configuration must not carry developer-personal values.

The chart config (``charts/cogniverse/files/config.json``) renders into every
runtime pod. It had shipped a developer's private Modal VLM endpoint (a fresh
tenant's ingestion POSTed keyframes there), a macOS ``/Users/amjain`` path, a
top-level ``device: mps`` override that fails on Linux/CUDA, and a
developer-specific ``default_tenant``. These pin them out so they can't return:
the endpoints/paths are emptied (the readers fail loud), and the device
override is removed so pods auto-detect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    REPO_ROOT / "configs" / "config.json",
    REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json",
]

FORBIDDEN_SUBSTRINGS = [
    "amit-jain",  # personal Modal deployment
    "/Users/amjain",  # macOS home path
    "flywheel_org",  # developer-specific tenant/org
]


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name + ":" + p.parent.name)
def test_no_developer_personal_values(path: Path):
    text = path.read_text(encoding="utf-8")
    hits = [s for s in FORBIDDEN_SUBSTRINGS if s in text]
    assert not hits, f"{path} still ships developer-personal values: {hits}"


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name + ":" + p.parent.name)
def test_no_top_level_mps_device_override(path: Path):
    """A top-level ``device: mps`` override forces Apple Metal on Linux pods.
    Absent, ModelLoader.get_device auto-detects cuda/mps/cpu."""
    text = path.read_text(encoding="utf-8")
    # Only the top-level override is forbidden; nested cpu device for the small
    # text classifier is fine. The top-level key sits at 2-space indent.
    assert '\n  "device": "mps"' not in text, f"{path} pins device=mps for all pods"


@pytest.mark.unit
def test_local_config_is_valid_json():
    """configs/config.json is plain JSON (the chart file is a Helm template)."""
    json.loads(CONFIGS[0].read_text(encoding="utf-8"))
