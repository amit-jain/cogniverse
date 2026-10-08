"""The process types profile validation accepts are the ones ingestion reads."""

import json
import shutil
from pathlib import Path

import pytest

from cogniverse_foundation.config.unified_config import PROCESS_TYPES
from cogniverse_runtime.ingestion.strategy import StrategyConfig

REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs"

# What each process type resolves to for a profile carrying no other hint.
EXPECTED = {
    "direct_video": ("direct_video", "windows"),
    "frame_based": ("frame_based", "frames"),
    "video_chunks": ("single_vector", "chunks"),
}


@pytest.fixture
def strategy_config(tmp_path):
    shutil.copy(REPO_CONFIGS / "config.json", tmp_path / "config.json")
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "ranking_strategies.json").write_text(json.dumps({}))
    return StrategyConfig(tmp_path)


def test_each_accepted_process_type_has_a_resolution():
    assert set(PROCESS_TYPES) == set(EXPECTED)


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_process_type_selects_its_strategy(strategy_config, process_type):
    resolved = strategy_config._resolve_processing_strategy(
        "custom_profile", {"process_type": process_type}
    )
    assert resolved == EXPECTED[process_type]


def test_an_unknown_process_type_falls_to_the_default(strategy_config):
    """Why validation refuses unknown types: ingestion would ignore them."""
    resolved = strategy_config._resolve_processing_strategy(
        "custom_profile", {"process_type": "single_vector"}
    )
    assert resolved == ("frame_based", "frames")
