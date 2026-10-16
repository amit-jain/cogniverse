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


@pytest.mark.parametrize(
    "profile", ("video_xclip_sv_chunk_6s", "video_embeddinggemma2_sv_chunk_6s")
)
def test_single_vector_chunk_profiles_store_one_document_per_chunk(profile):
    """Their schemas hold one vector per document, with the chunk's
    segment_id and span; stacking a video's chunks into one document is
    refused by the single-vector field."""
    strategy = StrategyConfig(REPO_CONFIGS).get_strategy(profile)

    assert (
        strategy.processing_type,
        strategy.segmentation,
        strategy.storage_mode,
    ) == ("single_vector", "chunks", "multi_doc")


def test_a_global_single_vector_profile_stores_one_document(strategy_config):
    assert [
        strategy_config._resolve_storage_mode({}, "single_vector", segmentation)
        for segmentation in ("global", "chunks", "windows")
    ] == ["single_doc", "multi_doc", "multi_doc"]
