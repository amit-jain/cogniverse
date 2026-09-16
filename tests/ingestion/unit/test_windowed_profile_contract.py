"""Shipped profiles that cut text into windows declare the chunk contract."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (  # noqa: E501
    EmbeddingGeneratorImpl,
)
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory
from tests.fixtures.shipped_config import load_shipped_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATHS = [
    REPO_ROOT / "configs" / "config.json",
    REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json",
]
SCHEMA_DIR = REPO_ROOT / "configs" / "schemas"

# The processor a segmentation strategy asks for decides which segment list
# the embedding stage receives, and that list decides which method embeds it.
SEGMENT_METHODS = {
    "document_file": ("document_files", "_process_document_segments"),
    "document_page": ("document_pages", "_process_document_visual_segments"),
    "code_file": ("code_files", "_process_code_segments"),
    "audio_file": ("audio_files", "_process_audio_segments"),
}
CHUNK_FIELDS = ("chunk_index", "chunk_count", "chunk_start", "chunk_end")


def _dispatched_segment_keys() -> set[str]:
    """The segment lists ``generate_embeddings`` routes to their own method."""
    return {
        const
        for const in EmbeddingGeneratorImpl.generate_embeddings.__code__.co_consts
        if isinstance(const, str) and const.endswith(("_files", "_pages"))
    }


def _windowed_segment_keys() -> set[str]:
    """The segment lists whose method cuts the text into model-sized windows."""
    return {
        key
        for key, method in SEGMENT_METHODS.values()
        if "self._text_windows("
        in inspect.getsource(getattr(EmbeddingGeneratorImpl, method))
    }


def _profile_segment_keys(profile: dict) -> set[str]:
    strategy_set = StrategyFactory.create_from_profile_config(copy.deepcopy(profile))
    requirements: dict = {}
    for strategy in strategy_set.get_all_strategies():
        requirements.update(strategy.get_required_processors())
    return {
        SEGMENT_METHODS[name][0] for name in requirements if name in SEGMENT_METHODS
    }


@pytest.mark.unit
@pytest.mark.ci_fast
def test_every_dispatched_segment_kind_is_mapped_to_its_method() -> None:
    assert _dispatched_segment_keys() == {key for key, _ in SEGMENT_METHODS.values()}


@pytest.mark.unit
@pytest.mark.ci_fast
def test_the_windowing_methods_are_the_text_ones() -> None:
    assert _windowed_segment_keys() == {"document_files", "code_files", "audio_files"}


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_windowing_profiles_declare_source_granularity_and_chunk_fields(
    config_path: Path,
) -> None:
    """One source must come back as one hit carrying its windows' spans.

    A profile whose text is cut into windows feeds one document per window,
    so without source granularity its own windows fill the result page, and
    without the chunk fields the spans are dropped on the way into the
    schema.
    """
    windowed = _windowed_segment_keys()
    profiles = load_shipped_config(config_path)["backend"]["profiles"]
    declared = {}
    for name, profile in profiles.items():
        if not _profile_segment_keys(profile) & windowed:
            continue
        schema = json.loads(
            (SCHEMA_DIR / f"{profile['schema_name']}_schema.json").read_text()
        )
        fields = {
            field["name"]: field["type"] for field in schema["document"]["fields"]
        }
        declared[name] = (
            profile.get("result_granularity"),
            tuple(fields.get(field) for field in CHUNK_FIELDS),
            schema.get("document_mapping", {}).get("id"),
        )

    chunk_ints = ("int", "int", "int", "int")
    assert declared == {
        "audio_clap_semantic": ("source", chunk_ints, "audio_id"),
        "document_text_semantic": ("source", chunk_ints, "document_id"),
        "lateon_mv": ("source", chunk_ints, "text_id"),
        "code_lateon_mv": ("source", chunk_ints, "code_id"),
    }
