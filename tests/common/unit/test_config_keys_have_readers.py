"""Every shipped config key either has a reader or stays on the ratchet."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config
from cogniverse_synthetic.utils import partition_profiles_by_sampleability
from tests.fixtures.shipped_config import load_shipped_config

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOTS = [REPO_ROOT / "libs", REPO_ROOT / "scripts", REPO_ROOT / "deploy"]
SOURCE_SUFFIXES = {".py", ".json", ".sh", ".toml", ".yaml", ".yml"}

CONFIG_PATHS = [
    REPO_ROOT / "configs" / "config.json",
    REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json",
    REPO_ROOT / "configs" / "examples" / "config.example.json",
]

STARTUP_PARSE_PATHS = CONFIG_PATHS[:2]

COMMON_RATCHET_KEYS = {
    "annotation_interval_minutes",
    "annotation_lookback_hours",
    "annotation_thresholds",
    "base_path",
    "blocked_patterns",
    "boundary_high",
    "boundary_low",
    "cleanup_on_startup",
    "cohere",
    "enable_ttl",
    "failure_lookback_hours",
    "feedback_interval_minutes",
    "input_rails",
    "intervals",
    "jina",
    "llm_judge",
    "max_annotations_per_batch",
    "max_annotations_per_cycle",
    "max_annotations_per_run",
    "min_annotations_for_optimization",
    "min_annotations_for_update",
    "min_days_between_optimizations",
    "modal_visual_judge",
    "ollama_bge",
    "ollama_mxbai",
    "output_rails",
    "persist_scores",
    "poll_interval_minutes",
    "quality_map",
    "score_annotation_name",
    "span_eval_batch_size",
    "span_eval_lookback_hours",
    "together_ai",
    "very_low_confidence",
}

CONFIG_EXTRA_RATCHET_KEYS = {
    "bucket",
    "enable_reflective_recompile",
    "lifecycle_expiration_days",
    "min_reflective_failures",
    "reflective_max_metric_calls",
}

CHART_EXTRA_RATCHET_KEYS = {
    "bucket",
    "lifecycle_expiration_days",
}

EXPECTED_RATCHET_KEYS = {
    CONFIG_PATHS[0]: COMMON_RATCHET_KEYS | CONFIG_EXTRA_RATCHET_KEYS,
    CONFIG_PATHS[1]: COMMON_RATCHET_KEYS | CHART_EXTRA_RATCHET_KEYS,
    CONFIG_PATHS[2]: COMMON_RATCHET_KEYS,
}

EXPECTED_KEY_COUNTS = {
    CONFIG_PATHS[0]: 248,
    CONFIG_PATHS[1]: 232,
    CONFIG_PATHS[2]: 204,
}

EXPECTED_AGENT_MAPPINGS = {
    "VIDEO": "search_agent",
    "DOCUMENT": "document_agent",
    "IMAGE": "image_search_agent",
    "AUDIO": "audio_analysis_agent",
    "CODE": "coding_agent",
    "WIKI": "document_agent",
}


# Mappings whose entry NAMES are data rather than config keys: code iterates
# them and never reads an entry by literal name, so demanding a literal reader
# would only reward hardcoding the names into libs/. Their values are still
# walked, so the keys INSIDE each entry are checked as usual.
DYNAMIC_MAPPING_PATHS = {
    ("backend", "profiles"),
    ("harness", "api_keys"),
    ("harness", "models"),
}


def _all_keys(node, seen: set[str], path: tuple[str, ...] = ()) -> set[str]:
    if isinstance(node, dict):
        names_are_data = path in DYNAMIC_MAPPING_PATHS
        for key, value in node.items():
            if not names_are_data:
                seen.add(key)
            _all_keys(value, seen, path + (key,))
    elif isinstance(node, list):
        for item in node:
            _all_keys(item, seen, path)
    return seen


def unread_config_keys(config_path: Path, source_roots: Sequence[Path]) -> set[str]:
    keys = _all_keys(load_shipped_config(config_path), set())
    blob = "\n".join(
        path.read_text(errors="replace")
        for root in source_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )
    return {key for key in keys if f'"{key}"' not in blob and f"'{key}'" not in blob}


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_no_new_shipped_config_key_lacks_a_reader(path: Path):
    keys = _all_keys(load_shipped_config(path), set())
    assert len(keys) == EXPECTED_KEY_COUNTS[path]

    unread = unread_config_keys(path, SOURCE_ROOTS)

    assert unread == EXPECTED_RATCHET_KEYS[path]


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize("path", STARTUP_PARSE_PATHS, ids=lambda p: p.name)
def test_shipped_configs_pass_system_tenant_startup_parse(path: Path):
    parsed = parse_synthetic_runtime_config(
        load_shipped_config(path), tenant_id=SYSTEM_TENANT_ID
    )

    modality_config = parsed.generator_config.get_optimizer_config("modality")
    mappings = {
        rule.modality: rule.agent_name for rule in modality_config.agent_mappings
    }
    assert mappings == EXPECTED_AGENT_MAPPINGS

    sampleable, internal = partition_profiles_by_sampleability(
        parsed.backend_config.profiles
    )
    assert internal == {}
    sampleable_modalities = {profile.type.upper() for profile in sampleable.values()}
    assert sampleable_modalities == set(EXPECTED_AGENT_MAPPINGS)

    for modality, agent_name in EXPECTED_AGENT_MAPPINGS.items():
        agent = parsed.agents_config[agent_name]
        assert agent["enabled"] is True
        assert modality in agent["modalities"]
