"""Every shipped config key either has a reader or stays on the ratchet."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config
from cogniverse_synthetic.utils import partition_profiles_by_sampleability

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
    "adaptive_segmentation",
    "allowed_topics",
    "annotation_interval_minutes",
    "annotation_lookback_hours",
    "annotation_thresholds",
    "artifacts_path",
    "base_path",
    "blocked_patterns",
    "boundary_high",
    "boundary_low",
    "cache_config",
    "cache_ttl_seconds",
    "chunk_based",
    "cleanup_on_startup",
    "cohere",
    "default_system_prompt",
    "dspy_enabled",
    "enable_caching",
    "enable_fallback",
    "enable_metrics",
    "enable_slow_path",
    "enable_tracing",
    "enable_ttl",
    "enabled_strategies",
    "ensemble_config",
    "entity_confidence_threshold",
    "export_metrics",
    "extract_audio",
    "failure_lookback_hours",
    "feedback_interval_minutes",
    "frame_features",
    "gliner_config",
    "gliner_label_optimization",
    "gliner_labels",
    "gliner_threshold_optimization",
    "include_original",
    "inference_config",
    "input_rails",
    "intervals",
    "jina",
    "keyframe_extraction_method",
    "keyword",
    "keyword_config",
    "llm_auto_annotator",
    "llm_judge",
    "log_file",
    "log_level",
    "max_acceptable_latency_ms",
    "max_annotations_per_batch",
    "max_annotations_per_cycle",
    "max_annotations_per_run",
    "max_cache_size",
    "max_frames_per_chunk",
    "max_length",
    "max_message_length",
    "max_patches",
    "max_pixels",
    "max_results_per_message",
    "max_routing_time_ms",
    "max_sub_questions",
    "messaging",
    "metadata_format",
    "metrics_batch_size",
    "metrics_export_dir",
    "min_accuracy",
    "min_annotations_for_optimization",
    "min_annotations_for_update",
    "min_days_between_optimizations",
    "min_entities_for_fast_path",
    "min_frames",
    "modal_visual_judge",
    "monitoring_config",
    "native_dimensions",
    "ollama_bge",
    "ollama_config",
    "ollama_mxbai",
    "optimization_config",
    "optimization_improvement_threshold",
    "output_rails",
    "performance_degradation_threshold",
    "persist_scores",
    "poll_interval_minutes",
    "quality_map",
    "query_analysis_module",
    "query_fusion_config",
    "query_inference_engine",
    "report_keywords",
    "score_annotation_name",
    "slow_path_confidence_threshold",
    "span_eval_batch_size",
    "span_eval_lookback_hours",
    "summary_keywords",
    "text_keywords",
    "tier_config",
    "together_ai",
    "use_hybrid",
    "use_pipeline_cache",
    "very_low_confidence",
    "video_keywords",
    "voting_method",
    "weights",
    "whisper_model",
}

CONFIG_EXTRA_RATCHET_KEYS = {
    "advisory",
    "bucket",
    "enable_reflective_recompile",
    "key_prefix",
    "lifecycle_expiration_days",
    "min_reflective_failures",
    "reflective_max_metric_calls",
}

CHART_EXTRA_RATCHET_KEYS = {
    "bucket",
    "key_prefix",
    "lifecycle_expiration_days",
}

EXAMPLE_EXTRA_RATCHET_KEYS = {
    "acoustic_embedding_dim",
    "semantic_binary_dim",
    "semantic_embedding_dim",
}

EXPECTED_RATCHET_KEYS = {
    CONFIG_PATHS[0]: COMMON_RATCHET_KEYS | CONFIG_EXTRA_RATCHET_KEYS,
    CONFIG_PATHS[1]: COMMON_RATCHET_KEYS | CHART_EXTRA_RATCHET_KEYS,
    CONFIG_PATHS[2]: COMMON_RATCHET_KEYS | EXAMPLE_EXTRA_RATCHET_KEYS,
}

EXPECTED_KEY_COUNTS = {
    CONFIG_PATHS[0]: 321,
    CONFIG_PATHS[1]: 306,
    CONFIG_PATHS[2]: 296,
}

EXPECTED_AGENT_MAPPINGS = {
    "VIDEO": "search_agent",
    "DOCUMENT": "document_agent",
    "IMAGE": "image_search_agent",
    "AUDIO": "audio_analysis_agent",
    "CODE": "coding_agent",
    "WIKI": "document_agent",
}


def _load_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    if "{{" in raw:
        raw = re.sub(r"\{\{[^}]*\}\}", "http://rendered.invalid", raw)
    return json.loads(raw)


def _all_keys(node, seen: set[str]) -> set[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            seen.add(key)
            _all_keys(value, seen)
    elif isinstance(node, list):
        for item in node:
            _all_keys(item, seen)
    return seen


def unread_config_keys(config_path: Path, source_roots: Sequence[Path]) -> set[str]:
    keys = _all_keys(_load_json(config_path), set())
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
    keys = _all_keys(_load_json(path), set())
    assert len(keys) == EXPECTED_KEY_COUNTS[path]

    unread = unread_config_keys(path, SOURCE_ROOTS)

    assert unread == EXPECTED_RATCHET_KEYS[path]


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize("path", STARTUP_PARSE_PATHS, ids=lambda p: p.name)
def test_shipped_configs_pass_system_tenant_startup_parse(path: Path):
    parsed = parse_synthetic_runtime_config(
        _load_json(path), tenant_id=SYSTEM_TENANT_ID
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
