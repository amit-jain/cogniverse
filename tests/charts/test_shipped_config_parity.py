"""Both shipped configs must parse, and must not drift apart.

``charts/cogniverse/files/config.json`` renders into every runtime pod;
``configs/config.json`` feeds local runs. They carry overlapping agent and
modality declarations, and a divergence between them is invisible until a pod
fails ``parse_synthetic_runtime_config`` at startup and crash-loops before
serving. Chart values are rendered by Helm before parsing.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from cogniverse_cli.modal_inference.vllm import _build_process_proxy_app

from cogniverse_agents.optimizer.golden_set_ground_truth import (
    canonicalize_golden_set_ground_truth_rows,
)
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_foundation.config.llm_factory import create_budgeted_dspy_lm
from cogniverse_foundation.config.unified_config import LLMConfig
from cogniverse_runtime.agent_dispatcher import GROUNDING_SEARCH_TIMEOUT_KEY
from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config
from tests.fixtures.shipped_config import load_shipped_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "configs" / "config.json"
CHART = REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json"
CONFIGS = [SHIPPED, CHART]


def _mappings(config: dict[str, Any]) -> list[tuple[str, str]]:
    rules = config["synthetic"]["optimizer_configs"]["modality"]["agent_mappings"]
    return [(rule["modality"], rule["agent_name"]) for rule in rules]


def _optimizer_floors(config: dict[str, Any]) -> dict[str, Any]:
    return config["routing"]["optimization_config"]["optimizer_floors"]


def _optimizer_configs(config: dict[str, Any]) -> dict[str, Any]:
    return config["synthetic"]["optimizer_configs"]


def _training_selection(config: dict[str, Any]) -> dict[str, Any]:
    return config["routing"]["optimization_config"]["training_selection"]


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_shipped_config_passes_system_tenant_startup_parse(path: Path):
    parsed = parse_synthetic_runtime_config(
        load_shipped_config(path), tenant_id=SYSTEM_TENANT_ID
    )

    assert parsed.backend_config.tenant_id == SYSTEM_TENANT_ID
    assert parsed.backend_config.profiles


_SELECTED_VIDEO = {"video": {"profile": "video_colpali_smol500_mv_frame"}}
# Two shipped audio profiles, so audio searches that name none need a default.
_SELECTED_AUDIO = {"audio": {"profile": "audio_clap_semantic"}}
_SELECTED = {**_SELECTED_VIDEO, **_SELECTED_AUDIO}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (SHIPPED, _SELECTED),
        (REPO_ROOT / "configs" / "examples" / "config.example.json", _SELECTED_VIDEO),
    ],
    ids=["config.json", "config.example.json"],
)
def test_local_configs_select_the_default_video_profile_by_name_only(
    path: Path, expected: dict
):
    """The Vespa search backend reads default_profiles.<type>.strategy as a
    ranking-strategy name ahead of the profile's own default_ranking, so a
    value no schema ranks with only hides that default."""
    config = json.loads(path.read_text())

    assert config["backend"]["default_profiles"] == expected


def _composed_chart_config(values: tuple[str, ...], *set_args: str) -> dict[str, Any]:
    """The runtime config.json a values composition renders."""
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART.parents[1]),
        "--show-only",
        "templates/configmap.yaml",
    ]
    for values_file in values:
        cmd += ["-f", str(CHART.parents[1] / values_file)]
    for arg in ("runtime.qualityMonitor.tenantId=test-tenant", *set_args):
        cmd += ["--set", arg]
    rendered = subprocess.run(
        cmd, check=True, capture_output=True, text=True, timeout=30
    )
    (configmap,) = [
        doc
        for doc in yaml.safe_load_all(rendered.stdout)
        if doc["metadata"]["name"] == "cogniverse-config"
    ]
    return json.loads(configmap["data"]["config.json"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "set_args", "expected"),
    [
        ((), (), _SELECTED_AUDIO),
        (
            ("values.k3s.yaml", "values.rocm.yaml", "values.modal-llm.yaml"),
            (),
            _SELECTED,
        ),
        (
            ("values.prod.yaml", "values.cuda.yaml"),
            (
                "minio.rootPassword=test-minio",
                "openshell.server.sshHandshakeSecret=test-handshake",
                "phoenix.postgres.auth.password=test-postgres",
                "redis.auth.password=test-redis",
                "runtime.primaryLLM.apiBase=https://student.example.com/v1",
            ),
            _SELECTED,
        ),
        (
            ("values.prod.yaml",),
            (
                "minio.rootPassword=test-minio",
                "openshell.server.sshHandshakeSecret=test-handshake",
                "phoenix.postgres.auth.password=test-postgres",
                "redis.auth.password=test-redis",
                "config.defaultProfiles.video=video_colpali_smol500_mv_frame",
                "inference.vllm_colpali.externalUrl=https://colpali.example.com",
                "inference.vllm_asr.enabled=false",
                "inference.vllm_asr.externalUrl=https://asr.example.com",
                "runtime.primaryLLM.apiBase=https://student.example.com/v1",
            ),
            _SELECTED,
        ),
    ],
    ids=["base", "k3s-rocm-modal", "prod-cuda-student", "prod-external"],
)
def test_composed_chart_config_passes_the_runtime_startup_parse(
    values: tuple[str, ...], set_args: tuple[str, ...], expected: dict
):
    """The runtime's own startup parse validates backend.default_profiles
    against the catalog, for each composition the chart selects a profile in."""
    parsed = parse_synthetic_runtime_config(
        _composed_chart_config(values, *set_args), tenant_id=SYSTEM_TENANT_ID
    )

    assert parsed.backend_default_profiles == expected


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_every_mapped_agent_declares_the_modality_it_is_mapped_for(path: Path):
    config = load_shipped_config(path)
    agents = config["agents"]

    undeclared = [
        (modality, agent_name)
        for modality, agent_name in _mappings(config)
        if modality not in (agents.get(agent_name, {}).get("modalities") or [])
    ]

    assert undeclared == []


@pytest.mark.unit
def test_shared_agents_declare_identical_modalities_across_shipped_configs():
    shipped_agents = load_shipped_config(SHIPPED)["agents"]
    chart_agents = load_shipped_config(CHART)["agents"]

    drift = {
        name: (
            shipped_agents[name].get("modalities"),
            chart_agents[name].get("modalities"),
        )
        for name in sorted(set(shipped_agents) & set(chart_agents))
        if shipped_agents[name].get("modalities")
        != chart_agents[name].get("modalities")
    }

    assert drift == {}


@pytest.mark.unit
def test_shared_agents_declare_identical_token_streaming_across_shipped_configs():
    """The chart config is what a pod runs, so a streaming declaration made in
    the repo copy alone leaves the deployed agent answering in whole chunks."""
    shipped_agents = load_shipped_config(SHIPPED)["agents"]
    chart_agents = load_shipped_config(CHART)["agents"]
    declared = {
        name: (
            shipped_agents[name].get("streams_answer_tokens", False),
            chart_agents[name].get("streams_answer_tokens", False),
        )
        for name in sorted(set(shipped_agents) & set(chart_agents))
    }

    assert {name: pair for name, pair in declared.items() if pair[0] != pair[1]} == {}
    assert {name for name, pair in declared.items() if pair[0]} == {
        "deep_research_agent",
        "detailed_report_agent",
        "summarizer_agent",
    }


@pytest.mark.unit
def test_shipped_configs_declare_identical_harness_surface():
    """The /v1 model map and key sources render the same in both configs.

    A model name that resolves to an agent locally and to nothing in the pod
    is a 404 only in production.
    """
    shipped = load_shipped_config(SHIPPED)["harness"]
    chart = load_shipped_config(CHART)["harness"]

    assert shipped == chart
    assert shipped == {
        "api_keys": {"$COGNIVERSE_HARNESS_API_KEY": "default"},
        "models": {
            "cogniverse": "gateway_agent",
            "cogniverse/search": "search_agent",
            "cogniverse/summarizer": "summarizer_agent",
            "cogniverse/deep-research": "deep_research_agent",
            "cogniverse/coding": "coding_agent",
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_every_harness_model_names_a_declared_agent(path: Path):
    config = load_shipped_config(path)
    declared = set(config["agents"])

    mapped = set(config["harness"]["models"].values())

    assert mapped - declared == set()


@pytest.mark.unit
def test_shipped_configs_declare_identical_agent_mappings():
    assert _mappings(load_shipped_config(SHIPPED)) == _mappings(
        load_shipped_config(CHART)
    )


@pytest.mark.unit
def test_shipped_configs_declare_identical_optimizer_floors():
    assert _optimizer_floors(load_shipped_config(SHIPPED)) == _optimizer_floors(
        load_shipped_config(CHART)
    )


@pytest.mark.unit
def test_shipped_configs_declare_identical_training_selection():
    # Drift guard: the charted config must match the shipped runtime config.
    assert _training_selection(load_shipped_config(SHIPPED)) == _training_selection(
        load_shipped_config(CHART)
    )


@pytest.mark.unit
def test_shipped_configs_declare_identical_optimizer_configs():
    """Drift guard: the chart config is what the cluster actually runs, so a
    scoring rule added only to the repo copy changes nothing in a deployment
    and silently makes local and served selection disagree."""
    assert _optimizer_configs(load_shipped_config(SHIPPED)) == _optimizer_configs(
        load_shipped_config(CHART)
    )


@pytest.mark.unit
def test_shipped_training_selection_matches_canonical_block():
    # Canonical pin: the shipped config carries the exact expected values.
    assert _training_selection(load_shipped_config(SHIPPED)) == {
        "simba_query_enhancement": {
            "trainset_cap": 300,
            "mmr_lambda": 0.7,
            "low_confirmation_threshold": 3,
            "downweight_age_days": 14,
            "downweight_factor": 0.5,
        },
        "profile_selection": {
            "trainset_cap": 300,
            "mmr_lambda": 0.7,
            "low_confirmation_threshold": 3,
            "downweight_age_days": 14,
            "downweight_factor": 0.5,
        },
        "entity_extraction": {
            "trainset_cap": 300,
            "mmr_lambda": 0.7,
            "low_confirmation_threshold": 3,
            "downweight_age_days": 14,
            "downweight_factor": 0.5,
            "confirmation_score_threshold": 0.7,
        },
    }


@pytest.mark.unit
def test_shipped_configs_declare_identical_teacher_request_bounds():
    """The teacher's model and endpoint are per-deployment, so the chart
    templates them; what a request to it may cost is not, and a window declared
    in one copy only leaves the other budgeting against nothing."""

    def bounds(config: dict[str, Any]) -> dict[str, Any]:
        teacher = config["llm_config"]["teacher"]
        return {
            key: teacher.get(key)
            for key in (
                "temperature",
                "max_tokens",
                "context_window",
                "request_timeout",
            )
        }

    assert (
        bounds(load_shipped_config(SHIPPED))
        == bounds(load_shipped_config(CHART))
        == {
            "temperature": 0.7,
            "max_tokens": 2048,
            "context_window": 4096,
            "request_timeout": 210.0,
        }
    )


# One-token completions against the Modal chat apps scaled to zero after 900 s
# idle, 2026-09-13: student cold 135.0 s, teacher cold 82.6 s (105-120 s on
# earlier days); warm calls under 0.9 s on both.
LONGEST_MEASURED_MODAL_CHAT_COLD_START_S = 135.0


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_the_teacher_lm_waits_out_a_cold_start_inside_the_modal_proxy_bound(
    path: Path,
):
    """Bootstrap and self-consistency calls reach the teacher directly, with no
    deadline of their own, and its Modal app scales to zero: the first call
    after an idle window pays the engine start. A timed-out call is retried
    once and a timeout never trips the LM fail-fast, so a timeout shorter than
    the cold start costs a whole extra call. The timeout is 1.5x the longest
    measured Modal chat cold start, rounded up. The Modal proxy cuts any
    upstream request at its own timeout once the engine is up, so the client's
    must be the shorter, or the proxy decides instead."""
    teacher = LLMConfig.from_dict(
        load_shipped_config(path)["llm_config"]
    ).resolve_teacher()
    lm = create_budgeted_dspy_lm(teacher)
    modal_proxy_timeout = (
        inspect.signature(_build_process_proxy_app)
        .parameters["request_timeout"]
        .default
    )

    assert (lm.kwargs["timeout"], lm.num_retries) == (210.0, 1)
    assert lm.kwargs["timeout"] >= 1.5 * LONGEST_MEASURED_MODAL_CHAT_COLD_START_S
    assert modal_proxy_timeout == 300.0
    assert lm.kwargs["timeout"] < modal_proxy_timeout


@pytest.mark.unit
def test_shipped_configs_declare_identical_answer_grounding_budget():
    """The dispatcher raises when this key is absent rather than searching
    unbounded, so a budget carried by one copy alone fails every grounded
    answer in the deployment that renders the other."""
    budgets = [
        load_shipped_config(path).get(GROUNDING_SEARCH_TIMEOUT_KEY) for path in CONFIGS
    ]

    assert budgets == [12.0, 12.0]


@pytest.mark.unit
def test_chart_agents_are_a_subset_of_the_reference_config():
    shipped_agents = set(load_shipped_config(SHIPPED)["agents"])
    chart_agents = set(load_shipped_config(CHART)["agents"])

    assert chart_agents - shipped_agents == set()


GOLDEN_DATASET = (
    REPO_ROOT
    / "charts"
    / "cogniverse"
    / "files"
    / "quality-monitor"
    / "golden_dataset.json"
)


def test_shipped_golden_dataset_passes_the_production_canonicalizer():
    """The quality-monitor sidecar seeds its tenant blob from this shipped file
    at startup, through ``canonicalize_golden_set_ground_truth_rows``. A row the
    canonicalizer rejects exits the sidecar, which leaves the runtime pod at 1/2
    and drops it from the Service endpoints, so the whole search API stops
    answering.
    """
    rows = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    canonical = canonicalize_golden_set_ground_truth_rows(rows)

    assert len(rows) == 125
    assert len(canonical) == 125
    assert [row["query"] for row in canonical[:2]] == [
        rows[0]["query"],
        rows[1]["query"],
    ]


def test_shipped_golden_dataset_carries_no_unusable_rows():
    """Every shipped golden row must be usable as a golden *query*: a
    non-empty query and at least one expected video to score it against."""
    rows = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))

    blank_queries = [
        i for i, row in enumerate(rows, 1) if not str(row.get("query", "")).strip()
    ]
    no_expected = [i for i, row in enumerate(rows, 1) if not row.get("expected_videos")]

    assert blank_queries == []
    assert no_expected == []


def _schema_config_identity(config: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Per profile: (schema_config.schema_name, schema_name) and the label
    schema_config gives the profile's embedding model."""
    return {
        name: (
            (profile["schema_config"].get("schema_name"), profile["schema_name"]),
            (
                profile["schema_config"].get("model_name"),
                profile["embedding_model"],
            ),
        )
        for name, profile in config["backend"]["profiles"].items()
        if "schema_name" in profile.get("schema_config", {})
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [*CONFIGS, REPO_ROOT / "configs" / "examples" / "config.example.json"],
    ids=lambda p: p.name if p.name != "config.json" else p.parent.name,
)
def test_shipped_configs_carry_the_promotion_gate(path: Path):
    """Every shipped config states the gate's three values, and they load."""
    from cogniverse_agents.routing.config import OptimizationTriggersConfig

    section = load_shipped_config(path)["automation_rules"]["optimization_triggers"]

    assert {
        key: section[key]
        for key in (
            "optimization_improvement_threshold",
            "optimization_gap_share",
            "optimization_skip_above_score",
        )
    } == {
        "optimization_improvement_threshold": 0.05,
        "optimization_gap_share": 0.5,
        "optimization_skip_above_score": 0.95,
    }
    triggers = OptimizationTriggersConfig(**section)
    assert (
        triggers.optimization_improvement_threshold,
        triggers.optimization_gap_share,
        triggers.optimization_skip_above_score,
    ) == (0.05, 0.5, 0.95)


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [*CONFIGS, REPO_ROOT / "configs" / "examples" / "config.example.json"],
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_schema_config_names_the_schema_and_model_the_profile_uses(path: Path):
    """``schema_config`` restates the profile's schema and model. A restatement
    that names a schema no file defines, or a model family the profile does not
    serve, misleads every reader of the profile."""
    identity = _schema_config_identity(load_shipped_config(path))

    mismatched_schemas = {
        name: names for name, (names, _) in identity.items() if names[0] != names[1]
    }
    tomoro_labels = {
        name: label
        for name, (_, (label, model)) in identity.items()
        if model == "TomoroAI/tomoro-colqwen3-embed-4b" and label is not None
    }
    assert mismatched_schemas == {}
    assert tomoro_labels == {
        "image_colpali_mv": "ColQwen3",
        "video_colpali_smol500_mv_frame": "ColQwen3",
        "video_colqwen_omni_mv_chunk_30s": "ColQwen3",
    }


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_the_omni_video_profile_is_pinned_whole(path: Path):
    """The EmbeddingGemma 2 chunk profile is the single-vector 6 s video
    profile: segmentation, transcription, service routing and schema settings
    are pinned together so a drift in any one of them is visible here."""
    profile = load_shipped_config(path)["backend"]["profiles"][
        "video_embeddinggemma2_sv_chunk_6s"
    ]

    assert profile == {
        "type": "video",
        "description": (
            "EmbeddingGemma 2 6-second chunk single-vector embeddings, 768-dim. "
            "Each chunk's frames (1 fps) and audio track are embedded together "
            "into one space with text, so a text query retrieves clips directly. "
            "Served by the vllm_omni_embed service."
        ),
        "schema_name": "video_embeddinggemma2_sv_chunk_6s",
        "result_granularity": "source",
        "embedding_model": "google/embeddinggemma-2",
        "pipeline_config": {
            "extract_keyframes": False,
            "transcribe_audio": True,
            "generate_descriptions": False,
            "generate_embeddings": True,
        },
        "strategies": {
            "segmentation": {
                "class": "SingleVectorSegmentationStrategy",
                "params": {
                    "strategy": "chunks",
                    "segment_duration": 6.0,
                    "segment_overlap": 1.0,
                },
            },
            "transcription": {
                "class": "AudioTranscriptionStrategy",
                "params": {"model": "openai/whisper-large-v3-turbo"},
            },
            "description": {"class": "NoDescriptionStrategy", "params": {}},
            "embedding": {"class": "SingleVectorEmbeddingStrategy", "params": {}},
        },
        "embedding_type": "single_vector",
        "model_loader": "embeddinggemma2",
        "modalities": ["video", "audio"],
        "inference_services": {
            "embedding": "vllm_omni_embed",
            "transcription": "vllm_asr",
        },
        "model_specific": {"chunk_duration": 6.0, "chunk_overlap": 1.0},
        "inference_config": {
            "model_type": "embeddinggemma2",
            "backend": "remote",
            "query_encoder": "google/embeddinggemma-2",
        },
        "schema_config": {
            "schema_name": "video_embeddinggemma2_sv_chunk_6s",
            "model_name": "EmbeddingGemma-2",
            "embedding_dim": 768,
            "binary_dim": 96,
            "chunk_based": True,
        },
    }


def test_the_omni_video_schema_indexes_the_profile_dims():
    """The schema stores the float and binary vectors at the widths the
    profile's ``schema_config`` declares, and maps the chunk fields the
    ingestion pipeline feeds."""
    dims = load_shipped_config(SHIPPED)["backend"]["profiles"][
        "video_embeddinggemma2_sv_chunk_6s"
    ]["schema_config"]
    schema = json.loads(
        (
            REPO_ROOT / "configs/schemas/video_embeddinggemma2_sv_chunk_6s_schema.json"
        ).read_text()
    )

    assert schema["name"] == dims["schema_name"]
    assert {f["name"]: f["type"] for f in schema["document"]["fields"]} == {
        "video_id": "string",
        "video_title": "string",
        "source_url": "string",
        "creation_timestamp": "long",
        "segment_id": "int",
        "start_time": "float",
        "end_time": "float",
        "embedding": f"tensor<float>(v[{dims['embedding_dim']}])",
        "embedding_binary": f"tensor<int8>(v[{dims['binary_dim']}])",
        "audio_transcript": "string",
        "entity_ids": "array<string>",
        "relation_ids": "array<string>",
        "claim_ids": "array<string>",
    }
    assert [profile["name"] for profile in schema["rank_profiles"]] == [
        "default",
        "bm25_only",
        "binary_binary",
        "float_float",
        "float_binary",
        "phased",
        "hybrid_float_bm25",
        "hybrid_binary_bm25",
        "hybrid_bm25_binary",
        "hybrid_bm25_float",
    ]
    assert schema["document_mapping"] == {
        "id": "video_id",
        "title": "video_title",
        "text_content": "audio_transcript",
        "created_at": "creation_timestamp",
        "created_at_format": "epoch",
        "embeddings": {"embedding": "embedding"},
        "metadata_fields": {"segment_index": "segment_id"},
    }
