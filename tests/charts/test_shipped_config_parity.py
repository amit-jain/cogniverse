"""Both shipped configs must parse, and must not drift apart.

``charts/cogniverse/files/config.json`` renders into every runtime pod;
``configs/config.json`` feeds local runs. They carry overlapping agent and
modality declarations, and a divergence between them is invisible until a pod
fails ``parse_synthetic_runtime_config`` at startup and crash-loops before
serving. Chart values are rendered by Helm before parsing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from cogniverse_agents.optimizer.golden_set_ground_truth import (
    canonicalize_golden_set_ground_truth_rows,
)
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
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


_SELECTED = {"video": {"profile": "video_colpali_smol500_mv_frame"}}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "set_args", "expected"),
    [
        ((), (), {}),
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
            for key in ("temperature", "max_tokens", "context_window")
        }

    assert (
        bounds(load_shipped_config(SHIPPED))
        == bounds(load_shipped_config(CHART))
        == {"temperature": 0.7, "max_tokens": 2048, "context_window": 4096}
    )


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
