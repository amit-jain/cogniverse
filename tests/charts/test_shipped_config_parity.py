"""Both shipped configs must parse, and must not drift apart.

``charts/cogniverse/files/config.json`` renders into every runtime pod;
``configs/config.json`` feeds local runs. They carry overlapping agent and
modality declarations, and a divergence between them is invisible until a pod
fails ``parse_synthetic_runtime_config`` at startup and crash-loops before
serving. Templated Helm values render to absolute URLs, so the parse here
substitutes one for each ``{{ ... }}`` expression.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "configs" / "config.json"
CHART = REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json"
CONFIGS = [SHIPPED, CHART]


def _rendered(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return json.loads(re.sub(r"\{\{[^}]*\}\}", "http://rendered.invalid", raw))


def _mappings(config: dict[str, Any]) -> list[tuple[str, str]]:
    rules = config["synthetic"]["optimizer_configs"]["modality"]["agent_mappings"]
    return [(rule["modality"], rule["agent_name"]) for rule in rules]


def _optimizer_floors(config: dict[str, Any]) -> dict[str, Any]:
    return config["routing"]["optimization_config"]["optimizer_floors"]


def _training_selection(config: dict[str, Any]) -> dict[str, Any]:
    return config["routing"]["optimization_config"]["training_selection"]


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_shipped_config_passes_system_tenant_startup_parse(path: Path):
    parsed = parse_synthetic_runtime_config(_rendered(path), tenant_id=SYSTEM_TENANT_ID)

    assert parsed.backend_config.tenant_id == SYSTEM_TENANT_ID
    assert parsed.backend_config.profiles


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_every_mapped_agent_declares_the_modality_it_is_mapped_for(path: Path):
    config = _rendered(path)
    agents = config["agents"]

    undeclared = [
        (modality, agent_name)
        for modality, agent_name in _mappings(config)
        if modality not in (agents.get(agent_name, {}).get("modalities") or [])
    ]

    assert undeclared == []


@pytest.mark.unit
def test_shared_agents_declare_identical_modalities_across_shipped_configs():
    shipped_agents = _rendered(SHIPPED)["agents"]
    chart_agents = _rendered(CHART)["agents"]

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
def test_shipped_configs_declare_identical_agent_mappings():
    assert _mappings(_rendered(SHIPPED)) == _mappings(_rendered(CHART))


@pytest.mark.unit
def test_shipped_configs_declare_identical_optimizer_floors():
    assert _optimizer_floors(_rendered(SHIPPED)) == _optimizer_floors(_rendered(CHART))


@pytest.mark.unit
def test_shipped_configs_declare_identical_training_selection():
    # Drift guard: the charted config must match the shipped runtime config.
    assert _training_selection(_rendered(SHIPPED)) == _training_selection(
        _rendered(CHART)
    )


@pytest.mark.unit
def test_shipped_training_selection_matches_canonical_block():
    # Canonical pin: the shipped config carries the exact expected values.
    assert _training_selection(_rendered(SHIPPED)) == {
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
        },
    }


@pytest.mark.unit
def test_chart_agents_are_a_subset_of_the_reference_config():
    shipped_agents = set(_rendered(SHIPPED)["agents"])
    chart_agents = set(_rendered(CHART)["agents"])

    assert chart_agents - shipped_agents == set()
