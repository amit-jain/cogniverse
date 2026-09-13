"""Both shipped configs must pass the runtime's strict synthetic startup parse.

``charts/cogniverse/files/config.json`` renders into every runtime pod and is
validated at startup by ``parse_synthetic_runtime_config`` for the system
tenant; a config that fails the parse crash-loops every deployed runtime pod
before it serves a single request. ``configs/config.json`` feeds the same parse
in local runs. The chart configuration is rendered by Helm before parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config
from cogniverse_synthetic.utils import partition_profiles_by_sampleability
from tests.fixtures.shipped_config import load_shipped_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    REPO_ROOT / "configs" / "config.json",
    REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json",
]

EXPECTED_AGENT_MAPPINGS = {
    "VIDEO": "search_agent",
    "DOCUMENT": "document_agent",
    "IMAGE": "image_search_agent",
    "AUDIO": "audio_analysis_agent",
    "CODE": "coding_agent",
    "WIKI": "document_agent",
}


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name + ":" + p.parent.name)
def test_config_passes_system_tenant_startup_parse(path: Path):
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
