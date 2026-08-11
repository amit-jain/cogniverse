"""Startup parse must accept the config the runtime itself produces.

``parse_synthetic_runtime_config`` validates the SYSTEM_TENANT config at
lifespan startup. That config is not the shipped file: the memory stack
registers its own backend profile through ``add_backend_profile``, which
persists into ConfigStore, so every later boot parses a config carrying
profiles no shipped file contains. A parse that rejects them crash-loops
every runtime pod before it serves a request.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.memory.manager import build_memory_profile
from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config
from cogniverse_synthetic.utils import partition_profiles_by_sampleability

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "config.json"
MEMORY_SCHEMA = "agent_memories"


def _shipped_config() -> dict[str, Any]:
    raw = SHIPPED_CONFIG.read_text(encoding="utf-8")
    return json.loads(re.sub(r"\{\{[^}]*\}\}", "http://rendered.invalid", raw))


def _config_after_memory_init() -> dict[str, Any]:
    config = _shipped_config()
    config["backend"]["profiles"][MEMORY_SCHEMA] = build_memory_profile(
        MEMORY_SCHEMA, 768
    )
    return config


@pytest.mark.unit
def test_startup_parse_accepts_memory_profile_registered_by_runtime():
    parsed = parse_synthetic_runtime_config(
        _config_after_memory_init(), tenant_id=SYSTEM_TENANT_ID
    )

    memory = parsed.backend_config.profiles[MEMORY_SCHEMA]
    assert memory.type == "memory"
    assert memory.schema_name == MEMORY_SCHEMA
    assert memory.embedding_model == "lightonai/DenseOn"
    assert memory.embedding_type == "dense"
    assert memory.schema_config == {"embedding_dims": 768}
    assert memory.extra_config == {
        "model": "lightonai/DenseOn",
        "embedding_dims": 768,
        "encoder": "denseon",
        "strategy": "semantic_search",
    }


@pytest.mark.unit
def test_memory_profile_is_not_a_synthetic_sampling_target():
    parsed = parse_synthetic_runtime_config(
        _config_after_memory_init(), tenant_id=SYSTEM_TENANT_ID
    )

    sampleable, internal = partition_profiles_by_sampleability(
        parsed.backend_config.profiles
    )
    assert set(internal) == {MEMORY_SCHEMA}
    assert MEMORY_SCHEMA not in sampleable
    assert {profile.type.upper() for profile in sampleable.values()} == {
        "VIDEO",
        "DOCUMENT",
        "IMAGE",
        "AUDIO",
        "CODE",
        "WIKI",
    }


@pytest.mark.unit
def test_parse_rejects_profile_whose_known_field_has_the_wrong_type():
    config = _config_after_memory_init()
    config["backend"]["profiles"][MEMORY_SCHEMA]["schema_config"] = "768"

    with pytest.raises(ValueError) as excinfo:
        parse_synthetic_runtime_config(config, tenant_id=SYSTEM_TENANT_ID)

    assert "backend.profiles.agent_memories.schema_config must be dict, got str" in str(
        excinfo.value
    )


@pytest.mark.unit
def test_parse_rejects_profile_missing_schema_name():
    config = _config_after_memory_init()
    del config["backend"]["profiles"][MEMORY_SCHEMA]["schema_name"]

    with pytest.raises(ValueError) as excinfo:
        parse_synthetic_runtime_config(config, tenant_id=SYSTEM_TENANT_ID)

    assert (
        "backend.profiles.agent_memories is missing required key 'schema_name'"
        in str(excinfo.value)
    )


@pytest.mark.unit
def test_parse_rejects_config_with_no_sampleable_profile():
    config = _shipped_config()
    config["backend"]["profiles"] = {
        MEMORY_SCHEMA: build_memory_profile(MEMORY_SCHEMA, 768)
    }
    config["backend"]["default_profiles"] = {}

    with pytest.raises(ValueError) as excinfo:
        parse_synthetic_runtime_config(config, tenant_id=SYSTEM_TENANT_ID)

    assert "no backend profile has a synthetic-sampleable modality" in str(
        excinfo.value
    )
