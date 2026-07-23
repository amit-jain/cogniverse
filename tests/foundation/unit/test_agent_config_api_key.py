"""AgentConfig persists the real llm_api_key; only display redacts it.

to_dict() redacted llm_api_key to "***" and the persistence path serialized
through it, so a config saved with a real key reloaded as the literal "***" —
the secret was overwritten by its own placeholder. Persistence now serializes
with redact=False while display keeps the mask.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
)
from cogniverse_foundation.config.manager import ConfigManager
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _config(api_key: str) -> AgentConfig:
    return AgentConfig(
        agent_name="search_agent",
        agent_version="1.0.0",
        agent_description="test",
        agent_url="http://x",
        capabilities=["search"],
        skills=[],
        module_config=ModuleConfig(module_type=DSPyModuleType.PREDICT, signature="S"),
        llm_api_key=api_key,
    )


def test_to_dict_redacts_by_default_but_not_for_persistence():
    cfg = _config("sk-secret-123")
    assert cfg.to_dict()["llm_api_key"] == "***"
    assert cfg.to_dict(redact=True)["llm_api_key"] == "***"
    assert cfg.to_dict(redact=False)["llm_api_key"] == "sk-secret-123"


def test_set_get_agent_config_preserves_api_key():
    cm = ConfigManager(store=InMemoryConfigStore())
    cm.set_agent_config("acme", "search_agent", _config("sk-secret-123"))

    got = cm.get_agent_config("acme", "search_agent")
    assert got is not None
    assert got.llm_api_key == "sk-secret-123", "the key was overwritten by its mask"
