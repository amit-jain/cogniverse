"""get_system_config must hand out an isolated copy.

The system config is cached on the ConfigManager instance and shared across
every reader. Returning the cached instance directly let a caller that mutated
a field (the get-modify-set path) or a nested dict poison the shared cache that
concurrent readers see. get_system_config returns a deep copy so a caller's
mutation is confined to its own copy.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _manager() -> ConfigManager:
    return ConfigManager(store=InMemoryConfigStore())


def test_scalar_field_mutation_does_not_poison_cache():
    cm = _manager()
    a = cm.get_system_config()
    original_port = a.backend_port

    a.backend_port = 99999

    b = cm.get_system_config()
    assert b.backend_port == original_port, "cache poisoned by a caller mutation"


def test_nested_dict_mutation_does_not_poison_cache():
    cm = _manager()
    a = cm.get_system_config()

    a.inference_service_urls["poison"] = "http://evil"

    b = cm.get_system_config()
    assert "poison" not in b.inference_service_urls


def test_get_modify_set_round_trip_persists():
    cm = _manager()
    cfg = cm.get_system_config()
    cfg.inference_service_urls = dict(cfg.inference_service_urls)
    cfg.inference_service_urls["vllm_colpali"] = "http://encoder:8000"
    cm.set_system_config(cfg)

    reread = cm.get_system_config()
    assert reread.inference_service_urls["vllm_colpali"] == "http://encoder:8000"
