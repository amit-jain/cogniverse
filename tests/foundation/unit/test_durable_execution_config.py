"""Per-tenant durable-execution enablement config.

Gates whether the long-running optimization workflows checkpoint their
progress. Default off; set per tenant and read by run_triggered_optimization.
"""

from __future__ import annotations

from cogniverse_foundation.config.unified_config import DurableExecutionConfig


def test_default_disabled(config_manager_memory):
    cfg = config_manager_memory.get_durable_execution_config("acme:acme")
    assert cfg.enabled is False
    assert cfg.tenant_id == "acme:acme"


def test_set_then_get_roundtrip(config_manager_memory):
    config_manager_memory.set_durable_execution_config(
        DurableExecutionConfig(enabled=True), tenant_id="acme:acme"
    )
    cfg = config_manager_memory.get_durable_execution_config("acme:acme")
    assert cfg.enabled is True
    assert cfg.tenant_id == "acme:acme"


def test_other_tenant_unaffected(config_manager_memory):
    config_manager_memory.set_durable_execution_config(
        DurableExecutionConfig(enabled=True), tenant_id="acme:acme"
    )
    other = config_manager_memory.get_durable_execution_config("globex:globex")
    assert other.enabled is False


def test_to_from_dict_roundtrip():
    cfg = DurableExecutionConfig(tenant_id="t", enabled=True)
    assert DurableExecutionConfig.from_dict(cfg.to_dict()) == cfg
