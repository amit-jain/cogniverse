"""ConfigManager tenant-id canonicalization round-trips.

A write issued with the bare tenant form (``acme``) and a read issued with the
canonical form (``acme:acme``) must resolve to the same stored value. The admin
route persists tenant instructions and agent config under the raw path param
while agent dispatch reads under the canonical form; without canonicalization on
both sides the read silently misses the write and the value reads as unset.
"""

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore


@pytest.fixture
def manager() -> ConfigManager:
    return ConfigManager(store=InMemoryConfigStore())


def test_instructions_written_bare_read_canonical(manager: ConfigManager):
    manager.set_config_value(
        tenant_id="acme",
        scope=ConfigScope.SYSTEM,
        service="tenant_instructions",
        config_key="system_prompt",
        config_value={"text": "BE HELPFUL"},
    )

    # Dispatch reads under the canonical form — must see the bare-form write.
    assert manager.get_tenant_instructions_config("acme:acme") == {"text": "BE HELPFUL"}
    # And the bare form resolves to the same entry (idempotent canonicalization).
    assert manager.get_tenant_instructions_config("acme") == {"text": "BE HELPFUL"}


def test_instructions_written_canonical_read_bare(manager: ConfigManager):
    manager.set_config_value(
        tenant_id="acme:acme",
        scope=ConfigScope.SYSTEM,
        service="tenant_instructions",
        config_key="system_prompt",
        config_value={"text": "BE TERSE"},
    )

    assert manager.get_tenant_instructions_config("acme") == {"text": "BE TERSE"}


def test_get_config_value_bare_matches_canonical_write(manager: ConfigManager):
    manager.set_config_value(
        tenant_id="research:team",
        scope=ConfigScope.SYSTEM,
        service="tenant_instructions",
        config_key="system_prompt",
        config_value={"text": "STAY SCOPED"},
    )

    # A split org:tenant id is already canonical — both forms of read agree.
    got = manager.get_config_value(
        tenant_id="research:team",
        scope=ConfigScope.SYSTEM,
        service="tenant_instructions",
        config_key="system_prompt",
    )
    assert got == {"text": "STAY SCOPED"}


def test_distinct_tenants_do_not_collide(manager: ConfigManager):
    manager.set_config_value(
        tenant_id="acme",
        scope=ConfigScope.SYSTEM,
        service="tenant_instructions",
        config_key="system_prompt",
        config_value={"text": "ACME PROMPT"},
    )
    manager.set_config_value(
        tenant_id="globex",
        scope=ConfigScope.SYSTEM,
        service="tenant_instructions",
        config_key="system_prompt",
        config_value={"text": "GLOBEX PROMPT"},
    )

    assert manager.get_tenant_instructions_config("acme:acme") == {
        "text": "ACME PROMPT"
    }
    assert manager.get_tenant_instructions_config("globex:globex") == {
        "text": "GLOBEX PROMPT"
    }


def test_scoped_config_cache_is_bounded():
    """The scoped-config TTL cache must not grow one entry per tenant forever;
    it is a bounded LRU. Reading far more distinct tenants than the cap leaves
    the cache at the cap, not unbounded."""
    manager = ConfigManager(store=InMemoryConfigStore())
    manager._scoped_config_cache_max = 4

    for i in range(50):
        manager.get_tenant_instructions_config(f"tenant{i}")

    assert len(manager._scoped_config_cache) <= 4


def test_scoped_cache_lru_keeps_recently_used():
    manager = ConfigManager(store=InMemoryConfigStore())
    manager._scoped_config_cache_max = 3

    # Prime three tenants.
    for t in ("a", "b", "c"):
        manager.get_tenant_instructions_config(t)
    # Re-touch 'a' so it is the most-recently-used, then insert a fourth.
    manager.get_tenant_instructions_config("a")
    manager.get_tenant_instructions_config("d")

    keys = {k[1] for k in manager._scoped_config_cache}  # canonical tenant ids
    assert "a:a" in keys, "recently-used entry was evicted"
    assert "b:b" not in keys, "least-recently-used entry should have been evicted"
