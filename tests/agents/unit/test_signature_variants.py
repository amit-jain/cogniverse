"""Unit tests for the signature variant registry + tenant-driven selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from cogniverse_agents.optimizer.signature_variants import (
    DEFAULT_VARIANT_ID,
    SignatureVariant,
    SignatureVariantRegistry,
    variant_qualified_agent_key,
)


@dataclass
class _StubTenantConfig:
    """Minimal stand-in for cogniverse_foundation.TenantConfig."""

    metadata: Optional[Dict[str, Any]] = None


class TestRegistry:
    def test_default_variant_implicit_for_any_agent(self):
        reg = SignatureVariantRegistry()
        variants = reg.list_for_agent("search_agent")
        assert any(v.variant_id == DEFAULT_VARIANT_ID for v in variants)

    def test_register_new_variant(self):
        reg = SignatureVariantRegistry()
        v = reg.register(
            "search_agent",
            "with_jurisdiction",
            description="adds jurisdiction + effective_date",
        )
        assert v.variant_id == "with_jurisdiction"
        assert reg.is_registered("search_agent", "with_jurisdiction")

    def test_re_register_identical_idempotent(self):
        reg = SignatureVariantRegistry()
        first = reg.register("x", "v1", description="d1")
        again = reg.register("x", "v1", description="d1")
        assert first == again

    def test_re_register_different_description_rejected(self):
        reg = SignatureVariantRegistry()
        reg.register("x", "v1", description="original")
        with pytest.raises(ValueError, match="already registered"):
            reg.register("x", "v1", description="changed")

    def test_replace_flag_overrides(self):
        reg = SignatureVariantRegistry()
        reg.register("x", "v1", description="original")
        replaced = reg.register("x", "v1", description="changed", replace=True)
        assert replaced.description == "changed"

    def test_empty_variant_id_rejected(self):
        reg = SignatureVariantRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            reg.register("x", "")

    def test_list_for_agent_sorted(self):
        reg = SignatureVariantRegistry()
        reg.register("x", "zebra")
        reg.register("x", "apple")
        ids = [v.variant_id for v in reg.list_for_agent("x")]
        # default + apple + zebra, sorted alphabetically
        assert ids == sorted(ids)


class TestTenantSelection:
    def test_no_config_returns_default(self):
        reg = SignatureVariantRegistry()
        assert reg.selected_for_tenant(None, "search_agent") == DEFAULT_VARIANT_ID

    def test_no_metadata_returns_default(self):
        reg = SignatureVariantRegistry()
        cfg = _StubTenantConfig(metadata=None)
        assert reg.selected_for_tenant(cfg, "search_agent") == DEFAULT_VARIANT_ID

    def test_variant_overrides_per_agent(self):
        reg = SignatureVariantRegistry()
        reg.register("search_agent", "with_jurisdiction")
        cfg = _StubTenantConfig(
            metadata={"signature_variants": {"search_agent": "with_jurisdiction"}}
        )
        assert reg.selected_for_tenant(cfg, "search_agent") == "with_jurisdiction"

    def test_other_agents_unaffected_by_per_agent_override(self):
        reg = SignatureVariantRegistry()
        reg.register("search_agent", "with_jurisdiction")
        cfg = _StubTenantConfig(
            metadata={"signature_variants": {"search_agent": "with_jurisdiction"}}
        )
        # summarizer wasn't overridden → default.
        assert reg.selected_for_tenant(cfg, "summarizer_agent") == DEFAULT_VARIANT_ID

    def test_unknown_variant_falls_back_to_default(self, caplog):
        reg = SignatureVariantRegistry()
        cfg = _StubTenantConfig(
            metadata={"signature_variants": {"search_agent": "totally_made_up"}}
        )
        out = reg.selected_for_tenant(cfg, "search_agent")
        assert out == DEFAULT_VARIANT_ID

    def test_malformed_metadata_falls_back_safely(self):
        reg = SignatureVariantRegistry()
        cfg = _StubTenantConfig(metadata={"signature_variants": "not a dict"})
        assert reg.selected_for_tenant(cfg, "search_agent") == DEFAULT_VARIANT_ID


class TestQualifiedAgentKey:
    def test_default_returns_bare_agent_type(self):
        # Critical for back-compat: existing artefacts should keep working
        # without renaming.
        assert variant_qualified_agent_key("search_agent", "default") == "search_agent"
        assert variant_qualified_agent_key("search_agent", "") == "search_agent"

    def test_non_default_carries_variant_suffix(self):
        out = variant_qualified_agent_key("search_agent", "with_jurisdiction")
        assert out == "search_agent::variant=with_jurisdiction"

    def test_unsafe_chars_in_variant_replaced(self):
        # Colons and slashes would break dataset names.
        assert variant_qualified_agent_key("x", "ns:value") == "x::variant=ns_value"
        assert variant_qualified_agent_key("x", "with/slash") == "x::variant=with_slash"

    def test_keys_are_unique_per_variant(self):
        # Two different non-default variants must produce distinct keys.
        k1 = variant_qualified_agent_key("search_agent", "with_jurisdiction")
        k2 = variant_qualified_agent_key("search_agent", "with_temporal_qualifiers")
        assert k1 != k2
        # ...but a re-call with the same variant is stable.
        assert k1 == variant_qualified_agent_key("search_agent", "with_jurisdiction")


class TestVariantDataclass:
    def test_variant_is_frozen(self):
        v = SignatureVariant(agent_type="x", variant_id="y", description="d")
        with pytest.raises(Exception):  # FrozenInstanceError
            v.variant_id = "changed"  # type: ignore[misc]


class TestSelectedForTenantWarning:
    """Only an EXPLICIT unregistered variant warns; the default path is silent."""

    class _Cfg:
        def __init__(self, meta):
            self.metadata = meta

    def test_no_override_returns_default_without_warning(self, caplog):
        import logging

        reg = SignatureVariantRegistry()
        with caplog.at_level(logging.WARNING):
            result = reg.selected_for_tenant(self._Cfg({}), "search_agent")

        assert result == DEFAULT_VARIANT_ID
        assert not any("not registered" in r.message for r in caplog.records)

    def test_explicit_unregistered_variant_warns(self, caplog):
        import logging

        reg = SignatureVariantRegistry()
        cfg = self._Cfg({"signature_variants": {"search_agent": "typo_v2"}})
        with caplog.at_level(logging.WARNING):
            result = reg.selected_for_tenant(cfg, "search_agent")

        assert result == DEFAULT_VARIANT_ID
        assert any("not registered" in r.message for r in caplog.records)
