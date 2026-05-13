"""Phase 6a — SignatureVariantRegistry + variant routing end-to-end.

Pins the shipped registry semantics + the runtime's per-tenant
selection HTTP route:

  * register is idempotent on identical (agent_type, variant_id, description);
    conflicting description raises ValueError with the exact substring;
  * selected_for_tenant falls back to "default" on missing config or
    unregistered selection (+ caplog warning);
  * variant_qualified_agent_key emits the canonical dataset-name suffix;
  * an artifact-manager canary on the default variant does NOT affect
    requests that select a variant_id (the variant has its own state);
  * PUT /admin/.../signature_variants/{agent} stores the selection,
    GET round-trips it (exact dict equality).
"""

from __future__ import annotations

import logging

import httpx
import pytest

from cogniverse_agents.optimizer.signature_variants import (
    DEFAULT_VARIANT_ID,
    SignatureVariantRegistry,
    variant_qualified_agent_key,
)
from tests.e2e.conftest import RUNTIME, skip_if_no_runtime, unique_id

# ---------------------------------------------------------------------------
# 1. register — idempotent for identical defs, raises on conflict
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRegisterVariantIdempotent:
    """Re-registering identical pin returns the same instance; conflict raises."""

    def test_idempotent_and_conflict(self) -> None:
        reg = SignatureVariantRegistry()
        first = reg.register(
            "query_enhancement_agent", "with_jurisdiction", "legal carve-out"
        )
        second = reg.register(
            "query_enhancement_agent", "with_jurisdiction", "legal carve-out"
        )
        # Idempotent re-register returns an EQUAL value (frozen dataclass).
        assert first == second
        assert first.agent_type == "query_enhancement_agent"
        assert first.variant_id == "with_jurisdiction"
        assert first.description == "legal carve-out"

        with pytest.raises(ValueError) as exc:
            reg.register(
                "query_enhancement_agent",
                "with_jurisdiction",
                "different description",
            )
        assert "already registered with a different definition" in str(exc.value), (
            exc.value
        )


# ---------------------------------------------------------------------------
# 2. selected_for_tenant fallback semantics
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestSelectedForTenantFallback:
    """Missing / wrong tenant selection collapses to default with a warning."""

    def test_no_metadata_returns_default(self) -> None:
        reg = SignatureVariantRegistry()
        # Plain dataclass-style stub — selected_for_tenant only reads .metadata.
        cfg = type("StubCfg", (), {"metadata": {}})()
        assert reg.selected_for_tenant(cfg, "search_agent") == DEFAULT_VARIANT_ID

    def test_none_config_returns_default(self) -> None:
        reg = SignatureVariantRegistry()
        assert reg.selected_for_tenant(None, "search_agent") == DEFAULT_VARIANT_ID

    def test_unregistered_selection_falls_back_with_warning(self, caplog) -> None:
        reg = SignatureVariantRegistry()
        cfg = type(
            "StubCfg",
            (),
            {"metadata": {"signature_variants": {"search_agent": "ghost_variant"}}},
        )()
        with caplog.at_level(logging.WARNING):
            chosen = reg.selected_for_tenant(cfg, "search_agent")
        assert chosen == DEFAULT_VARIANT_ID
        # Operators want to see the typo flagged loudly.
        assert any("falling back" in rec.message.lower() for rec in caplog.records), [
            rec.message for rec in caplog.records
        ]

    def test_registered_selection_returns_variant(self) -> None:
        reg = SignatureVariantRegistry()
        reg.register("search_agent", "with_jurisdiction", "legal carve-out")
        cfg = type(
            "StubCfg",
            (),
            {"metadata": {"signature_variants": {"search_agent": "with_jurisdiction"}}},
        )()
        assert reg.selected_for_tenant(cfg, "search_agent") == "with_jurisdiction"


# ---------------------------------------------------------------------------
# 3. variant_qualified_agent_key
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestVariantQualifiedAgentKey:
    """Default variant returns bare agent_type; custom gets ``::variant=`` suffix."""

    def test_default_returns_bare_agent_type(self) -> None:
        assert variant_qualified_agent_key("search_agent", "default") == "search_agent"

    def test_custom_variant_appends_qualifier(self) -> None:
        assert (
            variant_qualified_agent_key("search_agent", "with_jurisdiction")
            == "search_agent::variant=with_jurisdiction"
        )

    def test_special_chars_in_variant_id_sanitized(self) -> None:
        # ":" and "/" are dataset-name-unsafe in some backends; the helper
        # canonicalises them to "_" to keep dataset names stable.
        assert (
            variant_qualified_agent_key("search_agent", "legal:eu/de")
            == "search_agent::variant=legal_eu_de"
        )


# ---------------------------------------------------------------------------
# 4. ArtifactManager.qualified_agent_key isolates variant state from default
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestVariantsHaveSeparateArtifactKeys:
    """ArtifactManager.qualified_agent_key produces distinct keys per variant.

    The downstream ``promote_to_canary`` / ``load_for_request`` paths key on
    this string; if it weren't variant-distinct, a default-variant canary
    would route a variant-selecting request to the wrong dataset.
    """

    def test_qualified_keys_distinct_across_variants(self) -> None:
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        default_key = ArtifactManager.qualified_agent_key(
            "search_agent", DEFAULT_VARIANT_ID
        )
        variant_key = ArtifactManager.qualified_agent_key(
            "search_agent", "with_jurisdiction"
        )
        assert default_key == "search_agent"
        assert variant_key == "search_agent::variant=with_jurisdiction"
        assert default_key != variant_key


# ---------------------------------------------------------------------------
# 5. PUT/GET HTTP round-trip
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestSetVariantViaHTTP:
    """PUT /signature_variants/{agent} stores; GET returns exact dict equality."""

    def test_put_then_get_round_trip(self) -> None:
        tenant_id = unique_id("opt_var") + ":t1"
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            put_resp = client.put(
                f"/admin/tenants/{tenant_id}/signature_variants/query_enhancement_agent",
                json={"variant_id": "with_jurisdiction"},
            )
            assert put_resp.status_code == 200, put_resp.text[:300]
            put_body = put_resp.json()
            assert put_body == {
                "tenant_id": tenant_id,
                "selections": {"query_enhancement_agent": "with_jurisdiction"},
            }

            get_resp = client.get(
                f"/admin/tenants/{tenant_id}/signature_variants",
            )
            assert get_resp.status_code == 200, get_resp.text[:300]
            assert get_resp.json() == put_body

            # Empty variant_id rejected up front.
            bad = client.put(
                f"/admin/tenants/{tenant_id}/signature_variants/query_enhancement_agent",
                json={"variant_id": "   "},
            )
            assert bad.status_code == 400, bad.text[:300]
            assert "variant_id must be non-empty" in bad.json().get("detail", "")
