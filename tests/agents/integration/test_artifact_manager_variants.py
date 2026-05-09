"""C.6 wiring — variant_qualified_agent_key reaches the dataset wire.

Without this test, the SignatureVariantRegistry was orphan: ``variant_id``
was registered in metadata but never reached the artefact storage layer,
so two variants of the same agent shared a single dataset and the
variant story had no observable effect.

This test verifies, against a real Phoenix instance:

  * the public ``ArtifactManager.qualified_agent_key`` helper produces
    the documented dataset-key shape (default → bare agent_type;
    non-default → ``agent_type::variant=<id>``);
  * round-tripping prompts through real Phoenix using the qualified key
    lands in two distinct datasets — a non-default variant cannot read
    a default-variant's prompts and vice versa;
  * back-compat: callers that pass the bare agent_type (the existing
    behaviour, pre-C.6) continue to read/write the *same* dataset as
    callers who explicitly pass ``variant_id="default"``.
"""

from __future__ import annotations

import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.optimizer.signature_variants import DEFAULT_VARIANT_ID
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def manager(phoenix_container) -> ArtifactManager:
    """ArtifactManager wired to the docker-managed Phoenix container.

    `phoenix_container` is a session-scoped fixture in tests/conftest.py
    that boots a Phoenix instance on port 16006; if Docker isn't
    available it raises rather than skipping silently.
    """
    tenant_id = f"c6_int_{uuid.uuid4().hex[:8]}"
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": "http://localhost:16006",
            "grpc_endpoint": "localhost:14317",
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


class TestQualifiedKeyHelper:
    """Pure-function checks; no Phoenix required."""

    def test_default_variant_returns_bare_agent_type(self):
        # Back-compat invariant — datasets saved before C.6 must keep
        # working when the loader switches to qualified_agent_key.
        assert (
            ArtifactManager.qualified_agent_key("search_agent", DEFAULT_VARIANT_ID)
            == "search_agent"
        )
        assert ArtifactManager.qualified_agent_key("search_agent") == "search_agent"

    def test_non_default_variant_carries_suffix(self):
        out = ArtifactManager.qualified_agent_key("search_agent", "with_jurisdiction")
        assert out == "search_agent::variant=with_jurisdiction"

    def test_unsafe_chars_sanitized(self):
        # Colons / slashes would break Phoenix dataset names.
        assert ArtifactManager.qualified_agent_key("x", "ns:bad") == "x::variant=ns_bad"


@pytest.mark.asyncio
class TestVariantDatasetIsolation:
    async def test_two_variants_produce_distinct_datasets(
        self, manager: ArtifactManager
    ):
        # Save under the bare agent_type — this is the legacy path.
        await manager.save_prompts("search_agent", {"system": "DEFAULT_VARIANT_PROMPT"})
        # Save under a non-default variant key.
        variant_key = ArtifactManager.qualified_agent_key(
            "search_agent", "with_jurisdiction"
        )
        await manager.save_prompts(variant_key, {"system": "WITH_JURISDICTION_PROMPT"})

        # Two distinct datasets must exist on the Phoenix side.
        default_loaded = await manager.load_prompts("search_agent")
        variant_loaded = await manager.load_prompts(variant_key)

        assert default_loaded == {"system": "DEFAULT_VARIANT_PROMPT"}, (
            "default-variant load must not pick up the non-default variant's "
            "prompts — datasets are not isolated"
        )
        assert variant_loaded == {"system": "WITH_JURISDICTION_PROMPT"}, (
            "non-default-variant load must not pick up the default's "
            "prompts — datasets are not isolated"
        )

    async def test_default_variant_id_aliases_bare_agent_type(
        self, manager: ArtifactManager
    ):
        # Save through the bare agent_type, load through the qualified key
        # with default variant — must hit the same dataset (back-compat).
        await manager.save_prompts("search_agent", {"system": "BARE_WRITE"})
        qualified_default = ArtifactManager.qualified_agent_key(
            "search_agent", DEFAULT_VARIANT_ID
        )
        loaded = await manager.load_prompts(qualified_default)
        assert loaded == {"system": "BARE_WRITE"}, (
            "qualified_agent_key with default variant must read the same "
            "dataset as the bare agent_type — otherwise pre-C.6 datasets "
            "are silently abandoned on upgrade"
        )

    async def test_third_variant_independent_of_first_two(
        self, manager: ArtifactManager
    ):
        # Three variants → three datasets, no cross-contamination.
        v1 = ArtifactManager.qualified_agent_key("search_agent", "with_jurisdiction")
        v2 = ArtifactManager.qualified_agent_key(
            "search_agent", "with_temporal_qualifiers"
        )

        await manager.save_prompts("search_agent", {"system": "BASE"})
        await manager.save_prompts(v1, {"system": "JURIS"})
        await manager.save_prompts(v2, {"system": "TIME"})

        assert (await manager.load_prompts("search_agent")) == {"system": "BASE"}
        assert (await manager.load_prompts(v1)) == {"system": "JURIS"}
        assert (await manager.load_prompts(v2)) == {"system": "TIME"}

    async def test_variant_demonstrations_isolated(self, manager: ArtifactManager):
        # Same isolation must hold for demonstrations too — otherwise an
        # operator who ships a variant with new demos would corrupt the
        # default variant's bootstrapped few-shot set.
        v1 = ArtifactManager.qualified_agent_key("search_agent", "v1_demos")
        await manager.save_demonstrations(
            "search_agent",
            [{"input": "q-base", "output": "a-base"}],
        )
        await manager.save_demonstrations(
            v1,
            [{"input": "q-v1", "output": "a-v1"}],
        )
        base_demos = await manager.load_demonstrations("search_agent")
        v1_demos = await manager.load_demonstrations(v1)
        assert base_demos is not None and v1_demos is not None
        assert {d["input"] for d in base_demos} == {"q-base"}
        assert {d["input"] for d in v1_demos} == {"q-v1"}
